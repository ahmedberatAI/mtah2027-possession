"""Round 6, step 0 (docs/RAPOR.md): where is spellF1 lost? AGGREGATE output only.

Oracle ladder (cumulative; each rung makes one more component perfect, starting from v7):
  v7            v7 team probabilities + learned dead-ball model, v7 decoder (B2 soft minimum, pen 12)
  +D            true dead-ball mask
  +restart      true team for 2 s after every true restart
  +timing       predicted in-play team changes moved onto a true change to the same team within +-1 s
  +short        missed true spells shorter than 4 s inserted
  +spurious     predicted spells with no same-team true spell at IoU >= 0.3 replaced by the truth
Error budget (v7 and v7 with true D): the lost F1 credit (FN + FP over the five IoU thresholds) split into
missed true spells (no same-team overlap), partial overlaps (overlap below the threshold) and spurious
predictions (no same-team overlap), by spell duration. Latency of found in-play team changes, ball seen vs
missing at the true change.

Sets: DFL-trk (v7 out-of-fold probabilities), Metrica-ctrl and SkillCorner-phase (all-DFL v7 model).

  python -I src/exp_oracle.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402

pm = td.pm
TAG = "%s_native_v3" % bench._safe(model_v7.VERSION)
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))
EPS = 1e-4
BUCKETS = ((0, 2), (2, 4), (4, 8), (8, 15), (15, 1e9))


def probs(src):
    if src == "dfl@v3":
        return bench._load(os.path.join(bench.CACHE, "oof", TAG + ".pkl")), bench.dead_probs(src, "base")
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", TAG + ".txt"))
    ph = [bench.predict_team(mdl, f) for f in bench.features(model_v7.VERSION, src)]
    pdl = bench.dead_ext("base", "v3", src) if src.startswith("sc") else bench.dead_probs(src, "base")
    return ph, pdl


def spells(lab):
    return pm.possession_spells(lab, [(0, len(lab))])


def matched(gs, ps, thr):
    """Greedy one-to-one matching as the official scorer: (gt matched flags, pred matched flags)."""
    mg = np.zeros(len(gs), bool)
    mp = np.zeros(len(ps), bool)
    for iou, i, j in pm.spell_overlaps(gs, ps):
        if iou < thr:
            break
        if not mg[i] and not mp[j]:
            mg[i] = mp[j] = True
    return mg, mp


def overlapping(a_sp, b_sp):
    """Per spell of a: whether any same-team spell of b overlaps it."""
    out = np.zeros(len(a_sp), bool)
    for i, (a, b, k) in enumerate(a_sp):
        out[i] = any(k2 == k and d > a and c < b for c, d, k2 in b_sp)
    return out


def changes(lab):
    t = np.flatnonzero((lab[1:] < 2) & (lab[:-1] < 2) & (lab[1:] != lab[:-1])) + 1
    return t


def snap_timing(pred, gt, tol=25):
    out = pred.copy()
    gtc = changes(gt)
    for t in changes(pred):
        c = gtc[(np.abs(gtc - t) <= tol) & (gt[gtc] == pred[t])]
        if not len(c):
            continue
        g = int(c[np.argmin(np.abs(c - t))])
        if g > t:
            out[t:g] = pred[t - 1]
        elif g < t:
            out[g:t] = pred[t]
    return out


def insert_short(pred, gt, max_len=100):
    out = pred.copy()
    gs, ps = spells(gt), spells(pred)
    mg, _ = matched(gs, ps, 0.5)
    for (a, b, k), m in zip(gs, mg):
        if not m and b - a < max_len:
            out[a:b] = gt[a:b]
    return out


def drop_spurious(pred, gt):
    out = pred.copy()
    gs, ps = spells(gt), spells(pred)
    _, mp = matched(gs, ps, 0.3)
    for (a, b, k), m in zip(ps, mp):
        if not m:
            out[a:b] = gt[a:b]
    return out


def force_restart(ph, gt, k=50):
    ph = ph.copy()
    r = np.flatnonzero((gt[:-1] == 2) & (gt[1:] < 2)) + 1
    for t in r:
        ph[t:t + k] = np.where(gt[t] == 0, 1 - EPS, EPS)
    return ph


def budget(pairs):
    """Lost F1 credit split, summed over the five IoU thresholds and all halves, by duration bucket."""
    tab = {}
    tot = {"tp": 0, "n": 0}
    for gt, pred in pairs:
        gs, ps = spells(gt), spells(pred)
        g_ov, p_ov = overlapping(gs, ps), overlapping(ps, gs)
        for thr in pm.SPELL_IOUS:
            mg, mp = matched(gs, ps, thr)
            tot["tp"] += int(mg.sum())
            tot["n"] += len(gs) + len(ps)
            for (a, b, k), m, ov in zip(gs, mg, g_ov):
                if not m:
                    key = ("missed" if not ov else "partial (true)", _bucket(b - a))
                    tab[key] = tab.get(key, 0) + 1
            for (a, b, k), m, ov in zip(ps, mp, p_ov):
                if not m:
                    key = ("spurious" if not ov else "partial (pred)", _bucket(b - a))
                    tab[key] = tab.get(key, 0) + 1
    lost = sum(tab.values())
    return tab, lost, tot


def _bucket(n):
    s = n / 25.0
    for lo, hi in BUCKETS:
        if lo <= s < hi:
            return "%g-%gs" % (lo, hi) if hi < 1e9 else "%g+s" % lo
    return "?"


def latency(pairs, seen_list, tol=25):
    res = {"seen": [], "missing": []}
    found = {"seen": [0, 0], "missing": [0, 0]}
    for (gt, pred), seen in zip(pairs, seen_list):
        pc = changes(pred)
        for g in changes(gt):
            key = "seen" if seen[g] else "missing"
            found[key][1] += 1
            c = pc[(np.abs(pc - g) <= tol) & (pred[pc] == gt[g])]
            if len(c):
                found[key][0] += 1
                res[key].append(int(c[np.argmin(np.abs(c - g))] - g))
    return res, found


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, conv, dec in SETS:
        bl, gts = bench.blocks(src), bench.labels(src, conv)
        phl, pdl = probs(src)
        sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
        oracle_d = [np.where(g == 2, 1 - EPS, EPS) for g in gts]
        dec_ = lambda pds, phs: [sp.decode(td.emissions(a, b, 1.0, 1.0)).astype(np.int8) for a, b in zip(pds, phs)]
        ladder = [("v7", dec_(pdl, phl))]
        ladder.append(("+D", dec_(oracle_d, phl)))
        ladder.append(("+restart", dec_(oracle_d, [force_restart(p, g) for p, g in zip(phl, gts)])))
        ladder.append(("+timing", [snap_timing(p, g) for p, g in zip(ladder[-1][1], gts)]))
        ladder.append(("+short", [insert_short(p, g) for p, g in zip(ladder[-1][1], gts)]))
        ladder.append(("+spurious", [drop_spurious(p, g) for p, g in zip(ladder[-1][1], gts)]))
        print("\n=== %s vs %s ===" % (src, conv))
        prev = None
        for name, preds in ladder:
            r = bench.spell_by_group(bl, gts, preds)
            arr = np.array([r[k] for k in sorted(r)])
            v = arr[:, 0].mean()
            print("  ladder %-10s spellF1 %.4f (%s) macroF1 %.4f  pred/true spells %d/%d" % (
                name, v, "%+.4f" % (v - prev) if prev is not None else "      ", arr[:, 1].mean(), arr[:, 3].sum(),
                arr[:, 2].sum()), flush=True)
            prev = v
        for name, preds in ladder[:2]:
            tab, lost, tot = budget(list(zip(gts, preds)))
            print("  budget %-4s lost credit %d of %d (F1 = 2TP/n = %.4f):" % (name, lost, tot["n"], 2 * tot["tp"] / tot["n"]))
            for kind in ("missed", "partial (true)", "partial (pred)", "spurious"):
                cells = [(b, tab.get((kind, b), 0)) for b in [_bucket(int(lo * 25)) for lo, _ in BUCKETS]]
                s = sum(c for _, c in cells)
                print("    %-15s %5.1f%% | %s" % (kind, 100 * s / max(1, lost), " ".join("%s %d" % c for c in cells)))
        seen = [np.isfinite(b["B"][:, 0]) for b in bl]
        for name, preds in ladder[:2]:
            res, found = latency(list(zip(gts, preds)), seen)
            cells = []
            for key in ("seen", "missing"):
                v = np.array(res[key])
                if len(v):
                    cells.append("ball %s: found %d/%d, median %+.0f, mean %+.1f, p25 %+.0f, p75 %+.0f frames" % (
                        key, found[key][0], found[key][1], np.median(v), v.mean(), np.percentile(v, 25), np.percentile(v, 75)))
            print("  latency %-4s in-play team changes (pred - true): %s" % (name, " | ".join(cells)))
        ev = pm.evaluate(np.concatenate([np.concatenate([g, np.full(10, 3, np.int8)]) for g in gts]).astype(np.int8),
                         np.concatenate([np.concatenate([p, np.full(10, 3, np.int8)]) for p in ladder[0][1]]).astype(np.int8))
        tr = ev["transitions"]
        print("  official transition latency (v7, all types, typed by destination): " + " | ".join(
            "tol %d: F1 %.3f mean %+.1f median %+.1f" % (t, tr[t]["f1"], tr[t]["mean_latency"] or 0,
                                                         tr[t]["median_latency"] or 0) for t in sorted(tr)))


if __name__ == "__main__":
    main()
