"""Round 31 (docs/RAPOR.md, pre-registered): supervised boundary regression (round 30, idea B3 #1). For every in-play
change of team in v15's final labels (k -> 1-k, no dead ball between), one LightGBM predicts the signed offset to its
true twin (the nearest true change of the same direction within +-2 s, matched one-to-one, nearest first) and a second
one whether such a twin exists, from a +-2 s window canonicalised to the incoming / outgoing team: v15's raw and
smoothed team logit, C1's control / gains / loses / touch heads, the ball's speed and missingness, each team's
nearest-player distance, tracking touches and the nearest holder, the neighbouring spells. A change moves by the
rounded predicted offset when P(twin) >= 0.5 and |offset| >= 2 frames, never across a neighbouring boundary and leaving
both spells >= 8 frames. Training pools both dead channels and the three sources (labels in the target convention: DFL
evctrl, Metrica ctrl, SkillCorner evctrl), leave-one-match-out. Decision as every round: v15 (C1abETmS) against
v15 + boundary regression on the ten contexts, primary >= 0.005, guards >= -0.01. AGGREGATE output only.

  python -I src/exp_bound.py screen     v10 seed 0, verifier seed 0; go when DFL evctrl and Metrica ctrl each gain
                                        >= +0.003 (mean of the two dead channels)
  python -I src/exp_bound.py eval       3 v10 seeds x verifier seeds 0, 1; decision rule; SkillCorner evctrl (info)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import short_specialist as ss  # noqa: E402

WIN = 50
OFFS = np.array((-50, -25, -12, -6, -3, 0, 3, 6, 12, 25, 50))
MIN_LEN = 8
TRAIN_CONV = {"dfl@v3": "evctrl", "metrica@v3": "ctrl", "sc@v3": "evctrl"}
PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0, deterministic=True,
              force_row_wise=True)
ROUNDS = 300
NAMES = ("v15", "v15 + BR")


def changes(lab):
    """(frames, incoming teams) of the in-play changes of team."""
    lab = np.asarray(lab)
    f = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] < 2) & (lab[:-1] < 2)) + 1
    return f, lab[f]


def twins(pred, gt):
    """Signed offset of each predicted in-play change to its true twin (NaN when none)."""
    pf, pk = changes(pred)
    tf, tk = changes(gt)
    pairs = []
    for i, (f, k) in enumerate(zip(pf, pk)):
        lo, hi = np.searchsorted(tf, f - WIN), np.searchsorted(tf, f + WIN, side="right")
        pairs += [(abs(int(tf[j]) - int(f)), i, j) for j in range(lo, hi) if tk[j] == k]
    out = np.full(len(pf), np.nan)
    used = set()
    for d, i, j in sorted(pairs):
        if np.isnan(out[i]) and j not in used:
            out[i] = float(tf[j] - pf[i])
            used.add(j)
    return out


def _nearest(frames, f):
    """Offset to the nearest of frames within +-WIN (NaN when none)."""
    if not len(frames):
        return np.nan
    j = np.searchsorted(frames, f)
    best = np.nan
    for x in frames[max(0, j - 1):j + 1]:
        if abs(x - f) <= WIN and (np.isnan(best) or abs(x - f) < abs(best)):
            best = float(x - f)
    return best


def block_rows(pred, lr, ls, o, c):
    """Feature rows of the in-play changes of one decoded half; lr / ls: raw / smoothed logit of P(home); o: C1's
    (T, 9) outputs; c: exp_short_angles.frame_cues."""
    pred = np.asarray(pred)
    n = len(pred)
    pf, pk = changes(pred)
    bounds = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    dk = [np.minimum(c["dk"][0], 60.0), np.minimum(c["dk"][1], 60.0)]
    sp, miss = c["sp"], c["miss"].astype(np.float32)
    touch_t = np.asarray(c["touch_t"])
    touch_k = np.asarray(c["touch_k"])
    hold = [np.flatnonzero((c["near_team"] == k) & (dk[k] <= 1.5)) for k in (0, 1)]
    tch = [np.sort(touch_t[touch_k == k]) for k in (0, 1)]
    rows = []
    for f, inc in zip(pf, pk):
        out = 1 - inc
        sg = 1.0 if inc == 0 else -1.0
        idx = np.clip(f + OFFS, 0, n - 1)
        r = {"f": int(f), "inc": int(inc)}
        for off, i in zip(OFFS, idx):
            r["lr%d" % off] = sg * lr[i]
            r["ls%d" % off] = sg * ls[i]
            r["ci%d" % off], r["co%d" % off], r["cd%d" % off] = o[i, inc], o[i, out], o[i, 2]
            r["gi%d" % off], r["lo%d" % off] = o[i, 3 + inc], o[i, 5 + out]
            r["ti%d" % off], r["to%d" % off] = o[i, 7 + inc], o[i, 7 + out]
            r["di%d" % off], r["do%d" % off] = dk[inc][i], dk[out][i]
            r["sp%d" % off], r["ms%d" % off] = sp[i], miss[i]
        lo, hi = max(0, f - WIN), min(n, f + WIN + 1)
        for name, col in (("gi", 3 + inc), ("lo", 5 + out), ("ti", 7 + inc), ("to", 7 + out)):
            j = int(np.argmax(o[lo:hi, col]))
            r[name + "_arg"], r[name + "_max"] = lo + j - f, float(o[lo + j, col])
        for name, x in (("cross_raw", sg * lr[lo:hi]), ("cross_smo", sg * ls[lo:hi])):
            cr = np.flatnonzero((x[:-1] <= 0) & (x[1:] > 0)) + 1 + lo
            r[name] = _nearest(cr, f)
        sl = np.convolve(sg * lr[lo:hi], np.ones(5) / 5, mode="same")
        r["slope_arg"] = lo + int(np.argmax(np.diff(sl, prepend=sl[0]))) - f
        r["tch_in"], r["tch_out"] = _nearest(tch[inc], f), _nearest(tch[out], f)
        r["hold_in"], r["hold_out"] = _nearest(hold[inc], f), _nearest(hold[out], f)
        b = int(np.searchsorted(bounds, f))
        prev_b = bounds[b - 1] if b > 0 else 0
        next_b = bounds[b + 1] if b + 1 < len(bounds) else n
        r["prev_len"], r["next_len"] = f - prev_b, next_b - f
        r["prev_d"] = float(prev_b > 0 and pred[prev_b - 1] == 2)
        r["next_d"] = float(next_b < n and pred[next_b] == 2)
        r["miss_frac"] = float(miss[max(0, f - 25):f + 26].mean())
        rows.append(r)
    return rows


def keys(rows):
    return [k for k in rows[0] if k not in ("f", "inc", "y", "group", "src", "i", "dead")]


def shift(pred, rows, off, pv):
    """Moves each in-play change of pred by its predicted offset (rows in change order)."""
    out = np.asarray(pred).copy()
    bounds = np.flatnonzero(out[1:] != out[:-1]) + 1
    lower = 0
    moved = {}
    for r, d, p in zip(rows, off, pv):
        f = r["f"]
        b = int(np.searchsorted(bounds, f))
        prev_b = max(lower, bounds[b - 1] if b > 0 else 0)
        next_b = bounds[b + 1] if b + 1 < len(bounds) else len(out)
        g = f
        if p >= 0.5 and abs(d) >= 2 and prev_b + MIN_LEN <= next_b - MIN_LEN:
            target = f + int(round(d))
            g = int(np.clip(target, prev_b + MIN_LEN, next_b - MIN_LEN))
            if (g - f) * (target - f) <= 0:            # never pushed the other way by the length limits
                g = f
        moved[f] = g
        lower = g
    for f, g in moved.items():
        if g < f:
            out[g:f] = pred[f]
        elif g > f:
            out[f:g] = pred[f - 1]
    return out


def collect(k, q):
    """Change rows of v15's finals (v10 seed k, verifier seed q) on the three sources and both dead channels, with
    targets from the target convention, plus the finals themselves."""
    import exp_blend_weight as bw
    import exp_kpu
    exp_kpu.use("C1abETmS")
    rows, finals = [], {}
    for src, *_ in ev.SOURCES:
        blocks = bench.blocks(src)
        gts = bench.labels(src, TRAIN_CONV[src])
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
        outs = exp_kpu.outputs("C1", src)[ss.SEEDS[k]]
        raws = [0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(es.blend(es.v10_raw(k, src), members[k], 0.5), outs)]
        smos = exp_smooth.smooth(raws, 25)
        cues = [ea.frame_cues(b) for b in blocks]
        for dname in ev.DEADS:
            _, fin = bw._verified(src, k, dname, q, lambda s2: ev._rows4(s2, k, dname), "probs4")
            finals[(src, dname)] = fin
            for i, (b, pr, gt) in enumerate(zip(blocks, fin, gts)):
                rs = block_rows(pr, ea.logit(raws[i]), ea.logit(smos[i]), np.asarray(outs[i], np.float32), cues[i])
                y = twins(pr, gt)
                for r, yy in zip(rs, y):
                    r.update(y=yy, group="%s:%s" % (src, b["group"]), src=src, i=i, dead=dname)
                rows += rs
        del cues
        bench._blocks.clear()
        bench._labels.clear()
    return rows, finals


def lomo(rows):
    """Leave-one-match-out predicted offsets and P(twin)."""
    import lightgbm as lgb
    ks = keys(rows)
    X = np.array([[r[c] for c in ks] for r in rows], np.float32)
    y = np.array([r["y"] for r in rows])
    g = np.array([r["group"] for r in rows])
    valid = np.isfinite(y)
    off, pv = np.zeros(len(rows)), np.zeros(len(rows))
    for grp in np.unique(g):
        tr, te = g != grp, g == grp
        reg = lgb.train(dict(PARAMS, objective="huber", alpha=12.0), lgb.Dataset(X[tr & valid], y[tr & valid]), ROUNDS)
        clf = lgb.train(dict(PARAMS, objective="binary"), lgb.Dataset(X[tr], valid[tr].astype(int)), ROUNDS)
        off[te], pv[te] = reg.predict(X[te]), clf.predict(X[te])
    return off, pv


def refined(rows, off, pv, finals):
    """v15 + BR finals per (src, dead)."""
    by = {}
    for n, r in enumerate(rows):
        by.setdefault((r["src"], r["dead"], r["i"]), []).append(n)
    out = {}
    for (src, dname), fin in finals.items():
        out[(src, dname)] = [shift(pr, [rows[n] for n in by.get((src, dname, i), [])],
                                   [off[n] for n in by.get((src, dname, i), [])],
                                   [pv[n] for n in by.get((src, dname, i), [])]) for i, pr in enumerate(fin)]
    return out


def run(pairs):
    import exp_refine as er
    cells, info, diag = {}, {}, []
    for k, q in pairs:
        rows, finals = collect(k, q)
        off, pv = lomo(rows)
        new = refined(rows, off, pv, finals)
        y = np.array([r["y"] for r in rows])
        ok = np.isfinite(y)
        moved = (pv >= 0.5) & (np.abs(off) >= 2)
        diag.append((ok.mean(), np.median(np.abs(y[ok])), np.median(np.abs(y[ok] - np.round(off[ok]))), moved.mean(),
                     np.mean(np.abs(y[ok & moved] - np.round(off[ok & moved])) < np.abs(y[ok & moved]))))
        for src, convs, *_ in ev.SOURCES:
            blocks = bench.blocks(src)
            extra = ("evctrl",) if src.startswith("sc") else ()
            labs = {c: bench.labels(src, c) for c in convs + extra}
            for dname in ev.DEADS:
                for name, preds in zip(NAMES, (finals[(src, dname)], new[(src, dname)])):
                    for c in convs + extra:
                        (info if c in extra else cells).setdefault((src, c, dname, name), []).append(
                            er.by_match(blocks, labs[c], preds))
            bench._blocks.clear()
            bench._labels.clear()
        print("  seed %d / verifier %d done" % (k, q), flush=True)
    d = np.mean(diag, 0)
    print("changes with a true twin %.3f; |offset| median %.1f -> after the model %.1f frames; moved %.3f of changes, "
          "of which closer %.3f" % tuple(d), flush=True)
    return cells, info


def screen():
    import exp_refine as er
    cells, _ = run([(0, 0)])
    mean, _ = er.summarize(cells, list(NAMES), ref=NAMES[0])
    for src, conv in (("dfl@v3", "evctrl"), ("metrica@v3", "ctrl")):
        dl = [np.mean([mean[(src, conv, d, NAMES[1])][m] - mean[(src, conv, d, NAMES[0])][m]
                       for m in mean[(src, conv, d, NAMES[0])]]) for d in ev.DEADS]
        print("screen %s %s: %+.4f (learned %+.4f, true %+.4f) -> %s" % (
            src.split("@")[0], conv, np.mean(dl), dl[0], dl[1], "go" if np.mean(dl) >= 0.003 else "stop"))


def evaluate():
    import exp_blend_weight as bw
    import exp_refine as er
    cells, info = run([(k, q) for k in es.SEEDS for q in (0, 1)])
    mean, _ = er.summarize(cells, list(NAMES), ref=NAMES[0])
    bw.regret_boot(mean, list(NAMES), NAMES[0], NAMES[1])
    bw.decide(mean, NAMES[0], NAMES[1])
    print("\ninformation only, not in the decision rule:", flush=True)
    er.summarize(info, list(NAMES), ref=NAMES[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"screen": screen, "eval": evaluate}[sys.argv[1]]()
