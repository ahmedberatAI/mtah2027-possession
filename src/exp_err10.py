"""Round 10 diagnostic (before new training): where does v10 lose spell F1? AGGREGATE output only.

Contexts: DFL-trk, Metrica-ctrl, SkillCorner phase (held-out matches, V3 inputs), each with v10's learned D and the
true D, and the BAR-ATH public window (candidate_v10w.csv). Per context:
  ladder   cumulative oracle rungs from v10: +D (true dead mask), +restart (true team for 2 s after each true
           restart), +timing (predicted in-play changes moved onto a true change to the same team within +-1 s),
           +short (missed true spells < 4 s inserted), +spurious (unmatched predicted spells replaced by the truth);
           and the last three alone on v10 (learned / true D)
  modes    true spells at IoU 0.5: ok / wrong team / dead / split / merged / shifted, by duration, by how the spell
           starts (restart = after D, turnover = after the other team) and ends (stoppage / turnover)
  spur     predicted spells without a match at IoU 0.5: inside true D, a flip inside the other team's true spell, a
           fragment of a same-team true spell, mixed
  frames   alive-frame errors (wrong team or D) by situation: time to the nearest true change, ball missing in the
           input, ball speed, contested ball (both teams within 3 m), pitch third of the team in possession -- share
           of alive frames, error rate, share of all alive-frame errors; dead frames called alive by distance to
           the nearest true alive/dead boundary
Metrica-ctrl also: true spells by how they start (event type / subtype of the new team's first event; for turnovers
also the old team's last event), with recall at IoU 0.5 (3 games x 3 noise seeds, so counts are 3x).

  python -I src/exp_err10.py
"""
import os
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import evaluate as ev  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle as eo  # noqa: E402
import metrica_sample as ms  # noqa: E402

EPS = 1e-4
FPS = 25.0
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))
DUR = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9))
MODES = ("wrong team", "dead", "split", "merged", "shifted")
SUBMISSION = os.path.join(ev.ROOT, "submissions", "candidate_v10w.csv")


def dur_bin(n):
    s = n / FPS
    for lo, hi in DUR:
        if lo <= s < hi:
            return "%g-%gs" % (lo, hi) if hi < 1e9 else ">%gs" % lo
    return "?"


DUR_NAMES = [dur_bin(int(lo * FPS)) for lo, _ in DUR]


def spell_modes(gt, pred, thr=0.5):
    """Per true spell (frames, start kind, end kind, mode); per unmatched predicted spell (frames, where)."""
    gs, ps = eo.spells(gt), eo.spells(pred)
    mg, mp = eo.matched(gs, ps, thr)
    out = []
    for (a, b, k), ok in zip(gs, mg):
        prev = gt[a - 1] if a > 0 else 3
        nxt = gt[b] if b < len(gt) else 3
        start = "restart" if prev == 2 else "turnover" if prev in (0, 1) else "edge"
        end = "stoppage" if nxt == 2 else "turnover" if nxt in (0, 1) else "edge"
        if ok:
            mode = "ok"
        else:
            seg = pred[a:b]
            if np.mean(seg == 1 - k) >= 0.5:
                mode = "wrong team"
            elif np.mean(seg == 2) >= 0.5:
                mode = "dead"
            else:
                inside = [(c, d) for c, d, k2 in ps if k2 == k and d > a and c < b]
                if len(inside) >= 2:
                    mode = "split"
                elif inside and (inside[0][0] < a - 0.25 * (b - a) or inside[0][1] > b + 0.25 * (b - a)):
                    mode = "merged"
                else:
                    mode = "shifted"
        out.append((b - a, start, end, mode))
    spur = []
    for (c, d, k), ok in zip(ps, mp):
        if ok:
            continue
        seg = gt[c:d]
        where = ("in true D" if np.mean(seg == 2) >= 0.5 else "flip inside other team's spell"
                 if np.mean(seg == 1 - k) >= 0.5 else "fragment of same-team spell" if np.mean(seg == k) >= 0.5
                 else "mixed")
        spur.append((d - c, where))
    return out, spur


def _dist_to(ix, n):
    if not len(ix):
        return np.full(n, np.inf)
    t = np.arange(n)
    j = np.searchsorted(ix, t)
    lo = np.abs(t - ix[np.clip(j - 1, 0, len(ix) - 1)])
    hi = np.abs(ix[np.clip(j, 0, len(ix) - 1)] - t)
    return np.minimum(lo, hi) / FPS


def situations(b, gt):
    """Per frame situation names from the model's inputs (degraded tracking) and the truth."""
    L, W = b["pitch"]
    P = np.asarray(b["P"], np.float32) * np.array([L, W], np.float32)
    B = np.asarray(b["B"], np.float32) * np.array([L, W], np.float32)
    n = len(gt)
    team = np.asarray(b["team"])
    dt = _dist_to(eo.changes(gt), n)
    dd = _dist_to(np.flatnonzero((gt[1:] == 2) != (gt[:-1] == 2)) + 1, n)
    short = np.zeros(n, bool)
    for a, e, _ in eo.spells(gt):
        if e - a < 4 * FPS:
            short[a:e] = True
    when = np.where(dt <= 1, "<=1 s from a team change", np.where(
        dd <= 1, "<=1 s from a restart / stoppage", np.where(
            short, "inside a short (<4 s) spell", np.where(
                np.minimum(dt, dd) <= 3, "1-3 s from a change", "stable (>3 s from any change)"))))
    miss = ~np.isfinite(B[:, 0])
    v = np.full(n, np.nan)
    if n > 10:
        v[5:-5] = np.linalg.norm(B[10:] - B[:-10], axis=1) / (10 / FPS)
    ball = np.where(miss, "ball missing in input", np.where(~np.isfinite(v), "ball speed unknown", np.where(
        v < 2, "ball < 2 m/s", np.where(v < 8, "ball 2-8 m/s", np.where(v < 15, "ball 8-15 m/s", "ball > 15 m/s")))))
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(P - B[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d0 = np.min(np.where(team[None, :] == 0, dist, np.inf), 1)
    d1 = np.min(np.where(team[None, :] == 1, dist, np.inf), 1)
    cont = np.where(miss, "ball missing in input", np.where((d0 <= 3) & (d1 <= 3), "both teams within 3 m", np.where(
        (d0 <= 3) | (d1 <= 3), "one team within 3 m", "nobody within 3 m")))
    gk = np.asarray(b["gk"], bool)
    plus = {}
    for k in (0, 1):
        m = gk & (team == k)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = np.nanmedian(P[:, m, 0]) if m.any() else np.nan
        plus[k] = bool(x < L / 2) if np.isfinite(x) else True
    owner = np.where(gt < 2, gt, 0)
    xr = np.where(np.array([plus[0], plus[1]])[owner], B[:, 0], L - B[:, 0]) / L
    zone = np.where(miss, "ball missing in input", np.where(xr < 1 / 3, "own third", np.where(
        xr < 2 / 3, "middle third", "final third")))
    return {"when": when, "ball": ball, "contest": cont, "zone": zone, "d_ad": dd}


class Frames:
    def __init__(self):
        self.tab = defaultdict(lambda: np.zeros(4, np.int64))   # frames, errors, wrong team, called D
        self.dead = defaultdict(lambda: np.zeros(2, np.int64))  # dead frames, called alive

    def add(self, sit, gt, pred):
        alive = gt < 2
        wrong = alive & (pred < 2) & (pred != gt)
        called_d = alive & (pred == 2)
        for key in ("when", "ball", "contest", "zone"):
            names = sit[key]
            for name in np.unique(names[alive]):
                m = alive & (names == name)
                self.tab[(key, name)] += [m.sum(), (wrong | called_d)[m].sum(), wrong[m].sum(), called_d[m].sum()]
        dead = gt == 2
        dd = sit["d_ad"]
        for name, m in (("<=1 s from the boundary", dd <= 1), ("1-3 s", (dd > 1) & (dd <= 3)), (">3 s", dd > 3)):
            m = m & dead
            self.dead[name] += [m.sum(), (pred[m] < 2).sum()]

    def report(self, indent="    "):
        tot = sum(v for (k, _), v in self.tab.items() if k == "when")
        print("%salive frames %d, errors %.2f%% (wrong team %.2f%%, called D %.2f%%)" % (
            indent, tot[0], 100 * tot[1] / tot[0], 100 * tot[2] / tot[0], 100 * tot[3] / tot[0]))
        for key in ("when", "ball", "contest", "zone"):
            rows = sorted(((n, v) for (k, n), v in self.tab.items() if k == key), key=lambda r: -r[1][1])
            print("%s%-8s %s" % (indent, key, " | ".join(
                "%s: %.1f%% of frames, err %.1f%%, %.1f%% of errors" % (
                    n, 100 * v[0] / tot[0], 100 * v[1] / max(v[0], 1), 100 * v[1] / max(tot[1], 1)) for n, v in rows)))
        d = self.dead
        nd = sum(v[0] for v in d.values())
        if nd:
            print("%sdead frames called alive: %.2f%% of %d | %s" % (
                indent, 100 * sum(v[1] for v in d.values()) / nd, nd, " | ".join(
                    "%s: %.1f%% of dead frames, called alive %.1f%%, %.1f%% of such errors" % (
                        n, 100 * v[0] / nd, 100 * v[1] / max(v[0], 1),
                        100 * v[1] / max(sum(w[1] for w in d.values()), 1)) for n, v in d.items())))


def modes_report(rows, spur, indent="    "):
    n = len(rows)
    ok = sum(r[3] == "ok" for r in rows)
    print("%strue spells %d, recall@0.5 %.3f; predicted unmatched %d" % (indent, n, ok / max(n, 1), len(spur)))
    for title, idx, names in (("duration", 0, DUR_NAMES), ("start", 1, ("restart", "turnover")),
                              ("end", 2, ("stoppage", "turnover"))):
        cells = []
        for name in names:
            sel = [r for r in rows if (dur_bin(r[0]) if idx == 0 else r[idx]) == name]
            if not sel:
                continue
            miss = [r for r in sel if r[3] != "ok"]
            mc = Counter(r[3] for r in miss)
            cells.append("%s n %d rec %.2f miss %d (%s)" % (
                name, len(sel), 1 - len(miss) / len(sel), len(miss),
                ", ".join("%s %d" % (m, mc[m]) for m in MODES if mc[m])))
        print("%s%-8s %s" % (indent, title, " | ".join(cells)))
    allmiss = Counter(r[3] for r in rows if r[3] != "ok")
    print("%smisses by mode: %s" % (indent, ", ".join("%s %d (%.0f%%)" % (
        m, allmiss[m], 100 * allmiss[m] / max(sum(allmiss.values()), 1)) for m in MODES)))
    sc = Counter(w for _, w in spur)
    sd = Counter(dur_bin(f) for f, _ in spur)
    print("%sunmatched predicted spells: %s | by duration %s" % (indent, ", ".join(
        "%s %d" % kv for kv in sc.most_common()), ", ".join("%s %d" % (d, sd[d]) for d in DUR_NAMES if sd[d])))


def metrica_starts(blocks, gts, preds):
    cache = {}
    rows = []
    for b, gt, pred in zip(blocks, gts, preds):
        parts = b["key"].split("/")
        g, h = int(parts[1][1:]), int(parts[3][1:])
        if g not in cache:
            d = ms.parse(g)
            cache[g] = (ms.events(g), {hh: int(d["h%d_n0" % hh][0]) for hh in (1, 2)})
        evg, n0 = cache[g]
        e = evg[evg["period"] == h]
        st = e["start"].to_numpy() - n0[h]
        tm, ty, sub = e["team"].to_numpy(), e["type"].to_numpy(), e["subtype"].to_numpy()
        gs, ps = eo.spells(gt), eo.spells(pred)
        mg, _ = eo.matched(gs, ps, 0.5)
        for (a, e_, k), ok in zip(gs, mg):
            prev = gt[a - 1] if a > 0 else 3
            kind = "restart" if prev == 2 else "turnover" if prev in (0, 1) else "edge"
            i = np.flatnonzero((tm == k) & (st >= a - 12) & (st < e_))
            first = ("%s %s" % (ty[i[0]], sub[i[0]])).strip() if len(i) else "no event"
            j = np.flatnonzero((tm == 1 - k) & (st < a + 12))
            last = ("%s %s" % (ty[j[-1]], sub[j[-1]])).strip() if len(j) else "no event"
            rows.append((kind, first, last, e_ - a, bool(ok)))
    for kind in ("restart", "turnover"):
        sel = [r for r in rows if r[0] == kind]
        for label, idx in (("new team's first event", 1), ("old team's last event", 2)):
            if kind == "restart" and idx == 2:
                continue
            c = Counter(r[idx] for r in sel)
            cells = []
            other = [0, 0]
            for name, cnt in c.most_common():
                hit = sum(r[4] for r in sel if r[idx] == name)
                if cnt >= 15:
                    cells.append("%s n %d rec %.2f (<4 s %d%%)" % (name, cnt, hit / cnt, round(100 * np.mean(
                        [r[3] < 4 * FPS for r in sel if r[idx] == name]))))
                else:
                    other[0] += cnt
                    other[1] += hit
            if other[0]:
                cells.append("other n %d rec %.2f" % (other[0], other[1] / other[0]))
            print("    %s spells by %s: %s" % (kind, label, " | ".join(cells)))


def score(src, conv, preds):
    return e2.score(src, conv, preds).mean()


def external():
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src, conv, dec in SETS:
        blocks, gts = bench.blocks(src), bench.labels(src, conv)
        _, pdl = eo.probs(src)
        ph = v10[src]
        tdd = [np.where(g == 2, 1 - EPS, EPS) for g in gts]
        print("\n=== %s %s (%d halves) ===" % (src.split("@")[0], conv, len(gts)), flush=True)
        rungs = [("v10", e2.decode_all(dec, pdl, ph)), ("+D", e2.decode_all(dec, tdd, ph))]
        rungs.append(("+restart", e2.decode_all(dec, tdd, [eo.force_restart(p, g) for p, g in zip(ph, gts)])))
        rungs.append(("+timing", [eo.snap_timing(p, g) for p, g in zip(rungs[-1][1], gts)]))
        rungs.append(("+short", [eo.insert_short(p, g) for p, g in zip(rungs[-1][1], gts)]))
        rungs.append(("+spurious", [eo.drop_spurious(p, g) for p, g in zip(rungs[-1][1], gts)]))
        vals = [score(src, conv, p) for _, p in rungs]
        print("  ladder  " + " | ".join("%s %.4f%s" % (n, v, "" if i == 0 else " (%+.4f)" % (v - vals[i - 1]))
                                        for i, ((n, _), v) in enumerate(zip(rungs, vals))), flush=True)
        for dname, base in (("learned D", rungs[0][1]), ("true D", rungs[1][1])):
            b0 = score(src, conv, base)
            alone = [("timing", [eo.snap_timing(p, g) for p, g in zip(base, gts)]),
                     ("short", [eo.insert_short(p, g) for p, g in zip(base, gts)]),
                     ("spurious", [eo.drop_spurious(p, g) for p, g in zip(base, gts)])]
            print("  alone on v10 (%s) %.4f: %s" % (dname, b0, " | ".join(
                "%s %+.4f" % (n, score(src, conv, p) - b0) for n, p in alone)), flush=True)
        for dname, preds in (("learned D", rungs[0][1]), ("true D", rungs[1][1])):
            print("  -- %s" % dname)
            rows, spur, fr = [], [], Frames()
            for b, g, p in zip(blocks, gts, preds):
                r, s = spell_modes(g, p)
                rows += r
                spur += s
                fr.add(situations(b, g), g, p)
            modes_report(rows, spur)
            fr.report()
            if src.startswith("metrica"):
                metrica_starts(blocks, gts, preds)
            sys.stdout.flush()


def public():
    gt = ev.public_gt()
    letters = np.loadtxt(SUBMISSION, delimiter=",", skiprows=1, usecols=1, dtype=str)
    pred = ev.to_codes(letters)
    s0 = ev.score(pred)
    print("\n=== BAR-ATH public window (%s) ===" % os.path.basename(SUBMISSION))
    print("  spellF1 %.4f macroF1 %.4f alive/dead F1 %.4f team acc %.4f, true / predicted spells %d / %d" % (
        s0["spellF1"], s0["macroF1"], s0["alive_dead_f1"], s0["team_acc"], s0["n_gt"], s0["n_pred"]))
    idx = np.flatnonzero(gt < 3)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    halves = {blk["key"]: blk for blk in bench.blocks("barath")}
    rows, spur, fr = [], [], Frames()
    fixed = {k: pred.copy() for k in ("timing", "short", "spurious")}
    conf = np.zeros((3, 3), np.int64)
    for r in runs:
        a, e = int(r[0]), int(r[-1]) + 1
        g, p = gt[a:e].astype(np.int8), pred[a:e].astype(np.int8)
        rr, ss = spell_modes(g, p)
        rows += rr
        spur += ss
        for i in range(3):
            for j in range(3):
                conf[i, j] += int(((g == i) & (p == j)).sum())
        fixed["timing"][a:e] = eo.snap_timing(p, g)
        fixed["short"][a:e] = eo.insert_short(p, g)
        fixed["spurious"][a:e] = eo.drop_spurious(p, g)
        blk = next(v for v in halves.values() if v["a"] <= a and e - 1 <= v["b"])
        sl = slice(a - blk["a"], e - blk["a"])
        part = dict(blk, P=blk["P"][sl], B=blk["B"][sl])
        fr.add(situations(part, g), g, p)
    print("  frame confusion (rows true H/A/D, cols predicted H/A/D): %s" % "; ".join(
        " ".join("%d" % v for v in row) for row in conf))
    print("  alone on v10w: " + " | ".join("%s %+.4f" % (k, ev.score(v)["spellF1"] - s0["spellF1"])
                                          for k, v in fixed.items()))
    modes_report(rows, spur, indent="  ")
    fr.report(indent="  ")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    public()
    external()


if __name__ == "__main__":
    main()
