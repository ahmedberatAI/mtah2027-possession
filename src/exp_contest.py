"""Round 10 (user idea): special handling of contested-ball moments -- frames where players of BOTH teams are within
r m of the ball -- where v10 misses most short in-play possessions (src/exp_err10.py). AGGREGATE output only.

  diag     how often true changes of possession (found / missed by v10), v10's spurious changes and short true
           spells sit at a contest (within +-W frames), against the base rate of contest frames; same for "a player
           of the team taking over is within R m of the ball" (the physical precondition of a change)
  decode   decode-only variants on v10's three LightGBM seeds, eight contexts (learned / true D), paired against
           v10's decoder on the same seed:
             relief  switching INTO a team costs lam less within +-W frames of a contest
             gate    switching INTO team k costs mu more when no k player is within R m of the ball within +-W frames
                     (ball seen); delta lowers the base switch cost everywhere
             born    two copies of each team state: a spell that starts at a contest ends with a weaker
                     minimum-duration ramp (c instead of 48)
             narrow  at contests the +-25 logit smoothing is replaced by +-w
  train    v10's training (DFL native + Metrica ctrl weight 2, v7 + ball-free features) with sample weight
           1 + alpha at contest frames (CONTEST_ALPHA, LGB_SEED); raw probabilities to cache/bench/contest/
  python -I src/exp_contest.py diag | decode [model] | train
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_smooth  # noqa: E402
import exp_tswitch  # noqa: E402
import exp_v7  # noqa: E402
import train_dfl as td  # noqa: E402

OUT = os.path.join(bench.CACHE, "contest")
EPS = 1e-4
FPS = 25.0
BIG = 1e6
SRCS = {"dfl@v3": ("trk",), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp")}
DEC = {"dfl@v3": (32.0, 5, 25), "metrica@v3": (30.0, 8, 50), "sc@v3": (30.0, 8, 50)}
RAW = {0: os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"),
       1: os.path.join(bench.CACHE, "combo", "probs_a2_bfree_seed1_wm2.pkl"),
       2: os.path.join(bench.CACHE, "combo", "probs_a2_bfree_seed2_wm2.pkl")}
ALPHA = float(os.environ.get("CONTEST_ALPHA", "2"))


# -- signals ---------------------------------------------------------------------------------------------------------

def _signals(b):
    L, W = b["pitch"]
    P = np.asarray(b["P"], np.float32)[:, :, :2] * np.array([L, W], np.float32)
    B = np.asarray(b["B"], np.float32)[:, :2] * np.array([L, W], np.float32)
    team = np.asarray(b["team"])
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(P - B[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    d0 = np.min(np.where(team[None, :] == 0, d, np.inf), 1)
    d1 = np.min(np.where(team[None, :] == 1, d, np.inf), 1)
    return dict(d0=np.minimum(d0, 999).astype(np.float16), d1=np.minimum(d1, 999).astype(np.float16),
                seen=np.isfinite(B[:, 0]))


def prep(src):
    """Cached per-block signals, labels, groups and learned dead-ball probabilities (workers need no blocks)."""
    path = os.path.join(OUT, "prep_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    blocks = bench.blocks(src)
    out = dict(sig=[_signals(b) for b in blocks], meta=[(b["group"], b.get("seed", 0)) for b in blocks],
               labs={c: bench.labels(src, c) for c in SRCS[src]}, pdl=eo.probs(src)[1])
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def dilate(x, w):
    if w <= 0:
        return x.astype(bool)
    cs = np.concatenate(([0], np.cumsum(x.astype(np.int64))))
    n = len(x)
    t = np.arange(n)
    return (cs[np.minimum(n, t + w + 1)] - cs[np.maximum(0, t - w)]) > 0


def contest(s, r=3.0, w=12):
    return dilate((s["d0"] <= r) & (s["d1"] <= r) & s["seen"], w)


def near(s, k, R=2.0, w=12):
    return dilate((s["d%d" % k] <= R) & s["seen"], w)


def seen_share(s, w=12):
    x = s["seen"].astype(np.float64)
    cs = np.concatenate(([0], np.cumsum(x)))
    n = len(x)
    t = np.arange(n)
    hi, lo = np.minimum(n, t + w + 1), np.maximum(0, t - w)
    return (cs[hi] - cs[lo]) / (hi - lo)


# -- scoring -----------------------------------------------------------------------------------------------------------

def group_scores(meta, gts, preds):
    groups = {}
    for k, g, p in zip(meta, gts, preds):
        groups.setdefault(k, ([], []))
        groups[k][0].append(g)
        groups[k][1].append(p)
    return np.array([td.spell_score(*groups[k])[0] for k in sorted(groups)])


# -- diagnostic --------------------------------------------------------------------------------------------------------

def diag():
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src in SRCS:
        pr = prep(src)
        conv = SRCS[src][0]
        gts = pr["labs"][conv]
        tdd = [np.where(g == 2, 1 - EPS, EPS) for g in gts]
        preds = e2.decode_all(DEC[src], tdd, v10[src])
        acc = {k: [0, 0, 0] for k in ("true change found", "true change missed", "v10 spurious change",
                                      "short true spell found", "short true spell missed")}
        base = [0, 0, 0, 0]
        for s, g, p in zip(pr["sig"], gts, preds):
            c = contest(s)
            nk = (near(s, 0), near(s, 1))
            seen = seen_share(s) >= 0.5
            alive = g < 2
            base[0] += alive.sum()
            base[1] += (c & alive).sum()
            base[2] += ((~nk[0]) & seen & alive).sum()
            base[3] += ((~nk[1]) & seen & alive).sum()
            tc, pc = eo.changes(g), eo.changes(p)

            def add(key, t, k):
                acc[key][0] += 1
                acc[key][1] += int(c[t])
                acc[key][2] += int(nk[k][t] or not seen[t])
            for t in tc:
                k = int(g[t])
                hit = np.any((np.abs(pc - t) <= 25) & (p[pc] == k)) if len(pc) else False
                add("true change found" if hit else "true change missed", t, k)
            for t in pc:
                k = int(p[t])
                if not (np.any((np.abs(tc - t) <= 25) & (g[tc] == k)) if len(tc) else False):
                    add("v10 spurious change", t, k)
            gs, ps = eo.spells(g), eo.spells(p)
            mg, _ = eo.matched(gs, ps, 0.5)
            for (a, e, k), ok in zip(gs, mg):
                if e - a < 4 * FPS and a > 0 and g[a - 1] < 2:
                    add("short true spell found" if ok else "short true spell missed", a, int(k))
        print("%-10s alive frames at a contest (both teams <= 3 m, +-0.5 s) %.1f%%; with no H / A player <= 2 m "
              "(ball seen) %.1f%% / %.1f%%" % (src.split("@")[0], 100 * base[1] / base[0], 100 * base[2] / base[0],
                                               100 * base[3] / base[0]))
        for key, (n, c_, nr) in acc.items():
            print("    %-24s n %5d | at a contest %.1f%% | taking-over team near the ball (or ball unseen) %.1f%%"
                  % (key, n, 100 * c_ / max(n, 1), 100 * nr / max(n, 1)), flush=True)


# -- decode variants ---------------------------------------------------------------------------------------------------

VARIANTS = {
    "v10": {},
    "relief 4": dict(lam=4), "relief 8": dict(lam=8), "relief 12": dict(lam=12),
    "relief 8 r2": dict(lam=8, r=2.0), "relief 8 W25": dict(lam=8, W=25),
    "gate 4": dict(mu=4), "gate 8": dict(mu=8), "gate 16": dict(mu=16), "gate 8 R3": dict(mu=8, R=3.0),
    "gate 8 d4": dict(mu=8, delta=4), "gate 12 d6": dict(mu=12, delta=6),
    "relief 8 gate 8": dict(lam=8, mu=8),
    "born 0": dict(born=0.0), "born 16": dict(born=16.0), "born 24": dict(born=24.0), "born 32": dict(born=32.0),
    "born 24 relief 8": dict(born=24.0, lam=8),
    "narrow 6": dict(narrow=6), "narrow 12": dict(narrow=12), "narrow 12 born 24": dict(narrow=12, born=24.0),
}


def born_spec(dec, c):
    ramp, weak = exp_v7.ramp(175, 48.0), exp_v7.ramp(175, c)
    team = lambda e, r: dict(emis=e, out=e, L=len(r), min_exit=dec[1] - 1, exit_cost=r)
    macros = [team(0, ramp), team(1, ramp), dict(emis=2, out=2, L=dec[2]), team(0, weak), team(1, weak)]
    inf, p, d = np.inf, 12.0, dec[0]
    # rows: from H, A, D, H_c, A_c; cols: to the same
    pen = [[0, p, d, inf, p], [p, 0, d, p, inf], [d, d, 0, inf, inf], [inf, p, d, 0, p], [p, inf, d, p, 0]]
    ch = [[-1, 0, -1, -1, 2], [1, -1, -1, 3, -1], [-1] * 5, [-1, 0, -1, -1, 2], [1, -1, -1, 3, -1]]
    return decode2.Spec(macros, pen, ch)


def chan(v, s, n):
    """(n, 2): extra cost of entering A from H (col 0) and H from A (col 1)."""
    out = np.full((n, 2), -float(v.get("delta", 0.0)))
    if v.get("lam"):
        out -= v["lam"] * contest(s, v.get("r", 3.0), v.get("W", 12))[:, None]
    if v.get("mu"):
        seen = seen_share(s, v.get("W", 12)) >= 0.5
        for col, k in ((0, 1), (1, 0)):
            out[:, col] += v["mu"] * ((~near(s, k, v.get("R", 2.0), v.get("W", 12))) & seen)
    return out


def _decode_job(job):
    src, seed, names, model = job
    warnings.filterwarnings("ignore")
    pr = prep(src)
    raw = bench._load(RAW[seed] if model == "v10" else os.path.join(OUT, "probs_%s_seed%d.pkl" % (model, seed)))[src]
    lg = [e2.logit(np.clip(p, EPS, 1 - EPS)) for p in raw]
    from mlfeat import win_mean
    sm = {25: [win_mean(x, -25, 25) for x in lg]}
    dec = DEC[src]
    gts0 = pr["labs"][SRCS[src][0]]
    pds = {"learned D": pr["pdl"], "true D": [np.where(g == 2, 1 - EPS, EPS) for g in gts0]}
    sp3 = exp_tswitch.spec(dec)
    res = {}
    for name in names:
        v = VARIANTS[name]
        w_c = v.get("narrow")
        if w_c and w_c not in sm:
            sm[w_c] = [win_mean(x, -w_c, w_c) for x in lg]
        phs = []
        for i, s in enumerate(pr["sig"]):
            z = sm[25][i]
            if w_c:
                z = np.where(contest(s, v.get("r", 3.0), v.get("W", 12)), sm[w_c][i], z)
            phs.append(e2.sig(z))
        if "born" in v:
            sp = born_spec(dec, v["born"])
        else:
            sp = sp3
        for dname, pdd in pds.items():
            preds = []
            for i, (pd_, ph, s) in enumerate(zip(pdd, phs, pr["sig"])):
                em = td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0)
                c2 = chan(v, s, len(em))
                if "born" in v:
                    gate = np.where(contest(s, v.get("r", 3.0), v.get("W", 12)), 0.0, BIG)
                    c2 = np.column_stack([c2[:, 0], c2[:, 1], c2[:, 0] + gate, c2[:, 1] + gate])
                preds.append(sp.decode(em, c2).astype(np.int8))
            if name == "v10" and model == "v10":
                ref = e2.decode_all(dec, pdd, phs)
                assert all(np.array_equal(a, b) for a, b in zip(ref, preds)), "v10 variant must reproduce v10"
            for conv in SRCS[src]:
                res[(name, src, conv, dname)] = group_scores(pr["meta"], pr["labs"][conv], preds)
    return seed, res


def decode(model="v10", names=None):
    for src in SRCS:
        prep(src)
    names = list(names or VARIANTS)
    if "v10" not in names:
        names = ["v10"] + names
    chunks = {"dfl@v3": 2, "metrica@v3": 2, "sc@v3": 5}
    jobs = []
    for src, k in chunks.items():
        rest = [n for n in names if n != "v10"]
        if not rest:
            jobs.append((src, None, ["v10"]))
        for j in range(k):
            part = rest[j::k]
            if part:
                jobs.append((src, None, ["v10"] + part))
    from multiprocessing import Pool
    seeds = (0, 1, 2)
    if model != "v10":
        seeds = tuple(s for s in seeds if os.path.isfile(os.path.join(OUT, "probs_%s_seed%d.pkl" % (model, s))))
    all_jobs = [(src, seed, nm, model) for src, _, nm in jobs for seed in seeds]
    with Pool(int(os.environ.get("CONTEST_PROCS", "9"))) as pool:
        out = pool.map(_decode_job, all_jobs)
    ref = {}
    if model != "v10":                       # the new model's plain decode is compared with v10's on the same seed
        ref_jobs = [(src, seed, ["v10"], "v10") for src in SRCS for seed in seeds]
        with Pool(int(os.environ.get("CONTEST_PROCS", "9"))) as pool:
            for seed, res in pool.map(_decode_job, ref_jobs):
                for (name, src, conv, dname), arr in res.items():
                    ref[(seed, src, conv, dname)] = arr
    table = {}
    for seed, res in out:
        for (name, src, conv, dname), arr in res.items():
            table[(name, seed, src, conv, dname)] = arr
    ctxs = [(src, conv, d) for src in SRCS for conv in SRCS[src] for d in ("learned D", "true D")]
    means = {}
    print("model %s, seeds %s; per context: base mean, then variant - base (mean over seeds; seeds > 0)" % (
        model, list(seeds)))
    for src, conv, d in ctxs:
        cells = []
        base = [ref[(s, src, conv, d)] if model != "v10" else table[("v10", s, src, conv, d)] for s in seeds]
        bm = np.mean([b.mean() for b in base])
        for name in names:
            vals = [table[(name, s, src, conv, d)] for s in seeds]
            diffs = [v.mean() - b.mean() for v, b in zip(vals, base)]
            means.setdefault(name, {})[(src, conv, d)] = bm + np.mean(diffs)
            if name != "v10" or model != "v10":
                cells.append("%s %+.4f (%d/%d)" % (name, np.mean(diffs), sum(x > 0 for x in diffs), len(diffs)))
        print("%-8s %-5s %-9s base %.4f | %s" % (src.split("@")[0], conv, d, bm, " | ".join(cells)), flush=True)
    if model == "v10":
        regret = {n: max(max(means[m][c] for m in means) - means[n][c] for c in ctxs) for n in means}
        mean_gain = {n: np.mean([means[n][c] - means["v10"][c] for c in ctxs]) for n in means}
        print("worst-case regret (seed means) / mean gain over the 8 contexts: " + " | ".join(
            "%s %.4f / %+.4f" % (n, regret[n], mean_gain[n]) for n in sorted(regret, key=regret.get)))


# -- contest-weighted training -----------------------------------------------------------------------------------------

def rows_c(feats, labs, wfr, w):
    XF, XH, XA, y, wt = [], [], [], [], []
    for f, lab, q in zip(feats, labs, wfr):
        sl = slice(0, None, bench.STRIDE)
        XF.append(f["XF"][sl])
        XH.append(f["XH"][sl])
        XA.append(f["XA"][sl])
        y.append(lab[sl])
        wt.append(q[sl])
    XF, XH, XA, y, wt = (np.concatenate(XF), np.concatenate(XH), np.concatenate(XA), np.concatenate(y),
                         np.concatenate(wt))
    alive = (y == 0) | (y == 1)
    Xt = np.concatenate([np.hstack([XF[alive], XH[alive]]), np.hstack([XF[alive], XA[alive]])])
    yt = np.concatenate([y[alive] == 0, y[alive] == 1]).astype(int)
    return Xt, yt, (np.concatenate([wt[alive], wt[alive]]) * w).astype(np.float32)


def train():
    import exp_a2
    import exp_combo
    tag = "w%g" % ALPHA
    path = os.path.join(OUT, "probs_%s_seed%d.pkl" % (tag, exp_a2.SEED))
    if os.path.isfile(path):
        return
    fd, fm = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    wd = [1.0 + ALPHA * contest(s) for s in prep("dfl@v3")["sig"]]
    wm = [1.0 + ALPHA * contest(s) for s in prep("metrica@v3")["sig"]]
    w_m = 2.0 / 3.0
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = rows_c(fm, lm, wm, w_m)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([rows_c([fd[i] for i in tr], [ld[i] for i in tr], [wd[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = rows_c(fd, ld, wd, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, rows_c([fm[i] for i in tr], [lm[i] for i in tr], [wm[i] for i in tr], w_m)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)


ORACLE_SETS = {"contest r3 +-0.5 s": dict(r=3.0, W=12), "contest r5 +-1 s": dict(r=5.0, W=25),
               "near a true change +-1 s (cheat)": dict(change=25)}


def _oracle_job(job):
    src, names = job
    warnings.filterwarnings("ignore")
    pr = prep(src)
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))[src]
    gts0 = pr["labs"][SRCS[src][0]]
    pds = {"learned D": pr["pdl"], "true D": [np.where(g == 2, 1 - EPS, EPS) for g in gts0]}
    dec = DEC[src]
    res, share = {}, {}
    for name, alpha in names:
        o = ORACLE_SETS[name]
        phs, n_in, n_al = [], 0, 0
        for s, g, p in zip(pr["sig"], gts0, v10):
            z = e2.logit(p)
            if "change" in o:
                m = dilate(np.isin(np.arange(len(g)), eo.changes(g)), o["change"])
            else:
                m = contest(s, o["r"], o["W"])
            m = m & (g < 2)
            n_in += m.sum()
            n_al += (g < 2).sum()
            zt = np.where(g == 0, 4.0, -4.0)
            phs.append(e2.sig(np.where(m, (1 - alpha) * z + alpha * zt, z)))
        share[name] = n_in / max(n_al, 1)
        for dname, pdd in pds.items():
            preds = e2.decode_all(dec, pdd, phs)
            for conv in SRCS[src]:
                res[(name, alpha, conv, dname)] = group_scores(pr["meta"], pr["labs"][conv], preds).mean()
    base = {}
    for dname, pdd in pds.items():
        preds = e2.decode_all(dec, pdd, v10)
        for conv in SRCS[src]:
            base[(conv, dname)] = group_scores(pr["meta"], pr["labs"][conv], preds).mean()
    return src, res, base, share


def oracle():
    """Upper bounds for the contest idea: v10 (seed 0) with its smoothed team logit moved a share alpha of the way
    to the truth (+-4) on alive frames inside the set -- contests, or (not realisable) near a true change."""
    grid = [(n, a) for n in ORACLE_SETS for a in (1.0, 0.5, 0.25)]
    jobs = [(src, grid[i::3]) for src in SRCS for i in range(3)]
    from multiprocessing import Pool
    with Pool(9) as pool:
        out = pool.map(_oracle_job, jobs)
    res, base, share = {}, {}, {}
    for src, r, b, sh in out:
        res.update({(src,) + k: v for k, v in r.items()})
        base.update({(src,) + k: v for k, v in b.items()})
        share.setdefault(src, {}).update(sh)
    for src in SRCS:
        print("%s: share of alive frames per set: %s" % (src.split("@")[0], ", ".join(
            "%s %.1f%%" % (n, 100 * v) for n, v in share[src].items())))
        for conv in SRCS[src]:
            for dname in ("learned D", "true D"):
                b = base[(src, conv, dname)]
                print("  %-5s %-9s v10 %.4f | %s" % (conv, dname, b, " | ".join(
                    "%s a%.2g %+.4f" % (n, a, res[(src, n, a, conv, dname)] - b) for n, a in grid)), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        diag()
    elif mode == "oracle":
        oracle()
    elif mode == "decode":
        decode(sys.argv[2] if len(sys.argv) > 2 else "v10", sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    elif mode == "train":
        train()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
