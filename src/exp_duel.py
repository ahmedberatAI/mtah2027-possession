"""Round 10 (user idea): duel-outcome features (src/duel.py; 25 per team view) appended to v10's inputs (v7 + ball-free
features), v10's training (DFL native + Metrica ctrl weight 2), same folds as src/exp_combo.py. DUEL_SPEC=1 trains a
contest specialist instead: the same inputs, rows from contest frames only (both teams within 3 m of the ball,
+-0.5 s), its probabilities replacing v10's (same seed) on contest frames. Raw probabilities go to
cache/bench/contest/probs_duel[spec]_seed<k>.pkl; compare with  python -I src/exp_contest.py decode duel v10 .
AGGREGATE output only.

  LGB_SEED=k [DUEL_SPEC=1] python -I src/exp_duel.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import duel  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_contest as ec  # noqa: E402

OUT = os.path.join(bench.CACHE, "duel")
SPEC = os.environ.get("DUEL_SPEC") == "1"
W_M = 2.0


def _job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    return (i,) + duel.duel_feats(b["P"], b["B"], b["team"], b["gk"], b["pitch"])


def extra(src):
    path = os.path.join(OUT, "feat_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_job, [(src, i) for i in range(len(bench.blocks(src)))]), key=lambda r: r[0])
    out = [(r[1], r[2]) for r in res]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def aug_feats(src):
    return [dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))
            for f, (h, a) in zip(exp_combo.aug_feats(src), extra(src))]


def masks(src):
    """Training-row mask per block: everything, or contest frames for the specialist."""
    sig = ec.prep(src)["sig"]
    return [ec.contest(s) if SPEC else np.ones(len(s["seen"]), bool) for s in sig]


def rows(feats, labs, ms, w):
    return ec.rows_c(feats, labs, [m.astype(np.float32) for m in ms], w)


def nonzero(part):
    X, y, w = part
    keep = w > 0
    return X[keep], y[keep], w[keep]


def probs():
    tag = "duelspec" if SPEC else "duel"
    path = os.path.join(ec.OUT, "probs_%s_seed%d.pkl" % (tag, exp_a2.SEED))
    if os.path.isfile(path):
        return
    fd, fm = aug_feats("dfl@v3"), aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    md, mm = masks("dfl@v3"), masks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = nonzero(rows(fm, lm, mm, W_M / 3.0))
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([nonzero(rows([fd[i] for i in tr], [ld[i] for i in tr], [md[i] for i in tr], 1.0)), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = nonzero(rows(fd, ld, md, 1.0))
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, nonzero(rows([fm[i] for i in tr], [lm[i] for i in tr], [mm[i] for i in tr],
                                              W_M / 3.0))])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    g = mdl.feature_importance("gain")
    print("duel columns' gain share in the all-data model: %.3f" % (g[-25:].sum() / g.sum()), flush=True)
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in aug_feats("sc@v3")]
    if SPEC:                                   # the specialist only speaks on contest frames; v10 elsewhere
        base = bench._load(ec.RAW[exp_a2.SEED])
        for src in res:
            sig = ec.prep(src)["sig"]
            res[src] = [np.where(ec.contest(s), p, q) for s, p, q in zip(sig, res[src], base[src])]
    os.makedirs(ec.OUT, exist_ok=True)
    bench._dump(res, path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    probs()


if __name__ == "__main__":
    main()
