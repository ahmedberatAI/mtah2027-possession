"""Round 9: a learned temporal smoother ("stacker") on top of v10 (round 5's N20, src/exp_stack.py, did this on v7 and
lost; v10's probabilities are jumpier and gain from smoothing, v7's did not). v10 averages its team logits over a fixed
+-25-frame box; here a small LightGBM reads the out-of-fold v10 logit at several time scales -- centred means over +-6 ... +-200
frames, past-only and future-only means, their differences, the local spread -- and predicts the H / A label again
(STACK_V7=1 adds the same multi-scale summary of v7's logits: model diversity). No dead-ball inputs (the deletion
rescorer showed what depending on the D source does). Stacked out of fold: DFL by match, Metrica by game (the
first-stage probabilities are already out of fold, cache/bench/combo/probs_a2_bfree_wm2.pkl), SkillCorner from a
stacker trained on all of DFL + Metrica. v7 decoder; paired spell F1 against v10 and v7, learned and true D.
AGGREGATE output only.

  python -I src/exp_stack10.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
from mlfeat import win_mean  # noqa: E402

EPS = 1e-4
W_M = 2.0
V7 = os.environ.get("STACK_V7", "0") == "1"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.9,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0, num_threads=bench.THREADS,
              deterministic=True, force_row_wise=True)
ROUNDS = 200
STRIDE = 4


def multiscale(p):
    z = e2.logit(p)
    cols = [z]
    for w in (6, 12, 25, 50, 100, 200):
        cols.append(win_mean(z, -w, w))
    for w in (25, 100):
        past, fut = win_mean(z, -w, 0), win_mean(z, 0, w)
        cols += [past, fut, fut - past]
    m25 = win_mean(z, -25, 25)
    cols.append(np.sqrt(np.maximum(win_mean(z * z, -25, 25) - m25 * m25, 0.0)))
    return np.column_stack(cols).astype(np.float32)


def feats(src, v10raw, v7raw):
    out = []
    for i, p in enumerate(v10raw[src]):
        X = multiscale(p)
        if V7:
            X = np.hstack([X, multiscale(v7raw[i])])
        out.append(X)
    return out


def rows(X, labs, w):
    Xs, ys = [], []
    for x, lab in zip(X, labs):
        x, lab = x[::STRIDE], lab[::STRIDE]
        alive = lab < 2
        Xs.append(x[alive])
        ys.append((lab[alive] == bench.H).astype(int))
    y = np.concatenate(ys)
    return np.concatenate(Xs), y, np.full(len(y), w, np.float32)


def fit(parts):
    X, y, w = (np.concatenate([p[k] for p in parts]) for k in range(3))
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w), ROUNDS)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    print("stacker, v7 inputs" if V7 else "stacker", flush=True)
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    v7 = {src: exp_oracle.probs(src) for src in ("dfl@v3", "metrica@v3", "sc@v3")}
    X = {src: feats(src, raw, v7[src][0]) for src in v7}
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    st = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = rows(X["metrica@v3"], lm, W_M / 3.0)
    d_all = rows(X["dfl@v3"], ld, 1.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = fit([rows([X["dfl@v3"][i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                st["dfl@v3"][i] = mdl.predict(X["dfl@v3"][i], num_threads=bench.THREADS)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = fit([d_all, rows([X["metrica@v3"][i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                st["metrica@v3"][i] = mdl.predict(X["metrica@v3"][i], num_threads=bench.THREADS)
    mdl = fit([d_all, m_all])
    st["sc@v3"] = [mdl.predict(x, num_threads=bench.THREADS) for x in X["sc@v3"]]
    imp = mdl.feature_importance("gain")
    print("gain share (all-data stacker): " + " ".join("%.2f" % v for v in imp / imp.sum()), flush=True)
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = v7[src]
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            for w in (0, 12):
                v = e2.score(src, conv, e2.decode_all(dec, pds, exp_smooth.smooth(st[src], w)))
                cells.append("%s w%d %.4f (vs v10 %+.4f, %d/%d; vs v7 %+.4f)" % (
                    dname, w, v.mean(), v.mean() - r10.mean(), int((v > r10).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s acc %.4f (v10 %.4f) | %s" % (src, conv, e2.team_acc(st[src], labs),
                                                      e2.team_acc(v10[src], labs), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
