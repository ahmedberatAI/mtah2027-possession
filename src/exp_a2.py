"""Round 9, idea A2 (docs/RAPOR.md, open since round 1): add the Metrica sample games -- same vendor as BAR-ATH's
ATD, labels in the controlled-possession convention ("ctrl") -- to the team model's training. v7 features, sim V3.

  Metrica    leave-one-game-out: DFL (7) + the other 2 Metrica games -> held-out game (its 3 degradation seeds)
  DFL-trk    leave-one-match-out: the other 6 DFL + all 3 Metrica games -> held-out DFL match
  SkillCorner  all DFL + all Metrica -> SkillCorner (phase / player-possession labels)
Metrica rows get weight W_M / 3 (three seeds = one game). Dead-ball probabilities and decoder as v7; paired spell F1
against v7 (DFL only), learned and true D. AGGREGATE output only.

  python -I src/exp_a2.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "a2")
EPS = 1e-4
W_M = float(os.environ.get("A2_WM", "1.0"))
SEED = int(os.environ.get("LGB_SEED", "0"))      # robustness runs (round 9); 0 = the default model
LEAVES = int(os.environ.get("LGB_LEAVES", "0"))   # capacity runs (round 9); 0 = LGB_TEAM's
ROUNDS = int(os.environ.get("LGB_ROUNDS", "0"))   # 0 = bench.ROUNDS
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("sc@v3", "pp", (30.0, 8, 50)),
        ("dfl@v3", "evctrl", (32.0, 5, 25)))   # round 12: controlled possession from the synchronised DFL events


def rows(feats, labs, w):
    Xt, yt, _ = bench.team_rows(feats, labs)
    return Xt, yt, np.full(len(yt), w, np.float32)


def fit(parts):
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2] for p in parts])
    params = dict(bench.LGB_TEAM, num_threads=bench.THREADS, seed=SEED)
    if LEAVES:
        params.update(num_leaves=LEAVES, min_data_in_leaf=400)
    return lgb.train(params, lgb.Dataset(X, y, weight=w), ROUNDS or bench.ROUNDS)


def probs():
    path = os.path.join(OUT, "probs_wm%g.pkl" % W_M)
    if os.path.isfile(path):
        return bench._load(path)
    fd = bench.features(model_v7.VERSION, "dfl@v3")
    fm = bench.features(model_v7.VERSION, "metrica@v3")
    ld = bench.labels("dfl", "native")
    lm = bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = fit([rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = fit([d_all, rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in bench.features(model_v7.VERSION, "sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    new = probs()
    for src, conv, dec in SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v7 %.4f -> %.4f (%+.4f, better in %d/%d)" % (dname, ref.mean(), v.mean(),
                                                                         v.mean() - ref.mean(), int((v > ref).sum()),
                                                                         len(v)))
        print("W_M %g %-10s %-6s frame team acc v7 %.4f -> A2 %.4f | %s" % (
            W_M, src, conv, e2.team_acc(phl, labs), e2.team_acc(new[src], labs), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
