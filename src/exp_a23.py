"""Round 9: A2 + A3 -- the SkillCorner matches (player-possession labels "pp", weight W_S per match) added to A2's
team-model training (DFL native + Metrica ctrl, src/exp_a2.py). Every set is evaluated on held-out matches:

  DFL-trk      leave one DFL match out (train: other 6 DFL + all Metrica + all SkillCorner)
  Metrica      leave one game out      (train: DFL + other 2 Metrica games + all SkillCorner)
  SkillCorner  5 folds by match        (train: DFL + Metrica + the other SkillCorner folds)
Paired spell F1 against A2 (cached, src/exp_a2.py) and v7, learned and true D. AGGREGATE output only.

  python -I src/exp_a23.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "a23")
EPS = 1e-4
W_S = float(os.environ.get("A23_WS", "0.3"))


def probs():
    path = os.path.join(OUT, "probs_ws%g.pkl" % W_S)
    if os.path.isfile(path):
        return bench._load(path)
    fd = bench.features(model_v7.VERSION, "dfl@v3")
    fm = bench.features(model_v7.VERSION, "metrica@v3")
    fs = bench.features(model_v7.VERSION, "sc@v3")
    ld, lm, ls = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl"), bench.labels("sc@v3", "pp")
    bd, bm, bs = bench.blocks("dfl@v3"), bench.blocks("metrica@v3"), bench.blocks("sc@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm), "sc@v3": [None] * len(bs)}
    d_all, m_all, s_all = exp_a2.rows(fd, ld, 1.0), exp_a2.rows(fm, lm, 1.0 / 3.0), exp_a2.rows(fs, ls, W_S)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all, s_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], 1.0 / 3.0), s_all])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    groups = sorted({b["group"] for b in bs})
    for k in range(5):
        test = set(groups[k::5])
        tr = [i for i, b in enumerate(bs) if b["group"] not in test]
        mdl = exp_a2.fit([d_all, m_all, exp_a2.rows([fs[i] for i in tr], [ls[i] for i in tr], W_S)])
        for i, b in enumerate(bs):
            if b["group"] in test:
                res["sc@v3"][i] = bench.predict_team(mdl, fs[i])
        print("  SC fold %d done" % k, flush=True)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    new = probs()
    a2 = bench._load(os.path.join(bench.CACHE, "a2", "probs_wm1.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r2 = e2.score(src, conv, e2.decode_all(dec, pds, a2[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v7 %.4f A2 %.4f A23 %.4f (vs A2 %+.4f, better in %d/%d; vs v7 %+.4f)" % (
                dname, r7.mean(), r2.mean(), v.mean(), v.mean() - r2.mean(), int((v > r2).sum()), len(v),
                v.mean() - r7.mean()))
        print("W_S %g %-10s %-6s | %s" % (W_S, src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
