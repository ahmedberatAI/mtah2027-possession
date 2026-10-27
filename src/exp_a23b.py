"""Round 9: v9 + SkillCorner in the training -- with SkillCorner's player-possession labels put into the CONTROLLED
convention ("ctrl": bench.trk_ctrl's rule; 3.94 spells / alive min, median 10.2 s, like Metrica-ctrl's 4.01 / 10.3 s)
instead of the raw pp labels that pulled the model away from Metrica (src/exp_a23.py). v9 features (v7 + ball-free),
DFL native + Metrica ctrl (1/3 per seed copy) + SkillCorner ctrl (W_S per match), +-25-frame logit smoothing.
Held-out: DFL by match, Metrica by game, SkillCorner 5 folds by match. Paired spell F1 against v9 and v7.
AGGREGATE output only.

  python -I src/exp_a23b.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

OUT = os.path.join(bench.CACHE, "a23b")
EPS = 1e-4
W_S = float(os.environ.get("A23_WS", "0.3"))


def probs():
    path = os.path.join(OUT, "probs_ws%g.pkl" % W_S)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm, fs = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("metrica@v3"), exp_combo.aug_feats("sc@v3")
    ld, lm, ls = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl"), bench.labels("sc@v3", "ctrl")
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
    new = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    v9 = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2bf_w25.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r9 = e2.score(src, conv, e2.decode_all(dec, pds, v9[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v9 %.4f -> %.4f (vs v9 %+.4f, %d/%d; vs v7 %+.4f)" % (
                dname, r9.mean(), v.mean(), v.mean() - r9.mean(), int((v > r9).sum()), len(v), v.mean() - r7.mean()))
        print("W_S %g %-10s %-6s | %s" % (W_S, src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
