"""Round 9, idea E2: training augmentation with a second, independently degraded copy of every DFL match and Metrica
game (bench variant "v3s2", same sim.V3 with other random seeds) for the v9 model (DFL + Metrica, v7 + ball-free
features, +-25-frame logit smoothing). Copies share the match's weight (DFL 1/2 per copy, Metrica 1/6 per seed copy);
held-out matches are evaluated on their ORIGINAL v3 copy only, exactly as v9 (src/exp_combo.py). Paired spell F1
against v7 and v9, learned and true D. AGGREGATE output only.

  python -I src/exp_aug.py
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
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "aug")
EPS = 1e-4


def probs():
    path = os.path.join(OUT, "probs_aug.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    for src in ("dfl@v3s2", "metrica@v3s2"):
        bench.features(model_v7.VERSION, src)
    fd, fd2 = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("dfl@v3s2")
    fm, fm2 = exp_combo.aug_feats("metrica@v3"), exp_combo.aug_feats("metrica@v3s2")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = [exp_a2.rows(fm, lm, 1.0 / 6.0), exp_a2.rows(fm2, lm, 1.0 / 6.0)]
    d_all = [exp_a2.rows(fd, ld, 0.5), exp_a2.rows(fd2, ld, 0.5)]
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        parts = [exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 0.5),
                 exp_a2.rows([fd2[i] for i in tr], [ld[i] for i in tr], 0.5)] + m_all
        mdl = exp_a2.fit(parts)
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        parts = d_all + [exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], 1.0 / 6.0),
                         exp_a2.rows([fm2[i] for i in tr], [lm[i] for i in tr], 1.0 / 6.0)]
        mdl = exp_a2.fit(parts)
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit(d_all + m_all)
    del fd, fd2, fm, fm2
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    aug = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    v9 = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2bf_w25.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r9 = e2.score(src, conv, e2.decode_all(dec, pds, v9[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, aug[src]))
            cells.append("%s v7 %.4f v9 %.4f aug %.4f (vs v9 %+.4f, better in %d/%d; vs v7 %+.4f)" % (
                dname, r7.mean(), r9.mean(), v.mean(), v.mean() - r9.mean(), int((v > r9).sum()), len(v),
                v.mean() - r7.mean()))
        print("%-10s %-6s | %s" % (src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
