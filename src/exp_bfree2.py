"""Round 9: v9 variant with longer context for the ball-free interaction features -- each of src/bfree.py's columns
also averaged over +-2 s (win_mean -50..50) -- trained as v9 (DFL + Metrica, src/exp_combo.py protocol) and decoded
with and without the +-25-frame logit smoothing. Paired spell F1 against v7 and v9. AGGREGATE output only.

  python -I src/exp_bfree2.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_bfree  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
from mlfeat import win_mean  # noqa: E402

OUT = os.path.join(bench.CACHE, "bfree2")
EPS = 1e-4


def _long(x):
    return np.stack([win_mean(x[:, j].astype(np.float64), -50, 50) for j in range(x.shape[1])], 1).astype(np.float32)


def aug_feats(src):
    out = []
    for b, (h, a) in zip(bench.blocks(src), exp_bfree.extra(src)):
        f = bench._load(e2._feat_path(b))
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h, _long(h)]), XA=np.hstack([f["XA"], a, _long(a)])))
    return out


def probs():
    path = os.path.join(OUT, "probs.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = aug_feats("dfl@v3"), aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = exp_a2.rows(fm, lm, 1.0 / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = exp_a2.rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], 1.0 / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in aug_feats("sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    new = probs()
    v9 = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2bf_w25.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r9 = e2.score(src, conv, e2.decode_all(dec, pds, v9[src]))
            for w in (0, 25):
                v = e2.score(src, conv, e2.decode_all(dec, pds, exp_smooth.smooth(new[src], w)))
                cells.append("%s w%d %.4f (vs v9 %+.4f, %d/%d; vs v7 %+.4f)" % (
                    dname, w, v.mean(), v.mean() - r9.mean(), int((v > r9).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s | %s" % (src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
