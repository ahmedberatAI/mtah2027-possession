"""Round 9: re-test the player-level controller features (src/exp_ctrl2.py's 14 per-team features from the
SkillCorner-trained controller; SkillCorner itself out of fold) in the v10 regime -- DFL + Metrica (weight 2)
training, v7 + ball-free features, +-25-frame logit smoothing -- since smoothing and Metrica training turned the
ball-free features from a loss into a gain. Paired spell F1 against v10 and v7. AGGREGATE output only.

  python -I src/exp_ctrlv10.py
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
import exp_ctrl  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

OUT = os.path.join(bench.CACHE, "ctrlv10")
EPS = 1e-4
W_M = 2.0


def aug_feats(src):
    base = exp_combo.aug_feats(src)
    rows = exp_ctrl.build_rows(src)
    pc = e2.ctrl_p(src)
    out = []
    for f, r, p, b in zip(base, rows, pc, bench.blocks(src)):
        ch, ca = e2.ctrl_feats(*e2.per_frame(p, r, len(b["P"])))
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], ch]), XA=np.hstack([f["XA"], ca])))
    return out


def probs():
    path = os.path.join(OUT, "probs.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = aug_feats("dfl@v3"), aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = exp_a2.rows(fm, lm, W_M / 3.0)
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
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
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
    new = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v10 %.4f -> %.4f (vs v10 %+.4f, %d/%d; vs v7 %+.4f)" % (
                dname, r10.mean(), v.mean(), v.mean() - r10.mean(), int((v > r10).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s | %s" % (src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
