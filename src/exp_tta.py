"""Round 9: test-time augmentation by the pitch's left-right symmetry. Every held-out half is also mirrored across the
long axis (y -> 1 - y for players and ball), its v10 features recomputed, and v10's team logits averaged over the
original and mirrored inputs before the +-25-frame smoothing and the v7 decoder. v10 retrained as usual (DFL by
match, Metrica by game, all-data model for SkillCorner). Paired spell F1 against v10 and v7, learned and true D.
AGGREGATE output only.

  python -I src/exp_tta.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import bfree  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "tta")
EPS = 1e-4
W_M = 2.0


def _flip_job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = dict(bench.blocks(src)[i])
    P = b["P"].copy()
    P[:, :, 1] = 1.0 - P[:, :, 1]
    B = b["B"].copy()
    B[:, 1] = 1.0 - B[:, 1]
    b["P"], b["B"] = P, B
    f = model_v7._features(b)
    h, a = bfree.bf_feats(P, b["team"], b["gk"], b["pitch"])
    return i, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))


def flipped(src):
    path = os.path.join(OUT, "flip_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_flip_job, [(src, i) for i in range(len(bench.blocks(src)))]), key=lambda r: r[0])
    out = [r[1] for r in res]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def probs():
    path = os.path.join(OUT, "probs.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("metrica@v3")
    xd, xm = flipped("dfl@v3"), flipped("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"orig": {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)},
           "flip": {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}}
    m_all = exp_a2.rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["orig"]["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
                res["flip"]["dfl@v3"][i] = bench.predict_team(mdl, xd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = exp_a2.rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["orig"]["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
                res["flip"]["metrica@v3"][i] = bench.predict_team(mdl, xm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm, xd, xm
    res["orig"]["sc@v3"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@v3")]
    res["flip"]["sc@v3"] = [bench.predict_team(mdl, f) for f in flipped("sc@v3")]
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    res = probs()
    orig = {k: exp_smooth.smooth(v, 25) for k, v in res["orig"].items()}
    tta = {k: exp_smooth.smooth([e2.sig(0.5 * (e2.logit(p) + e2.logit(q))) for p, q in zip(v, res["flip"][k])], 25)
           for k, v in res["orig"].items()}
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, orig[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, tta[src]))
            cells.append("%s v10 %.4f TTA %.4f (vs v10 %+.4f, %d/%d; vs v7 %+.4f)" % (
                dname, r10.mean(), v.mean(), v.mean() - r10.mean(), int((v > r10).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s | %s" % (src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
