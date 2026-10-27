"""Round 9: v9 + two more identity-free team signals (bfree.bf_feats2: running-direction coherence / forward share,
composition of the densest 10 m neighbourhood), trained as v9 (DFL + Metrica, src/exp_combo.py protocol) and decoded
after the +-25-frame logit smoothing. Paired spell F1 against v9 and v7. AGGREGATE output only.

  python -I src/exp_bfree3.py
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
import exp_bfree  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

OUT = os.path.join(bench.CACHE, "bfree3")
EPS = 1e-4


def _job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    return (i,) + bfree.bf_feats2(b["P"], b["team"], b["gk"], b["pitch"])


def extra2(src):
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
    out = []
    for b, (h, a), (h2, a2) in zip(bench.blocks(src), exp_bfree.extra(src), extra2(src)):
        f = bench._load(e2._feat_path(b))
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h, h2]), XA=np.hstack([f["XA"], a, a2])))
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
        print("%-10s %-6s | %s" % (src, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
