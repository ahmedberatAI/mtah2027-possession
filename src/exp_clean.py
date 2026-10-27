"""Round 10: would more precise positions (what a video pipeline could add) resolve contested-ball moments? v10's
recipe (v7 + ball-free features, DFL native + Metrica ctrl weight 2, same folds, seed 0) trained AND tested on sim
inputs with exact positions -- bench variant v3p0: V3's masks (ball dropouts, player track gaps; same random stream)
but no position noise and no aerial parallax. With the true D: spell F1 per context and frame team accuracy (+-25
smoothing) at contests / inside short (<4 s) in-play spells at contests (contest frames taken from the V3 inputs so the
frame sets are the same), against v10 on V3. AGGREGATE output only.

  python -I src/exp_clean.py
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
import exp_contest as ec  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_smooth  # noqa: E402
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "clean")
VAR = "v3p0"
EPS = 1e-4


def feats(src):
    bench.features(model_v7.VERSION, src)
    ex = exp_bfree.extra(src)
    out = []
    for b, (h, a) in zip(bench.blocks(src), ex):
        f = bench._load(e2._feat_path(b))
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
    return out


def probs():
    path = os.path.join(OUT, "probs_%s_seed%d.pkl" % (VAR, exp_a2.SEED))
    if os.path.isfile(path):
        return bench._load(path)
    sd, sm, ss = "dfl@" + VAR, "metrica@" + VAR, "sc@" + VAR
    fd, fm = feats(sd), feats(sm)
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica", "ctrl")
    bd, bm = bench.blocks(sd), bench.blocks(sm)
    ones = lambda fs: [np.ones(len(f["XF"]), np.float32) for f in fs]
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = ec.rows_c(fm, lm, ones(fm), 2.0 / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([ec.rows_c([fd[i] for i in tr], [ld[i] for i in tr], ones([fd[i] for i in tr]), 1.0),
                          m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = ec.rows_c(fd, ld, ones(fd), 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, ec.rows_c([fm[i] for i in tr], [lm[i] for i in tr], ones([fm[i] for i in tr]),
                                           2.0 / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in feats(ss)]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    new = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    ref = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src in ec.SRCS:
        pr = ec.prep(src)
        gts0 = pr["labs"][ec.SRCS[src][0]]
        tdd = [np.where(g == 2, 1 - EPS, EPS) for g in gts0]
        cells = []
        for name, phs in (("v10 V3", ref[src]), ("exact positions", new[src])):
            preds = e2.decode_all(ec.DEC[src], tdd, phs)
            sc = " ".join("%s %.4f" % (c, ec.group_scores(pr["meta"], pr["labs"][c], preds).mean())
                          for c in ec.SRCS[src])
            acc = {}
            for key in ("all alive", "contest", "short spell at contest", "short spell elsewhere"):
                ok = n = 0
                for s, g, p in zip(pr["sig"], gts0, phs):
                    c = ec.contest(s) & (g < 2)
                    short = np.zeros(len(g), bool)
                    for a, e, k in eo.spells(g):
                        if e - a < 100 and a > 0 and g[a - 1] < 2:
                            short[a:e] = True
                    m = {"all alive": g < 2, "contest": c, "short spell at contest": short & c,
                         "short spell elsewhere": short & ~c & (g < 2)}[key]
                    ok += ((p > 0.5) == (g == 0))[m].sum()
                    n += m.sum()
                acc[key] = ok / max(n, 1)
            cells.append("%s: spellF1 (true D) %s | team acc %s" % (name, sc, ", ".join(
                "%s %.3f" % kv for kv in acc.items())))
        print("%-8s %s" % (src.split("@")[0], "\n         ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
