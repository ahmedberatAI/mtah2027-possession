"""Round 9: v10 trained on pre-smoothed DFL (sim.V4: open DFL is ~5x rougher frame to frame than BAR-ATH's filtered
ATD; Metrica and SkillCorner are as smooth as BAR-ATH). Round 5's N26 found V4 training neutral for v7, but v10's
ball-free features are built from player velocities, where roughness matters more. Training: DFL@V4 (native) +
Metrica@V3 (ctrl, weight 2; Metrica is smooth already) -- or Metrica@V4 with V4T_METRICA=v4. Held-out predictions on
the V3 inputs (the usual test, against v10) and on the V4 inputs (against v10 applied to them, src/exp_stress.py).
Paired spell F1, learned and true D. AGGREGATE output only.

  python -I src/exp_v4train.py
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
import exp_fab  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import exp_stress  # noqa: E402

OUT = os.path.join(bench.CACHE, "v4train")
EPS = 1e-4
W_M = 2.0
MVAR = os.environ.get("V4T_METRICA", "v3")
TESTS = ("v3", "v4")


def probs():
    path = os.path.join(OUT, "probs_m%s.pkl" % MVAR)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = exp_combo.aug_feats("dfl@v4"), exp_combo.aug_feats("metrica@" + MVAR)
    td = {v: fd if v == "v4" else exp_combo.aug_feats("dfl@" + v) for v in TESTS}
    tm = {v: fm if v == MVAR else exp_combo.aug_feats("metrica@" + v) for v in TESTS}
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {v: {"dfl": [None] * len(bd), "metrica": [None] * len(bm)} for v in TESTS}
    m_all = exp_a2.rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                for v in TESTS:
                    res[v]["dfl"][i] = bench.predict_team(mdl, td[v][i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = exp_a2.rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                for v in TESTS:
                    res[v]["metrica"][i] = bench.predict_team(mdl, tm[v][i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm, td, tm
    for v in TESTS:
        res[v]["sc"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@" + v)]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    print("training DFL@v4 + Metrica@%s" % MVAR, flush=True)
    res = probs()
    ref = {"v3": bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl")),
           "v4": {k: exp_smooth.smooth(p, 25) for k, p in exp_stress.v10_on("v4").items()}}
    dead4 = exp_fab.cross_probs("v3", "v4")
    for v in TESTS:
        for base, conv, dec in exp_stress.SETS:
            src = "%s@%s" % (base, v)
            labs = bench.labels(src, conv)
            pdl = exp_oracle.probs(src)[1] if v == "v3" else dead4[base][1]
            r10 = ref[v][src] if v == "v3" else ref[v][base]
            new = exp_smooth.smooth(res[v][base], 25)
            cells = []
            for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
                a = e2.score(src, conv, e2.decode_all(dec, pds, r10))
                b = e2.score(src, conv, e2.decode_all(dec, pds, new))
                cells.append("%s v10 %.4f V4-trained %.4f (%+.4f, %d/%d)" % (dname, a.mean(), b.mean(),
                                                                          b.mean() - a.mean(), int((b > a).sum()),
                                                                          len(b)))
            print("%s test %-8s %-6s | %s" % (v.upper(), base, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
