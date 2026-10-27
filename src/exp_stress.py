"""Round 9 stress test (research a19): does v10's advantage over v7 survive a test-time degradation the models were not
trained on? Both are trained on sim V3 inputs as usual and evaluated on held-out matches degraded with sim.V5 (V3
plus ATD-like fabricated straight-line ball runs on 3% of live frames, src/exp_fab.py).

  v7    the v3-trained team / dead models applied to V5 inputs (cache/bench/fab/probs_v3_on_v5.pkl)
  v10   DFL + Metrica (weight 2) + ball-free features, trained on V3, applied to V5 inputs, +-25-frame smoothing
Same learned dead-ball probabilities (v3 dead model on V5 inputs) and v7 decoder for both; paired spell F1 on
DFL-trk, Metrica-ctrl, SkillCorner phase / player-possession, learned and true D. AGGREGATE output only.

  python -I src/exp_stress.py [variant, default v5; e.g. v4 = V3 on pre-smoothed tracks, as smooth as BAR-ATH]
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
import exp_smooth  # noqa: E402

OUT = os.path.join(bench.CACHE, "stress")
EPS = 1e-4
W_M = 2.0
SETS = (("dfl", "trk", (32.0, 5, 25)), ("metrica", "ctrl", (30.0, 8, 50)), ("sc", "phase", (30.0, 8, 50)),
        ("sc", "pp", (30.0, 8, 50)))


def v10_on(var):
    path = os.path.join(OUT, "probs_v10_on_%s.pkl" % var)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("metrica@v3")
    td, tm = exp_combo.aug_feats("dfl@" + var), exp_combo.aug_feats("metrica@" + var)
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl": [None] * len(bd), "metrica": [None] * len(bm)}
    m_all = exp_a2.rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl"][i] = bench.predict_team(mdl, td[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = exp_a2.rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica"][i] = bench.predict_team(mdl, tm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm, td, tm
    res["sc"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@" + var)]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    var = sys.argv[1] if len(sys.argv) > 1 else "v5"
    import exp_fab
    v7 = exp_fab.cross_probs("v3", var)
    v10 = {k: exp_smooth.smooth(p, 25) for k, p in v10_on(var).items()}
    for base, conv, dec in SETS:
        src = "%s@%s" % (base, var)
        labs = bench.labels(src, conv)
        ph7, pdl = v7[base]
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, ph7))
            v = e2.score(src, conv, e2.decode_all(dec, pds, v10[base]))
            cells.append("%s v7 %.4f v10 %.4f (%+.4f, better in %d/%d)" % (dname, r7.mean(), v.mean(),
                                                                          v.mean() - r7.mean(), int((v > r7).sum()),
                                                                          len(v)))
        print("%s test %-8s %-6s | %s" % (var.upper(), base, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
