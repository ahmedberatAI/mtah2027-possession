"""Round 9: stress test of the touch features (src/touch.py) on degradations the models are not trained on, as
src/exp_stress.py did for v10: v10 and v10 + touch, both trained on sim V3 inputs (DFL native + Metrica ctrl weight 2),
applied to held-out matches degraded with
  v5   V3 + ATD-like fabricated straight-line ball runs (3% of live frames; BAR-ATH's tactical ball has them)
  v4   V3 on pre-smoothed tracks (ball low-passed over 4 frames: blunted kicks, fewer detected touches)
Same learned dead-ball probabilities and v7 decoder for both; paired spell F1, learned and true D.
AGGREGATE output only.

  python -I src/exp_touch_stress.py v5
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
import exp_fab  # noqa: E402
import exp_smooth  # noqa: E402
import exp_stress  # noqa: E402
import exp_touch  # noqa: E402

OUT = os.path.join(bench.CACHE, "touch")
EPS = 1e-4
W_M = 2.0


def touch_on(var):
    path = os.path.join(OUT, "probs_touch_on_%s.pkl" % var)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = exp_touch.aug_feats("dfl@v3"), exp_touch.aug_feats("metrica@v3")
    td, tm = exp_touch.aug_feats("dfl@" + var), exp_touch.aug_feats("metrica@" + var)
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
    res["sc"] = [bench.predict_team(mdl, f) for f in exp_touch.aug_feats("sc@" + var)]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    var = sys.argv[1] if len(sys.argv) > 1 else "v5"
    dead = exp_fab.cross_probs("v3", var)
    v10 = {k: exp_smooth.smooth(p, 25) for k, p in exp_stress.v10_on(var).items()}
    new = {k: exp_smooth.smooth(p, 25) for k, p in touch_on(var).items()}
    for base, conv, dec in exp_stress.SETS:
        src = "%s@%s" % (base, var)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", dead[base][1]), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            a = e2.score(src, conv, e2.decode_all(dec, pds, v10[base]))
            b = e2.score(src, conv, e2.decode_all(dec, pds, new[base]))
            cells.append("%s v10 %.4f touch %.4f (%+.4f, %d/%d)" % (dname, a.mean(), b.mean(), b.mean() - a.mean(),
                                                                   int((b > a).sum()), len(b)))
        print("%s test %-8s %-6s | %s" % (var.upper(), base, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
