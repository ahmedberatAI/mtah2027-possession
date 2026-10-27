"""Round 9: paired match bootstrap of v10 + the touch switch relief (model_v7 option ts_lam = 8, ts_w = 6; decoder only,
so no seed noise) against plain v10, on the V3 test inputs and on the unseen V5 / V4 degradations (src/exp_tcombo.py's
decoder). B = 5000. AGGREGATE output only.

  python -I src/exp_boot3.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_fab  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import exp_stress  # noqa: E402
import exp_tcombo  # noqa: E402

EPS = 1e-4
B = 5000
NAME = "s8 W6"


def boot(rng, d):
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return "%+.4f [5%% %+.4f] P(>v10) %.3f" % (d.mean(), np.quantile(means, 0.05), np.mean(means > 0))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref, out = exp_tcombo.run(src, conv, dec, v10[src], pds, [NAME])
            cells.append("%s %s" % (dname, boot(rng, out[NAME] - ref)))
        print("V3 %-10s %-6s | %s" % (src, conv, " | ".join(cells)), flush=True)
    for var in ("v5", "v4"):
        dead = exp_fab.cross_probs("v3", var)
        probs = {k: exp_smooth.smooth(p, 25) for k, p in exp_stress.v10_on(var).items()}
        for base, conv, dec in exp_stress.SETS:
            src = "%s@%s" % (base, var)
            labs = bench.labels(src, conv)
            cells = []
            for dname, pds in (("learned D", dead[base][1]),
                               ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
                ref, out = exp_tcombo.run(src, conv, dec, probs[base], pds, [NAME])
                cells.append("%s %s" % (dname, boot(rng, out[NAME] - ref)))
            print("%s %-10s %-6s | %s" % (var.upper(), base, conv, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
