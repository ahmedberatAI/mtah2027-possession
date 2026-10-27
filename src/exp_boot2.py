"""Round 9: paired match bootstrap of v11 (v10 + touch features) against v10 with the LightGBM seeds pooled: per match
the difference is the mean over the seeds run (0, 1, 2) of (v11_seed - v10_seed) spell F1, so seed noise averages out before
matches are resampled (B = 5000). AGGREGATE output only.

  python -I src/exp_boot2.py
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
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

EPS = 1e-4
B = 5000
C = bench.CACHE
PAIRS = [(os.path.join(C, "combo", "probs_a2_bfree_wm2.pkl"), os.path.join(C, "touch", "probs.pkl"))]
PAIRS += [(os.path.join(C, "combo", "probs_a2_bfree_seed%d_wm2.pkl" % s), os.path.join(C, "touch", "probs_seed%d.pkl" % s))
          for s in (1, 2, 3) if os.path.isfile(os.path.join(C, "touch", "probs_seed%d.pkl" % s))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    pairs = [tuple({k: exp_smooth.smooth(v, 25) for k, v in bench._load(p).items()} for p in pair) for pair in PAIRS]
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            d = np.mean([e2.score(src, conv, e2.decode_all(dec, pds, b[src])) -
                         e2.score(src, conv, e2.decode_all(dec, pds, a[src])) for a, b in pairs], 0)
            means = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            cells.append("%s %+.4f [5%% %+.4f, 95%% %+.4f] P(v11 > v10) %.3f" % (
                dname, d.mean(), np.quantile(means, 0.05), np.quantile(means, 0.95), np.mean(means > 0)))
        print("%-10s %-6s %d matches | %s" % (src, conv, len(labs) // 2 if src != "metrica@v3" else 9, " | ".join(cells)),
              flush=True)


if __name__ == "__main__":
    main()
