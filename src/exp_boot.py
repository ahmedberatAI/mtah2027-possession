"""Round 9 (research a19): paired match bootstrap of the candidates against v7. Per set the per-match spell F1 of v7,
v8 (A2), v9 (A2 + ball-free features + smoothing) and v10 (v9, Metrica weight 2) at pen 12 is computed once; matches are resampled with
replacement (B = 5000) and the share of resamples in which the candidate's mean beats v7's is reported with the 5 /
95% quantiles of the mean difference. AGGREGATE output only.

  python -I src/exp_boot.py               (BOOT_VS=v10: v11 = v10 + touch features against v10; "P(>v7)" then reads P(>v10))
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402

EPS = 1e-4
B = 5000
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("sc@v3", "pp", (30.0, 8, 50)))
CANDS = {"v8": os.path.join(bench.CACHE, "a2", "probs_wm1.pkl"),
         "v9": os.path.join(bench.CACHE, "combo", "probs_a2bf_w25.pkl"),
         "v10": os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl")}
SMOOTH = {"v10": 25}                 # cached unsmoothed: smooth as the model does
REF = None                           # None = v7; BOOT_VS=v10: the touch candidate (v11) against v10
if os.environ.get("BOOT_VS") == "v10":
    CANDS = {"v11": os.path.join(bench.CACHE, "touch", "probs.pkl")}
    SMOOTH = {"v11": 25}
    REF = os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    import exp_smooth
    cands = {k: bench._load(v) for k, v in CANDS.items()}
    for k, w in SMOOTH.items():
        cands[k] = {src: exp_smooth.smooth(ph, w) for src, ph in cands[k].items()}
    ref_ph = None if REF is None else {s: exp_smooth.smooth(ph, 25) for s, ph in bench._load(REF).items()}
    for src, conv, dec in SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl if ref_ph is None else ref_ph[src]))
            cells = []
            for name, res in cands.items():
                v = e2.score(src, conv, e2.decode_all(dec, pds, res[src]))
                d = v - ref
                idx = rng.integers(0, len(d), (B, len(d)))
                means = d[idx].mean(1)
                cells.append("%s %+.4f [5%% %+.4f, 95%% %+.4f] P(>v7) %.3f" % (
                    name, d.mean(), np.quantile(means, 0.05), np.quantile(means, 0.95), np.mean(means > 0)))
            print("%-10s %-6s %-9s %d matches | %s" % (src, conv, dname, len(ref), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
