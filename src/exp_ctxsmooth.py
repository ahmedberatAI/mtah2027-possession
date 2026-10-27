"""Round 9: context-dependent smoothing of v10's team probabilities -- a wider logit window where the ball has been
missing (evidence weak), a narrower one where it is seen: z = s * mean_{+-w_seen}(z) + (1 - s) * mean_{+-w_miss}(z),
s = share of frames with a ball within +-1 s. Decode-only on the cached v10 probabilities (cache/bench/combo/
probs_a2_bfree_wm2.pkl), against v10 (uniform +-25) and v7; worst-case regret over the eight contexts.
AGGREGATE output only.

  python -I src/exp_ctxsmooth.py
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
from mlfeat import win_mean  # noqa: E402

EPS = 1e-4
VARIANTS = {"uniform 25 (v10)": (25, 25), "seen 12 / miss 25": (12, 25), "seen 25 / miss 50": (25, 50),
            "seen 12 / miss 50": (12, 50), "seen 25 / miss 12": (25, 12)}


def ctx_smooth(p, seen, ws, wm):
    z = e2.logit(p)
    s = win_mean(seen.astype(np.float64), -25, 25)
    s = np.where(np.isfinite(s), s, 0.0)
    return e2.sig(s * win_mean(z, -ws, ws) + (1 - s) * win_mean(z, -wm, wm))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        seen = [np.isfinite(b["B"][:, 0]) for b in bench.blocks(src)]
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            rows[ctx] = {}
            cells = []
            for name, (ws, wm) in VARIANTS.items():
                ph = [ctx_smooth(p, s, ws, wm) for p, s in zip(raw[src], seen)]
                v = e2.score(src, conv, e2.decode_all(dec, pds, ph))
                rows[ctx][name] = v.mean()
                cells.append("%s %+.4f" % (name, v.mean() - r7.mean()))
            print("%-26s v7 %.4f | %s" % (ctx, r7.mean(), " | ".join(cells)), flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in VARIANTS}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
