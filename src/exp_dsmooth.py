"""Round 9: should v10's +-25-frame logit smoothing skip dead-ball frames? The team model is trained on alive frames
only, yet it is applied (and smoothed) everywhere, so the first / last second of every spell next to a stoppage is
averaged with team logits from dead frames (which may be noise -- or a useful "who restarts" signal). Decode-only on
the cached raw v10 probabilities (cache/bench/combo/probs_a2_bfree_wm2.pkl):
  alive-weighted   z_s(t) = sum a(s) z(s) / sum a(s) over |s - t| <= 25, a = alive probability of the context's
                   dead-ball source (learned P(dead), or the true D); plain mean where the window is all dead
  split at D       decode v10 once, then average only within the same decoded alive run, decode again
against v10 (plain +-25) on the eight contexts; worst-case regret. AGGREGATE output only.

  python -I src/exp_dsmooth.py
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
W = 25


def weighted(z, a):
    num = win_mean(z * a, -W, W)
    den = win_mean(a, -W, W)
    plain = win_mean(z, -W, W)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), plain)


def split(z, y):
    """+-W mean within runs of decoded alive frames (dead frames keep the plain mean)."""
    out = win_mean(z, -W, W)
    alive = y < 2
    idx = np.flatnonzero(alive)
    if len(idx):
        for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            out[r[0]:r[-1] + 1] = win_mean(z[r[0]:r[-1] + 1], -W, W)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        zs = [e2.logit(p) for p in raw[src]]
        v10 = [e2.sig(win_mean(z, -W, W)) for z in zs]
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            first = e2.decode_all(dec, pds, v10)
            phs = {"v10": v10,
                   "alive-weighted": [e2.sig(weighted(z, 1 - np.asarray(pd_, np.float64))) for z, pd_ in zip(zs, pds)],
                   "split at D": [e2.sig(split(z, y)) for z, y in zip(zs, first)]}
            res = {k: e2.score(src, conv, e2.decode_all(dec, pds, ph)) for k, ph in phs.items()}
            rows[ctx] = {k: v.mean() for k, v in res.items()}
            ref = res["v10"]
            print("%-24s v10 %.4f | %s" % (ctx, ref.mean(), " | ".join(
                "%s %+.4f (%d/%d)" % (k, v.mean() - ref.mean(), int((v > ref).sum()), len(v))
                for k, v in res.items() if k != "v10")), flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in ("v10", "alive-weighted", "split at D")}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
