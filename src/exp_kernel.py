"""Round 9: the shape and centre of v10's logit smoothing. v10 averages the team logits over a centred +-25-frame box;
here the same cached raw v10 probabilities (cache/bench/combo/probs_a2_bfree_wm2.pkl) are smoothed with other
kernels -- Gaussian, triangular, median, and the box shifted forward / backward in time (a lagging team model would
gain from looking ahead) -- before the same v7 decoder. Decode-only; paired spell F1 against v10 on the eight
contexts (DFL-trk, Metrica-ctrl, SkillCorner phase / pp; learned and true D) and worst-case regret.
AGGREGATE output only.

  python -I src/exp_kernel.py              (KERNEL_SET=mix: box + a share of the raw / +-6 logit)
"""
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
from mlfeat import win_mean  # noqa: E402

EPS = 1e-4


def conv(z, k):
    """NaN-aware normalised convolution of z with the (centred) kernel k."""
    ok = np.isfinite(z)
    num = np.convolve(np.where(ok, z, 0.0), k, "same")
    den = np.convolve(ok.astype(np.float64), k, "same")
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def gauss(s):
    t = np.arange(-int(3 * s), int(3 * s) + 1)
    return np.exp(-0.5 * (t / s) ** 2)


def tri(w):
    t = np.arange(-w, w + 1)
    return (w + 1 - np.abs(t)).astype(np.float64)


KERNELS = {"box 25 (v10)": lambda z: win_mean(z, -25, 25)}
for _s in (10, 15, 20):
    KERNELS["gauss %d" % _s] = (lambda s: lambda z: conv(z, gauss(s)))(_s)
for _w in (35, 50):
    KERNELS["tri %d" % _w] = (lambda w: lambda z: conv(z, tri(w)))(_w)
KERNELS["median 25"] = lambda z: median_filter(np.nan_to_num(z), size=51, mode="nearest")
KERNELS["median 12 + box 25"] = lambda z: win_mean(median_filter(np.nan_to_num(z), size=25, mode="nearest"), -25, 25)
for _d in (-8, -4, 4, 8):
    KERNELS["box 25 shift %+d" % _d] = (lambda d: lambda z: win_mean(z, -25 + d, 25 + d))(_d)
if os.environ.get("KERNEL_SET") == "mix":   # box + a share of the raw logit (a central spike: short-spell evidence)
    KERNELS = {"box 25 (v10)": KERNELS["box 25 (v10)"]}
    for _a in (0.1, 0.2, 0.35):
        KERNELS["box 25 + raw x%.2f" % _a] = (lambda a: lambda z: (1 - a) * win_mean(z, -25, 25) + a * z)(_a)
    for _a in (0.2, 0.35):
        KERNELS["box 25 + box 6 x%.2f" % _a] = (lambda a: lambda z: (1 - a) * win_mean(z, -25, 25)
                                                + a * win_mean(z, -6, 6))(_a)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))
    rows = {}
    for src, conv_, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv_)
        zs = [e2.logit(p) for p in raw[src]]
        phs = {name: [e2.sig(f(z)) for z in zs] for name, f in KERNELS.items()}
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv_, dname)
            res = {name: e2.score(src, conv_, e2.decode_all(dec, pds, ph)) for name, ph in phs.items()}
            ref = res["box 25 (v10)"]
            rows[ctx] = {name: v.mean() for name, v in res.items()}
            print("%-24s v10 %.4f | %s" % (ctx, ref.mean(), " | ".join(
                "%s %+.4f (%d/%d)" % (name, v.mean() - ref.mean(), int((v > ref).sum()), len(v))
                for name, v in res.items() if name != "box 25 (v10)")), flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in KERNELS}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
