"""Find the broadcast -> tactical frame offset(s) by matching team centroids. Aggregate output.

Hypothesis: the live parts of the broadcast run in real time, so live broadcast frame i shows
tactical frame i + offset (one offset per continuous recording block). Replays/close-ups break
the match and are themselves a dead-ball cue.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def centroids(d):
    P = d["players"]
    team = d["team"]
    out = {}
    with np.errstate(invalid="ignore"):
        for k in (0, 1):
            m = team == k
            out["x%d" % k] = np.nanmean(P[:, m, 0], axis=1)
            out["y%d" % k] = np.nanmean(P[:, m, 1], axis=1)
            out["n%d" % k] = (~np.isnan(P[:, m, 0])).sum(1)
    out["bx"] = d["ball"][:, 0]
    out["by"] = d["ball"][:, 1]
    return out


def signal(c, valid):
    """Home-minus-away centroid x: a 1-D signal that moves with play, NaN where unusable."""
    s = c["x0"] - c["x1"]
    ok = valid & (c["n0"] >= 6) & (c["n1"] >= 6) & np.isfinite(s)
    return np.where(ok, s, np.nan)


def xcorr_offsets(a, b, max_lag, step=1):
    """For lags in [-max_lag, max_lag]: mean |a[i] - b[i+lag]| over overlapping valid frames,
    computed via FFT-free sliding on a downsampled grid (step frames)."""
    a = a[::step]
    b = b[::step]
    res = []
    for lag in range(-max_lag // step, max_lag // step + 1):
        if lag >= 0:
            x, y = a[:len(a) - lag] if lag else a, b[lag:]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        m = min(len(x), len(y))
        x, y = x[:m], y[:m]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 500:
            res.append((lag * step, np.inf, ok.sum()))
            continue
        res.append((lag * step, float(np.mean(np.abs(x[ok] - y[ok]))), int(ok.sum())))
    return res


if __name__ == "__main__":
    tac = atd.load("tactical")
    bro = atd.load("broadcast")
    hb = pd.read_csv(atd.paths("broadcast")["homography"], usecols=["valid"])["valid"].values.astype(bool)
    ct, cb = centroids(tac), centroids(bro)
    st = signal(ct, np.ones(len(ct["x0"]), bool))
    sb = signal(cb, hb)
    # coarse: broadcast frame i <-> tactical frame i + lag; tactical is longer, so lag >= 0 mostly
    n_t, n_b = len(st), len(sb)
    pad = np.full(n_t, np.nan)
    pad[:n_b] = sb
    coarse = xcorr_offsets(pad, st, max_lag=40000, step=25)
    coarse = sorted(coarse, key=lambda r: r[1])[:8]
    print("coarse best lags (tactical = broadcast + lag), mean |diff| of centroid-difference signal:")
    for lag, err, cnt in coarse:
        print("  lag {:>6}  err {:.4f}  n {}".format(lag, err, cnt))
    # per block: split the broadcast into chunks of 5 minutes and find each chunk's best lag
    print("\nper 5-min broadcast chunk: best lag and error (refined to 1 frame)")
    chunk = 7500
    rows = []
    for start in range(0, n_b, chunk):
        seg = np.full(n_t, np.nan)
        seg[start:min(n_b, start + chunk)] = sb[start:start + chunk]
        best = None
        for lag in range(0, 40001, 25):
            x = seg[:n_t - lag] if lag else seg
            y = st[lag:]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 1000:
                continue
            e = float(np.mean(np.abs(x[m] - y[m])))
            if best is None or e < best[1]:
                best = (lag, e, int(m.sum()))
        if best is None:
            print("  chunk {:>6}: no overlap".format(start))
            continue
        lag0 = best[0]
        for lag in range(max(0, lag0 - 30), lag0 + 31):
            x = seg[:n_t - lag] if lag else seg
            y = st[lag:]
            m = np.isfinite(x) & np.isfinite(y)
            e = float(np.mean(np.abs(x[m] - y[m])))
            if e < best[1]:
                best = (lag, e, int(m.sum()))
        rows.append((start, best))
        print("  chunk {:>6}-{:<6}: lag {:>6}  err {:.4f}  n {}".format(start, start + chunk, *best))
