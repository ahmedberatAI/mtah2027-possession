"""Broadcast -> tactical alignment using the ball position (aggregate output only)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import atd
sys.stdout.reconfigure(encoding="utf-8")
tac, bro = atd.load("tactical"), atd.load("broadcast")
hb = pd.read_csv(atd.paths("broadcast")["homography"], usecols=["valid"])["valid"].values.astype(bool)
bt = atd.to_metres(tac["ball"][:, :2]); bb = atd.to_metres(bro["ball"][:, :2])
bb[~hb] = np.nan
n_t, n_b = len(bt), len(bb)

def cost(seg_idx, lag):
    j = seg_idx + lag
    ok = (j >= 0) & (j < n_t)
    i, j = seg_idx[ok], j[ok]
    d = np.linalg.norm(bb[i] - bt[j], axis=1)
    d = d[np.isfinite(d)]
    if len(d) < 150: return np.inf, len(d), np.nan
    return float(np.median(d)), len(d), float(np.mean(d < 2.0))

chunk = 3000   # 2 minutes
print("chunk(broadcast frames) | best lag | median dist m | n overlap | share < 2 m | 2nd best lag (median)")
res = []
for start in range(0, n_b, chunk):
    idx = np.arange(start, min(n_b, start + chunk))
    idx = idx[np.isfinite(bb[idx, 0])]
    if len(idx) < 200: print("%6d: too few broadcast ball frames" % start); continue
    sub = idx[::3]
    scores = []
    for lag in range(-2000, 42001, 10):
        c, m, s = cost(sub, lag)
        scores.append((c, lag))
    scores.sort()
    best_lag = scores[0][1]
    fine = sorted((cost(idx, l)[0], l) for l in range(best_lag - 12, best_lag + 13))
    c, m, s = cost(idx, fine[0][1])
    second = next(((c2, l2) for c2, l2 in scores if abs(l2 - best_lag) > 250), (np.nan, None))
    res.append((start, fine[0][1], c, s))
    print("%6d-%-6d | %7d | %6.2f | %5d | %5.2f | %s (%.2f)" % (start, start + chunk, fine[0][1], c, m, s, second[1], second[0]))
