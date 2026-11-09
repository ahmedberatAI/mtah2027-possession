"""Aggregate only: W3 (v23) against v22 per context from round 96's per-match cells (means, paired 95% CI, matches ahead)."""
import os, pickle, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import exp_refine as er
cells = pickle.load(open(os.path.join(ROOT, "cache", "lab87", "op", "r96_cells.pkl"), "rb"))
rows = [("dfl", "evctrl"), ("metrica", "ctrl"), ("dfl", "trk"), ("sc", "phase"), ("sc", "pp"), ("sc", "evctrl"), ("st", "evctrl"), ("st", "native")]
for cond in ("IP", "OPs"):
    for K, c in rows:
        if (K, cond, "v22") not in cells:
            continue
        a, b = cells[(K, cond, "v22")][c], cells[(K, cond, "W3")][c]
        ms = sorted(a)
        x, y = np.array([a[m] for m in ms]), np.array([b[m] for m in ms])
        lo, hi = er.boot(y - x)
        print("%-4s %-8s %-7s v22 %.4f  v23 %.4f  diff %+.4f [%+.4f, %+.4f]  ahead %d/%d" % (cond, K, c, x.mean(), y.mean(), (y - x).mean(), lo, hi, int((y > x).sum()), len(ms)))
