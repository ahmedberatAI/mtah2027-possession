"""v22 against v21 on the target, aggregates only (no labels, the public window is not treated apart)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import atd
from exp_repair import spells, match_share
codes = {"H": 0, "A": 1, "D": 2, "X": 3}
def load(p):
    with open(p) as fh:
        next(fh)
        return np.array([codes[ln.strip().split(",")[1]] for ln in fh], np.int8)
a, b = load(sys.argv[2]), load(sys.argv[1])
g = atd.graded_mask()
print("graded frames that differ %.4f (team %.4f, live/dead %.4f)" % (np.mean(a[g] != b[g]), np.mean((a[g] != b[g]) & (a[g] < 2) & (b[g] < 2)), np.mean((a[g] != b[g]) & ((a[g] == 2) | (b[g] == 2)))))
sa, sb = [], []
for _, x, e in atd.HALVES:
    sa += [(u + x, w + x, k) for u, w, k in spells(a[x:e + 1])]
    sb += [(u + x, w + x, k) for u, w, k in spells(b[x:e + 1])]
da = np.array([w - u for u, w, _ in sa]) / 25
db = np.array([w - u for u, w, _ in sb]) / 25
print("possessions v21 %d (< 4 s %d), candidate %d (< 4 s %d); dead share %.4f / %.4f" % (len(sa), (da < 4).sum(), len(sb), (db < 4).sum(), np.mean(a[g] == 2), np.mean(b[g] == 2)))
print("spells of v21 with a same-team spell of the candidate at IoU >= .3 / .5 / .7: %.3f %.3f %.3f; the other way: %.3f %.3f %.3f" % (match_share(sa, sb, .3), match_share(sa, sb, .5), match_share(sa, sb, .7), match_share(sb, sa, .3), match_share(sb, sa, .5), match_share(sb, sa, .7)))
