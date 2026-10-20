"""How do the public-GT transitions line up with what the tracking shows? Aggregate offsets only.

For each GT transition type we measure, in frames relative to the GT transition frame:
  H<->A : first frame (searching -50..+75) at which the nearest player within R m is of the new team
          and the last frame before it where the nearest within R m was the old team
  alive->D : first frame the ball is outside the pitch / disappears
  D->alive : first frame the ball moves faster than v m/s, and the team of the nearest player there
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "external", "more-than-a-hack-2027", "evaluation"))
import atd  # noqa: E402
import possession_metrics as pm  # noqa: E402
from eda import pct, load_gt, nearest  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
L, W = atd.PITCH
d = atd.load("tactical")
P, B, team = d["players"], d["ball"], d["team"]
gt = load_gt()
runs = pm.valid_runs(gt)
trans = pm.transitions(gt, runs)
ball_seen = ~np.isnan(B[:, 0])
d1, t1, dh, da, d2 = nearest(P, B, team)
bxy = atd.to_metres(B[:, :2])
vel = np.full(len(B), np.nan)
vel[1:] = np.linalg.norm(np.diff(bxy, axis=0), axis=1) * atd.FPS
outside = ball_seen & ((B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1))

for R in (1.0, 1.5, 2.5):
    first_new, last_old = [], []
    for f, a, b in trans:
        if a == pm.D or b == pm.D:
            continue
        lo, hi = f - 50, f + 75
        idx = np.arange(lo, hi)
        close = ball_seen[idx] & (d1[idx] <= R)
        new = idx[close & (t1[idx] == b)]
        old = idx[close & (t1[idx] == a) & (idx < (new[0] if len(new) else hi))]
        first_new.append(new[0] - f if len(new) else np.nan)
        last_old.append(old[-1] - f if len(old) else np.nan)
    print("H<->A, R={} m: first new-team close frame - GT frame: {}".format(R, pct(first_new, (0, 25, 50, 75, 100))))
    print("              last old-team close frame - GT frame:  {}".format(pct(last_old, (0, 25, 50, 75, 100))))

offs_out, offs_gone = [], []
for f, a, b in trans:
    if b != pm.D:
        continue
    idx = np.arange(f - 75, f + 75)
    o = idx[outside[idx]]
    g = idx[~ball_seen[idx]]
    offs_out.append(o[0] - f if len(o) else np.nan)
    offs_gone.append(g[0] - f if len(g) else np.nan)
print("alive->D: first ball-outside frame - GT frame: " + pct(offs_out, (0, 50, 100)))
print("alive->D: first ball-missing frame - GT frame: " + pct(offs_gone, (0, 50, 100)))

for v in (2.0, 4.0):
    offs, agree = [], []
    for f, a, b in trans:
        if a != pm.D:
            continue
        idx = np.arange(f - 125, f + 75)
        m = idx[ball_seen[idx] & (vel[idx] > v)]
        offs.append(m[0] - f if len(m) else np.nan)
        if len(m):
            agree.append(int(t1[m[0]] == b))
    print("D->alive: first ball speed > {} m/s frame - GT frame: {}; nearest team == restarting team {}/{}".format(
        v, pct(offs, (0, 50, 100)), sum(agree), len(agree)))

# per-frame: during GT possession, how often is the nearest player (<=1.5 m) of the OTHER team?
win = gt != pm.X
for k, name in ((0, "H"), (1, "A")):
    m = win & (gt == k) & ball_seen & (d1 <= 1.5)
    print("GT {}: frames with a player <=1.5 m: {}, of which nearest is other team {:.1%}".format(
        name, m.sum(), (t1[m] != k).mean()))
