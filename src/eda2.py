"""Second EDA pass: ball gaps, the ball speed channel, goalkeeper slots, dead-ball signatures.
AGGREGATE output only."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "external", "more-than-a-hack-2027", "evaluation"))
import atd  # noqa: E402
import possession_metrics as pm  # noqa: E402
from eda import pct, runs_of, load_gt, nearest  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
L, W = atd.PITCH

d = atd.load("tactical")
P, B, team = d["players"], d["ball"], d["team"]
graded = atd.graded_mask()
gt = load_gt()
win = gt != pm.X
n = len(B)
ball_seen = ~np.isnan(B[:, 0])

print("=" * 100)
print("1. BALL SPEED CHANNEL")
bs_seen = ~np.isnan(B[:, 2])
print("ball speed present on {} frames; of which ball position present: {:.1%}".format(
    bs_seen.sum(), ball_seen[bs_seen].mean()))
minute = np.arange(n) // (60 * atd.FPS)
mins = np.unique(minute[bs_seen])
print("minutes of video containing a ball speed value: {} distinct minutes".format(len(mins)))
calc = np.full(n, np.nan)
bxy = atd.to_metres(B[:, :2])
step = np.linalg.norm(np.diff(bxy, axis=0), axis=1) * atd.FPS
calc[1:] = step
both = bs_seen & np.isfinite(calc)
if both.any():
    r = np.corrcoef(B[both, 2], calc[both])[0, 1]
    print("corr(ball speed channel, speed from positions) = {:.3f}; ratio median {:.2f}".format(
        r, np.median(calc[both] / np.maximum(B[both, 2], 1e-3))))
ps = P[:, :, 2]
pxy = atd.to_metres(P[:, :, :2])
pstep = np.linalg.norm(np.diff(pxy, axis=0), axis=2) * atd.FPS
ok = np.isfinite(pstep) & np.isfinite(ps[1:])
print("player speed channel vs position-derived speed (m/s): corr {:.3f}; median ratio {:.3f}".format(
    np.corrcoef(ps[1:][ok], pstep[ok])[0, 1], np.median(pstep[ok] / np.maximum(ps[1:][ok], 1e-3))))
print("position-derived player speed (m/s, graded): " + pct(pstep[graded[1:] & graded[:-1]][
    np.isfinite(pstep[graded[1:] & graded[:-1]])], (5, 50, 95, 99)))

print("=" * 100)
print("2. BALL GAPS IN GRADED PLAY")
gaps = [(a, b) for a, b in runs_of(~ball_seen & graded)]
gaps = [(a, b) for a, b in gaps if a > 0 and b < n and ball_seen[a - 1] and ball_seen[b]]
lens = np.array([b - a for a, b in gaps])
print("gaps bounded by visible ball on both sides: {}".format(len(gaps)))
before = bxy[[a - 1 for a, b in gaps]]
after = bxy[[b for a, b in gaps]]
jump = np.linalg.norm(after - before, axis=1)
print("ball displacement across a gap (m): " + pct(jump, (10, 25, 50, 75, 90)))
def edge_dist(xy):
    return np.minimum.reduce([xy[:, 0], L - xy[:, 0], xy[:, 1], W - xy[:, 1]])
eb, ea = edge_dist(before), edge_dist(after)
print("distance of last-seen ball to nearest pitch line (m), negative = outside: " +
      pct(eb, (10, 25, 50, 75, 90)))
print("distance of first-seen ball after gap to nearest line (m): " + pct(ea, (10, 25, 50, 75, 90)))
print("share of gaps whose last-seen ball is within 2 m of a line or outside: {:.1%}".format(
    np.mean(eb < 2)))
print("share of gaps whose first-seen ball is within 2 m of a line or outside: {:.1%}".format(
    np.mean(ea < 2)))
goal_area = lambda xy: ((xy[:, 0] < 16.5) | (xy[:, 0] > L - 16.5)) & (np.abs(xy[:, 1] - W / 2) < 20.16)
print("share of gaps whose first-seen ball is inside a penalty box: {:.1%}".format(goal_area(after).mean()))
mean_spd = np.nanmean(np.where(np.isnan(P[:, :, 2]), np.nan, P[:, :, 2]), axis=1)
gap_spd = np.array([np.nanmean(mean_spd[a:b]) for a, b in gaps])
vis_spd = mean_spd[graded & ball_seen]
print("mean player speed during gaps (m/s): " + pct(gap_spd, (10, 25, 50, 75, 90)))
print("mean player speed while ball visible (m/s): " + pct(vis_spd, (10, 25, 50, 75, 90)))

# public window composition of gap frames
gap_mask = ~ball_seen & graded
for k, name in enumerate(("H", "A", "D")):
    m = win & (gt == k)
    print("public window GT {}: {:.1%} of its frames are inside ball gaps".format(name, gap_mask[m].mean()))
m = win & gap_mask
print("public window gap frames by GT label: H {:.1%} A {:.1%} D {:.1%}".format(
    (gt[m] == 0).mean(), (gt[m] == 1).mean(), (gt[m] == 2).mean()))

print("=" * 100)
print("3. GOALKEEPER SLOTS")
gk = np.flatnonzero(d["slots"]["position_type"].values == "Goalkeeper")
for s in gk:
    occ = ~np.isnan(P[graded, s, 0])
    x = P[graded, s, 0][occ] * L
    print("slot {} team {}: occupancy {:.1%}, x (m) ".format(s, team[s], occ.mean()) +
          pct(x, (5, 50, 95)))

print("=" * 100)
print("4. DEAD-BALL SIGNATURES OVER THE WHOLE MATCH (no labels)")
# smoothed mean player speed: 2 s window
k = 50
ms = np.where(np.isnan(mean_spd), np.nanmean(mean_spd[graded]), mean_spd)
cs = np.concatenate(([0], np.cumsum(ms)))
idx = np.arange(n)
lo, hi = np.maximum(0, idx - k), np.minimum(n, idx + k + 1)
smooth = (cs[hi] - cs[lo]) / (hi - lo)
for thr in (0.9, 1.1, 1.3, 1.5):
    for min_len in (75, 125, 250):
        r = [(a, b) for a, b in runs_of((smooth < thr) & graded) if b - a >= min_len]
        print("  slow stretches (mean speed < {} m/s, >= {:.0f} s): {:3d}  covering {:6.0f} s".format(
            thr, min_len / 25, len(r), sum(b - a for a, b in r) / 25))
for kname, k_ in (("H", 0), ("A", 1), ("D", 2)):
    mm = win & (gt == k_)
    print("smoothed (+-2 s) mean speed in public GT {}: ".format(kname) + pct(smooth[mm], (10, 50, 90)))

# ball outside the pitch
bo = ball_seen & ((B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1)) & graded
r = [(a, b) for a, b in runs_of(bo) if b - a >= 5]
print("\nball-outside-pitch episodes (>=0.2 s): {}  total {:.0f} s".format(len(r), sum(b - a for a, b in r) / 25))
d1, t1, dh, da, d2 = nearest(P, B, team)
alive_ok = ball_seen & graded
print("frames ball visible & nearest player > 15 m: {:.2%}".format(np.mean(d1[alive_ok] > 15)))
