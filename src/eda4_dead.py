"""Per-segment AGGREGATES of tracking cues inside the public GT dead-ball segments and around
their boundaries (public window only)."""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluate as ev
pm = ev.pm
sys.stdout.reconfigure(encoding="utf-8")
f = pickle.load(open(os.path.join(os.path.dirname(HERE), "cache", "features_tactical.pkl"), "rb"))
gt = ev.public_gt()
segs = pm.segments(gt, [(11185, 18685)])
seen, bs, bo, ed = f["seen"], f["bspeed"], f["ball_out"], f["edge_dist"]
spd = f["spd_all_s25"]; d1 = f["d1"]
print("per GT segment: label dur | ball seen | ball still(<0.5m/s) | ball out | min edge dist | mean spd | max spd(+-1s) | near<=1.5m")
for a, b, k in segs:
    m = slice(a, b)
    s = seen[m]
    still = (bs[m] < 0.5) & s
    print("%s %5.1fs | %4.0f%% | %4.0f%% | %4.0f%% | %5.1f | %4.2f | %4.2f | %4.0f%%" % (
        pm.LABELS[k], (b - a) / 25, 100 * s.mean(), 100 * still.sum() / max(1, s.sum()),
        100 * np.nanmean(np.where(s, bo[m] > 0, np.nan)) if s.any() else float('nan'),
        np.nanmin(ed[m]) if s.any() else float('nan'), np.nanmean(spd[m]), np.nanmax(spd[m]),
        100 * np.mean((d1[m] <= 1.5) & s)))
print()
print("boundaries of D segments: cues in the 3 s before / after the boundary")
for a, b, k in segs:
    if k != pm.D:
        continue
    for name, t in (("alive->D", a), ("D->alive", b)):
        pre, post = slice(t - 75, t), slice(t, t + 75)
        print("%-9s seen pre %3.0f%% post %3.0f%% | ball speed pre %4.1f post %4.1f | out pre %3.0f%% post %3.0f%% | spd pre %4.2f post %4.2f" % (
            name, 100 * seen[pre].mean(), 100 * seen[post].mean(), np.nanmean(bs[pre]), np.nanmean(bs[post]),
            100 * np.nanmean(np.where(seen[pre], bo[pre] > 0, np.nan)), 100 * np.nanmean(np.where(seen[post], bo[post] > 0, np.nan)),
            np.nanmean(spd[pre]), np.nanmean(spd[post])))
