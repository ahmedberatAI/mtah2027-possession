"""How useful is the aligned broadcast? Aggregates only."""
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import broadcast, evaluate as ev
from eda import pct
sys.stdout.reconfigure(encoding="utf-8")
t = time.time()
b = broadcast.build()
pickle.dump(b, open(os.path.join(os.path.dirname(HERE), "cache", "broadcast_on_tactical.pkl"), "wb"), protocol=4)
print("built in %.1fs" % (time.time() - t))
f = pickle.load(open(os.path.join(os.path.dirname(HERE), "cache", "features_tactical.pkl"), "rb"))
g = f["graded"]; seen = f["seen"]
print("graded frames with a broadcast frame: %.1f%%; broadcast homography valid: %.1f%%; live-agreeing: %.1f%%" % (
    100 * b["b_have"][g].mean(), 100 * b["b_valid"][g].mean(), 100 * b["b_live"][g].mean()))
print("player agreement (m) where valid: " + pct(b["b_agree"][g & b["b_valid"]], (10, 50, 75, 90, 95)))
both = g & seen & b["b_seen"] & b["b_live"]
dist = np.linalg.norm(f["bxy"][both] - b["b_ball"][both], axis=1)
print("ball disagreement tactical vs broadcast (m) when both seen & live: " + pct(dist, (25, 50, 75, 90, 95)))
gap = g & ~seen
fill = gap & b["b_seen"] & b["b_live"]
print("tactical ball missing on %d graded frames; broadcast(live) ball available on %d of them (%.1f%%)" % (
    gap.sum(), fill.sum(), 100 * fill.sum() / gap.sum()))
print("combined ball coverage on graded frames: %.1f%% (tactical alone %.1f%%)" % (
    100 * np.mean((seen | (b["b_seen"] & b["b_live"]))[g]), 100 * seen[g].mean()))
gt = ev.public_gt(); win = gt != 3
for k, name in enumerate("HAD"):
    m = win & (gt == k)
    print("public GT %s: broadcast valid %.0f%%  live %.0f%%  broadcast ball %.0f%%  tactical-gap filled by broadcast %.0f%%" % (
        name, 100 * b["b_valid"][m].mean(), 100 * b["b_live"][m].mean(), 100 * (b["b_seen"] & b["b_live"])[m].mean(),
        100 * np.mean((b["b_seen"] & b["b_live"])[m & ~seen]) if (m & ~seen).any() else float('nan')))
