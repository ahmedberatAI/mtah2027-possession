"""Build and cache the feature dicts: tactical-only and tactical+broadcast-ball."""
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import features, broadcast
CACHE = os.path.join(os.path.dirname(HERE), "cache")
t = time.time()
b = broadcast.build()
pickle.dump(b, open(os.path.join(CACHE, "broadcast_on_tactical.pkl"), "wb"), protocol=4)
f = features.build()
f.update(b)
pickle.dump(f, open(os.path.join(CACHE, "features_tactical.pkl"), "wb"), protocol=4)
override = np.where((b["b_seen"] & b["b_live"])[:, None], b["b_ball"], np.nan)
f2 = features.build(ball_override=override)
f2.update(b)
pickle.dump(f2, open(os.path.join(CACHE, "features_merged.pkl"), "wb"), protocol=4)
print("cached in %.1fs; merged ball coverage on graded: %.1f%% (tactical %.1f%%)" % (
    time.time() - t, 100 * f2["seen"][f2["graded"]].mean(), 100 * f["seen"][f["graded"]].mean()))
