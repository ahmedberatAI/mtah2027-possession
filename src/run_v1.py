import os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import features, model_v1, evaluate as ev
sys.stdout.reconfigure(encoding="utf-8")
t = time.time()
cache = os.path.join(os.path.dirname(HERE), "cache", "features_tactical.pkl")
if os.path.isfile(cache) and "--refresh" not in sys.argv:
    f = pickle.load(open(cache, "rb"))
else:
    f = features.build(); pickle.dump(f, open(cache, "wb"), protocol=4)
print("features in %.1fs" % (time.time() - t))
t = time.time()
lab = model_v1.predict(f)
print("decode in %.1fs" % (time.time() - t))
print("public :", ev.fmt(ev.score(lab)))
print("full   :", ev.fmt(ev.diagnostics(lab)))
