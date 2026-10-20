import os, sys, pickle, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import model_v1, evaluate as ev
sys.stdout.reconfigure(encoding="utf-8")
f = pickle.load(open(os.path.join(os.path.dirname(HERE), "cache", "features_tactical.pkl"), "rb"))
print("%6s %6s %5s %5s | %7s %7s %4s | %5s %6s %5s %6s" % ("pteam","pdead","minp","mind","macro","spell","np","Nposs","mean_s","Ndead","dshare"))
for pt, pd_, mp, md in itertools.product([3, 8, 15, 30], [10, 40], [10], [50]):
    lab = model_v1.predict(f, dict(pen_team=pt, pen_dead=pd_, min_poss=mp, min_dead=md))
    s = ev.score(lab); g = ev.diagnostics(lab)
    print("%6g %6g %5d %5d | %7.4f %7.4f %4d | %5d %6.1f %5d %6.3f" % (pt, pd_, mp, md, s["macroF1"], s["spellF1"], s["n_pred"], g["n_poss"], g["poss_mean_s"], g["n_dead_segs"], g["dead_share"]))
