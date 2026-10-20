"""Segment-level comparison of a prediction with the PUBLIC ground truth window (public data)."""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluate as ev
pm = ev.pm

def show(lab, lo=11185, hi=18685):
    gt = ev.public_gt(); lab = ev.to_codes(lab)
    runs = [(lo, hi)]
    G = pm.segments(gt, runs); Pp = pm.segments(lab, runs)
    print("GT segments (%d)                          | PRED segments (%d)" % (len(G), len(Pp)))
    for i in range(max(len(G), len(Pp))):
        g = G[i] if i < len(G) else None; p = Pp[i] if i < len(Pp) else None
        gs = "%6d-%6d %s %5.1fs" % (g[0], g[1], pm.LABELS[g[2]], (g[1]-g[0])/25) if g else ""
        ps = "%6d-%6d %s %5.1fs" % (p[0], p[1], pm.LABELS[p[2]], (p[1]-p[0])/25) if p else ""
        print("%-40s | %s" % (gs, ps))

if __name__ == "__main__":
    import model_v1
    f = pickle.load(open(os.path.join(os.path.dirname(HERE), "cache", "features_tactical.pkl"), "rb"))
    show(model_v1.predict(f, dict(pen_team=8, pen_dead=40)))
