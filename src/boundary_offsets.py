"""Offsets of predicted alive/dead boundaries vs public GT boundaries (public data only)."""
import os, sys, pickle, importlib, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, evaluate as ev
pm = ev.pm
f = pickle.load(open(os.path.join(os.path.dirname(HERE), "cache", "features_merged.pkl"), "rb"))
gt = ev.public_gt()
def alive_dead_offsets(lab):
    g = (gt == 2).astype(int); p = (ev.to_codes(lab) == 2).astype(int)
    out = []
    for kind, val in (("->D", 1), ("D->", 0)):
        gb = [t for t in range(11186, 18685) if g[t] == val and g[t - 1] != val]
        pb = [t for t in range(11186, 18685) if p[t] == val and p[t - 1] != val]
        for t in gb:
            c = [q - t for q in pb if abs(q - t) <= 250]
            out.append((kind, min(c, key=abs) if c else None))
    return out
for spec in sys.argv[1:]:
    name, _, params = spec.partition("=")
    lab = importlib.import_module(name).predict(f, json.loads(params) if params else {})
    r = ev.score(lab); offs = alive_dead_offsets(lab)
    print("%-45s macro %.3f spell %.3f adF1 %.3f | %s" % (spec[:45], r["macroF1"], r["spellF1"], r["alive_dead_f1"],
          " ".join("%s%s" % (k, ("%+d" % o) if o is not None else "miss") for k, o in offs)))
