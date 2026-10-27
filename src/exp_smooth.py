"""Round 9 diagnostic: do the feature models that raised frame accuracy but lost spell F1 (roles, ball-free features)
just produce jumpier probabilities? Their cached team probabilities and v7's are smoothed in logit space (centred
mean over +-w frames) before the same v7 decoder; paired spell F1 against UNSMOOTHED v7 (learned and true D) on
DFL-trk, Metrica-ctrl, SkillCorner phase. AGGREGATE output only.

  python -I src/exp_smooth.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
from mlfeat import win_mean  # noqa: E402

EPS = 1e-4
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))
MODELS = {"roles": os.path.join(bench.CACHE, "role", "ph_role_%s_%%s.pkl" % exp_oracle.TAG),
          "ball-free": os.path.join(bench.CACHE, "bfree", "ph_%s.pkl")}
WS = (0, 6, 12, 25)


def smooth(ph, w):
    if w == 0:
        return ph
    return [e2.sig(win_mean(e2.logit(p), -w, w)) for p in ph]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, conv, dec in SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        models = {"v7": phl}
        for name, pat in MODELS.items():
            models[name] = bench._load(pat % src.replace("@", "-"))
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            cells = []
            for name, ph in models.items():
                for w in WS:
                    if name == "v7" and w == 0:
                        continue
                    v = e2.score(src, conv, e2.decode_all(dec, pds, smooth(ph, w)))
                    cells.append("%s w%d %+.4f (%d/%d)" % (name, w, v.mean() - ref.mean(), int((v > ref).sum()),
                                                         len(v)))
            print("%-10s %-6s %-9s v7 %.4f | %s" % (src, conv, dname, ref.mean(), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
