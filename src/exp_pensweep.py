"""Round 9: re-choose pen_team for a cached alternative team model by the worst-case-regret protocol (as
src/exp_role_pen.py), v7 included. Alternative probabilities: cache/bench/<dir>/probs_<name>.pkl ({src: [p]}).
Sets: DFL-trk (learned D), Metrica-ctrl and SkillCorner phase / player-possession (learned / true D).
AGGREGATE output only.

  python -I src/exp_pensweep.py <dir> <name>      e.g.  trainvar trk
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_oracle  # noqa: E402
import exp_role_pen as rp  # noqa: E402

PENS = (2, 4, 6, 8, 12, 16, 24)
EPS = 1e-4


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    d, name = sys.argv[1], sys.argv[2]
    alt = bench._load(os.path.join(bench.CACHE, d, "probs_%s.pkl" % name))
    res = {}
    for src in ("dfl@v3", "metrica@v3", "sc@v3"):
        phl, pdl = exp_oracle.probs(src)
        bl = bench.blocks(src)
        dvars = [("learned D", pdl)]
        if src != "dfl@v3":
            dvars.append(("true D", [np.where(g == 2, 1 - EPS, EPS) for g in bench.labels(src, rp.CONVS[src][0])]))
        for dname, pds in dvars:
            for model, phs in (("v7", phl), (name, alt[src])):
                for pen in PENS:
                    preds = rp.decode_all(pen, rp.DEC[src], pds, phs)
                    for conv in rp.CONVS[src]:
                        r = bench.spell_by_group(bl, bench.labels(src, conv), preds)
                        res.setdefault("%s %s %s" % (src.split("@")[0], conv, dname), {})[model, pen] = float(
                            np.mean([r[k][0] for k in sorted(r)]))
            print("  %s %s done" % (src, dname), flush=True)
    sets = sorted(res)
    models = ("v7", name)
    print("%-26s %s" % ("set", " ".join("%s%-3d" % (m[0], p) for m in models for p in PENS)))
    for s in sets:
        print("%-26s %s" % (s, " ".join("%.3f" % res[s][m, p] for m in models for p in PENS)))
    regret = {(m, p): max(max(res[s].values()) - res[s][m, p] for s in sets) for m in models for p in PENS}
    best = sorted(regret.items(), key=lambda kv: kv[1])[:6]
    print("worst-case regret: " + " | ".join("%s pen %d %.4f" % (m, p, r) for (m, p), r in best))
    print("  v7 pen 12 regret %.4f" % regret["v7", 12])


if __name__ == "__main__":
    main()
