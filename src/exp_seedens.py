"""Round 9: seed ensemble of the v9 team model -- the logits of three LightGBM seeds (0 / 1 / 2, src/exp_combo.py with
LGB_SEED) averaged, then smoothed over +-w frames (12 / 25) -- against v7 and v9 (seed 0, w 25); worst-case regret over
the eight contexts (DFL-trk, Metrica-ctrl, SkillCorner phase / player-possession; learned and true D).
AGGREGATE output only.

  python -I src/exp_seedens.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

EPS = 1e-4
D = os.path.join(bench.CACHE, "combo")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    seeds = [bench._load(os.path.join(D, n)) for n in ("probs_a2_bfree.pkl", "probs_a2_bfree_seed1.pkl",
                                                       "probs_a2_bfree_seed2.pkl")]
    ens = {src: [e2.sig(np.mean([e2.logit(s[src][i]) for s in seeds], 0)) for i in range(len(seeds[0][src]))]
           for src in seeds[0]}
    cands = {"v9": {k: exp_smooth.smooth(v, 25) for k, v in seeds[0].items()},
             "ens w12": {k: exp_smooth.smooth(v, 12) for k, v in ens.items()},
             "ens w25": {k: exp_smooth.smooth(v, 25) for k, v in ens.items()}}
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            rows[ctx] = {"v7": r7.mean()}
            cells = []
            for name, res in cands.items():
                v = e2.score(src, conv, e2.decode_all(dec, pds, res[src]))
                rows[ctx][name] = v.mean()
                cells.append("%s %+.4f (%d/%d)" % (name, v.mean() - r7.mean(), int((v > r7).sum()), len(v)))
            print("%-26s v7 %.4f | %s" % (ctx, r7.mean(), " | ".join(cells)), flush=True)
    names = list(next(iter(rows.values())))
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in names}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
