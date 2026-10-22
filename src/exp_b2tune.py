"""B2 soft-minimum parameters for v7 (D0 frames, c cost, pen_team), on SkillCorner-phase (20 matches),
Metrica-ctrl and DFL-trk, paired against v7's defaults (175, 48, 12). Team model = v7's (bench combo tag);
dead-ball evidence = base V3 DFL model. AGGREGATE output only.

  python -I src/exp_b2tune.py
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402

TAG = "%s_native_v3" % bench._safe(model_v7.VERSION)
GRID = list(itertools.product((100, 175, 250, 375), (32.0, 48.0, 64.0, 96.0), (4.0, 8.0, 12.0, 20.0)))
REF = (175, 48.0, 12.0)
SETS = (("sc@v3", "phase", dict(pen_dead=30.0, min_poss=8, min_dead=50)),
        ("metrica@v3", "ctrl", dict(pen_dead=30.0, min_poss=8, min_dead=50)),
        ("dfl@v3", "trk", dict(pen_dead=32.0, min_poss=5, min_dead=25)))


def probs(src):
    if src == "dfl@v3":
        return bench._load(os.path.join(bench.CACHE, "oof", TAG + ".pkl")), bench.dead_probs(src, "base")
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", TAG + ".txt"))
    ph = [bench.predict_team(mdl, f) for f in bench.features(model_v7.VERSION, src)]
    pdl = bench.dead_ext("base", "v3", src) if src.startswith("sc") else bench.dead_probs(src, "base")
    return ph, pdl


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    allres = {}
    for src, conv, dec in SETS:
        phl, pdl = probs(src)
        S = [td.emissions(pd_, ph, 1.0, 1.0) for pd_, ph in zip(pdl, phl)]
        bl, gts = bench.blocks(src), bench.labels(src, conv)
        res = {}
        for D0, c, pt in GRID:
            sp = decode2.duration_spec(pt, dec["pen_dead"], dec["min_poss"], dec["min_dead"], exp_v7.ramp(D0, c))
            r = bench.spell_by_group(bl, gts, [sp.decode(s).astype(np.int8) for s in S])
            res[(D0, c, pt)] = np.array([r[k][0] for k in sorted(r)])
        allres[src] = res
        ref = res[REF]
        best = sorted(res, key=lambda k: -res[k].mean())[:6]
        print("\n%s vs %s: v7 default %s = %.4f" % (src, conv, REF, ref.mean()))
        for k in best:
            d = res[k] - ref
            print("  D0 %3d c %4.0f pen %4.0f  %.4f  %+.4f (wins %d/%d)" % (k + (res[k].mean(), d.mean(), int((d > 0).sum()),
                                                                           len(d))), flush=True)
    # configurations good everywhere: mean rank over the three sets
    ranks = {k: 0.0 for k in GRID}
    for src, res in allres.items():
        order = sorted(res, key=lambda k: -res[k].mean())
        for i, k in enumerate(order):
            ranks[k] += i / len(allres)
    print("\nbest mean rank over the three sets:")
    for k in sorted(ranks, key=ranks.get)[:8]:
        print("  D0 %3d c %4.0f pen %4.0f  mean rank %.1f | " % (k + (ranks[k],)) + " ".join(
            "%s %+.4f" % (src.split("@")[0], allres[src][k].mean() - allres[src][REF].mean()) for src in allres))
    json.dump({src: {"%d_%g_%g" % k: v.tolist() for k, v in res.items()} for src, res in allres.items()},
              open(os.path.join(bench.REPORTS, "b2tune.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
