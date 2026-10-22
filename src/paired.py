"""Paired comparison of bench candidates: per match (DFL) / per game x seed (Metrica) spellF1 differences
at each candidate's own best penalty and at fixed penalties. AGGREGATE output only.

  python -I src/paired.py <var> <tag_ref> <tag_new> [<tag_new2> ...]
tags are bench.run tags (cache/bench/oof/<tag>.pkl, cache/bench/models/<tag>.txt); the feature version is
read from reports/bench/<tag>.json.
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402

PLAN = (("dfl", "trk", "dflcv", (32, 48)), ("dfl", "native", "dflcv", (4, 8)),
        ("metrica", "ctrl", "v5", (20, 32)), ("metrica", "events", "v5", (8, 12)))


def per_group(src, pdl, phl, conv, dec, pt):
    bl = bench.blocks(src)
    preds = [bench.decode(pd_, ph, pt, bench.DECODERS[dec]) for pd_, ph in zip(pdl, phl)]
    r = bench.spell_by_group(bl, bench.labels(src, conv), preds)
    return np.array([r[k][0] for k in sorted(r)])


def probs(tag, var, base):
    info = json.load(open(os.path.join(bench.REPORTS, tag + ".json"), encoding="utf-8"))
    version = info["version"]
    src = bench.src_name(base, var)
    if base == "dfl":
        ph = bench._load(os.path.join(bench.CACHE, "oof", tag + ".pkl"))
    else:
        mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
        ph = [bench.predict_team(mdl, f) for f in bench.features(version, src)]
    return src, bench.dead_probs(src, "base"), ph


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    var, ref, news = sys.argv[1], sys.argv[2], sys.argv[3:]
    for base, conv, dec, fixed in PLAN:
        src, pdl, ph_ref = probs(ref, var, base)
        curves = {}
        for tag in [ref] + news:
            ph = ph_ref if tag == ref else probs(tag, var, base)[2]
            curves[tag] = {pt: per_group(src, pdl, ph, conv, dec, pt) for pt in bench.PEN_GRID}
        print("\n%s vs %s (%s decoder; dead-ball = base V3 model for all)" % (src, conv, dec))
        bref = max(curves[ref], key=lambda k: curves[ref][k].mean())
        for tag in news:
            bnew = max(curves[tag], key=lambda k: curves[tag][k].mean())
            d = curves[tag][bnew] - curves[ref][bref]
            cells = ["own best %d vs %d: %+.4f (wins %d/%d)" % (bnew, bref, d.mean(), int((d > 0).sum()), len(d))]
            for pt in fixed:
                d = curves[tag][pt] - curves[ref][pt]
                cells.append("@%d %+.4f (%d/%d)" % (pt, d.mean(), int((d > 0).sum()), len(d)))
            print("  %-36s %s" % (tag, " | ".join(cells)))


if __name__ == "__main__":
    main()
