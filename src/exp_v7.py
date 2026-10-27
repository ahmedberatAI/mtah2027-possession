"""Round-3 synthesis (docs/RAPOR.md): do the feature gains (own_half, C6, c145) and the B2 soft-minimum
decoder add up? Candidates x decoders on DFL-trk / Metrica-ctrl (sim V3), paired per match / game x seed
against base_native_v3 + the standard decoder at its own best penalty. AGGREGATE output only.

  python -I src/exp_v7.py <tag> [<tag> ...]
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
import decode2  # noqa: E402
import train_dfl as td  # noqa: E402

REF = "base_native_v3"
SETS = (("dfl", "trk", dict(pen_dead=32.0, min_poss=5, min_dead=25)),
        ("metrica", "ctrl", dict(pen_dead=30.0, min_poss=8, min_dead=50)))
B2 = dict(D0=175, c=48.0)
PENS_STD = (12, 20, 32, 48, 64, 96)
PENS_B2 = (4, 8, 12, 20)


def ramp(D0, c):
    j = np.arange(D0)
    return c * np.clip(1.0 - (j + 1) / float(D0), 0, None)


def team_probs(tag, base):
    info = json.load(open(os.path.join(bench.REPORTS, tag + ".json"), encoding="utf-8"))
    src = bench.src_name(base, "v3")
    if base == "dfl":
        return bench._load(os.path.join(bench.CACHE, "oof", tag + ".pkl"))
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    return [bench.predict_team(mdl, f) for f in bench.features(info["version"], src)]


def per_group(base, conv, pdl, phl, spec):
    src = bench.src_name(base, "v3")
    bl = bench.blocks(src)
    preds = [spec.decode(td.emissions(pd_, ph, 1.0, 1.0)).astype(np.int8) for pd_, ph in zip(pdl, phl)]
    r = bench.spell_by_group(bl, bench.labels(src, conv), preds)
    return np.array([r[k][0] for k in sorted(r)])


def curves(base, conv, dec, pdl, phl):
    out = {}
    for pt in PENS_STD:
        out[("std", pt)] = per_group(base, conv, pdl, phl,
                                     decode2.std_spec(pt, dec["pen_dead"], dec["min_poss"], dec["min_dead"]))
    for pt in PENS_B2:
        out[("b2", pt)] = per_group(base, conv, pdl, phl, decode2.duration_spec(
            pt, dec["pen_dead"], dec["min_poss"], dec["min_dead"], ramp(B2["D0"], B2["c"])))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    tags = [REF] + [t for t in sys.argv[1:] if t != REF]
    for base, conv, dec in SETS:
        src = bench.src_name(base, "v3")
        pdl = bench.dead_probs(src, "base")
        res = {t: curves(base, conv, dec, pdl, team_probs(t, base)) for t in tags}
        ref_std = max((k for k in res[REF] if k[0] == "std"), key=lambda k: res[REF][k].mean())
        r0 = res[REF][ref_std]
        print("\n%s vs %s | reference %s std @%d = %.4f" % (src, conv, REF, ref_std[1], r0.mean()))
        for t in tags:
            for kind in ("std", "b2"):
                keys = [k for k in res[t] if k[0] == kind]
                kb = max(keys, key=lambda k: res[t][k].mean())
                d = res[t][kb] - r0
                cells = " ".join("@%d %.4f" % (k[1], res[t][k].mean()) for k in keys)
                print("  %-58s %-3s best @%-2d %.4f  delta %+.4f (wins %d/%d) | %s" % (
                    t[:58], kind, kb[1], res[t][kb].mean(), d.mean(), int((d > 0).sum()), len(d), cells))


if __name__ == "__main__":
    main()
