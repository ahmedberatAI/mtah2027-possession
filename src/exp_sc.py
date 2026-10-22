"""Out-of-domain check on the 20 SkillCorner open matches (idea A3, docs/FIKIRLER.md): team models trained on
degraded DFL (bench tags) applied to SkillCorner degraded with sim V3, labels from SkillCorner's phases of
play (a control-based possession convention close to the public GT statistics). Standard decoder vs the B2
soft minimum, per match, paired against base_native_v3 + standard decoder at its own best penalty.
AGGREGATE output only.

  python -I src/exp_sc.py [--oracle-dead] [--pp] <tag> [<tag> ...]     (--pp: player-possession labels)
--oracle-dead decodes with the true D frames (the DFL dead-ball model transfers poorly to SkillCorner: frame
accuracy 0.90, spellF1 0.58 vs 0.75 with true D), which isolates the team side.
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
import exp_v7  # noqa: E402
import train_dfl as td  # noqa: E402

SRC = "sc@v3"
CONV = ["phase"]          # set from --pp in main()
REF = "base_native_v3"
DEC = dict(pen_dead=30.0, min_poss=8, min_dead=50)        # model_v5's decoder settings
PENS_STD = (8, 12, 20, 32, 48, 64, 96)
PENS_B2 = (4, 8, 12, 20, 32)


def team_probs(tag):
    info = json.load(open(os.path.join(bench.REPORTS, tag + ".json"), encoding="utf-8"))
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    return [bench.predict_team(mdl, f) for f in bench.features(info["version"], SRC)]


def per_match(pdl, phl, spec):
    bl = bench.blocks(SRC)
    preds = [spec.decode(td.emissions(pd_, ph, 1.0, 1.0)).astype(np.int8) for pd_, ph in zip(pdl, phl)]
    r = bench.spell_by_group(bl, bench.labels(SRC, CONV[0]), preds)
    return np.array([r[k][0] for k in sorted(r)]), np.array([r[k][1] for k in sorted(r)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    oracle = "--oracle-dead" in sys.argv
    conv = "pp" if "--pp" in sys.argv else "phase"
    CONV[0] = conv
    tags = [REF] + [t for t in sys.argv[1:] if t != REF and not t.startswith("--")]
    labs = bench.labels(SRC, conv)
    pdl = ([np.where(lab == 2, 1 - 1e-4, 1e-4) for lab in labs] if oracle
           else bench.dead_ext("base", "v3", SRC))
    g = np.concatenate([lab < 3 for lab in labs])
    ad = np.mean(np.concatenate([(pd_ > 0.5) == (lab == 2) for pd_, lab in zip(pdl, labs)])[g])
    print("SkillCorner (20 matches, sim V3), dead-ball evidence = %s: frame alive/dead acc %.4f" % (
        "TRUE D (oracle)" if oracle else "all-DFL base V3 model", ad))
    res = {}
    for t in tags:
        phl = team_probs(t)
        acc = np.mean([np.mean((ph[lab < 2] > 0.5) == (lab[lab < 2] == 0)) for ph, lab in zip(phl, labs)])
        cur = {}
        for pt in PENS_STD:
            cur[("std", pt)] = per_match(pdl, phl, decode2.std_spec(pt, DEC["pen_dead"], DEC["min_poss"],
                                                                    DEC["min_dead"]))
        for pt in PENS_B2:
            cur[("b2", pt)] = per_match(pdl, phl, decode2.duration_spec(
                pt, DEC["pen_dead"], DEC["min_poss"], DEC["min_dead"], exp_v7.ramp(175, 48.0)))
        res[t] = (acc, cur)
    ref_key = max((k for k in res[REF][1] if k[0] == "std"), key=lambda k: res[REF][1][k][0].mean())
    r0 = res[REF][1][ref_key][0]
    print("reference %s std @%d spellF1 %.4f" % (REF, ref_key[1], r0.mean()))
    for t in tags:
        acc, cur = res[t]
        for kind in ("std", "b2"):
            keys = [k for k in cur if k[0] == kind]
            kb = max(keys, key=lambda k: cur[k][0].mean())
            d = cur[kb][0] - r0
            print("  %-58s team acc %.4f | %-3s best @%-2d spellF1 %.4f macroF1 %.4f delta %+.4f (wins %d/%d) | %s" % (
                t[:58], acc, kind, kb[1], cur[kb][0].mean(), cur[kb][1].mean(), d.mean(), int((d > 0).sum()), len(d),
                " ".join("@%d %.4f" % (k[1], cur[k][0].mean()) for k in keys)), flush=True)


if __name__ == "__main__":
    main()
