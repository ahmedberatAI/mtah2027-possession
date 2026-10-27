"""Driver for the team-feature ideas (docs/RAPOR.md group C, src/feat_team2.py) on sim V3.
AGGREGATE output only.

Each version is "base+feat_team2:<fn>"; feat_team2 adds team-perspective (T) columns only, so the frame
features XF -- the dead-ball model's inputs -- are identical to "base". After checking that, the base V3
dead-ball cache is aliased for the version, so bench.run reuses the baseline's dead-ball probabilities
(no retraining, identical D evidence) and only the team model changes.

  BENCH_THREADS=6 python -I src/exp_features.py run c1 c3 c5 c2 c4     # bench.run per version
  BENCH_THREADS=6 python -I src/exp_features.py table c1 c3 ...        # deltas vs base_native_v3
  BENCH_THREADS=6 python -I src/exp_features.py groups c1 20           # per-match / per-game deltas
"""
import json
import os
import shutil
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402

VAR = "v3"
BASE_TAG = "base_native_v3"


def version_of(name):
    return "base" if name == "base" else "base+feat_team2:%s" % name


def tag_of(name, conv="native"):
    return "%s_%s_%s" % (bench._safe(version_of(name)), conv, VAR)


def ensure_dead_alias(version, var=VAR):
    dst = os.path.join(bench.CACHE, "dead", "%s_%s.pkl" % (bench._safe(version), var))
    if os.path.isfile(dst):
        return
    for src in (bench.src_name("dfl", var), bench.src_name("metrica", var), "barath"):
        fb, fv = bench.features("base", src), bench.features(version, src)
        same = all(a["fk"] == b["fk"] and np.array_equal(a["XF"], b["XF"], equal_nan=True) for a, b in zip(fb, fv))
        if not same:
            raise ValueError("%s changes the frame features on %s; the dead model must be retrained" % (version, src))
    shutil.copyfile(os.path.join(bench.CACHE, "dead", "base_%s.pkl" % var), dst)


def run(names, conv="native"):
    for nm in names:
        v = version_of(nm)
        ensure_dead_alias(v)
        bench.run(v, conv, var=VAR)


def _load_res(tag):
    with open(os.path.join(bench.REPORTS, tag + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


def _best(curve):
    k = max(curve, key=lambda p: curve[p][0])
    return curve[k][0], int(k)


def _at(curve, p):
    return curve[str(p)][0] if str(p) in curve else curve[p][0]


def _barath(tag):
    with open(os.path.join(bench.REPORTS, tag + ".txt"), encoding="utf-8") as fh:
        rows = [l for l in fh if "BAR-ATH v5 pen_team" in l]
    out = {}
    for l in rows:
        p = int(l.split("pen_team")[1].split("|")[0])
        sp = float(l.split("public spellF1")[1].split()[0])
        mf = float(l.split("macroF1")[1].split()[0])
        npos = int(l.split("full n_poss")[1].split()[0])
        out[p] = (sp, mf, npos)
    return out


def table(names, conv="native"):
    base = _load_res(BASE_TAG)
    bb = _barath(BASE_TAG)
    hdr = ("| version | DFL acc nat | DFL nat best | DFL acc trk | DFL trk best (pen) | trk@20 | trk@32 | "
           "Met ctrl best (pen) | ctrl@20 | ctrl@32 | Met events best | Met acc ctrl | BAR pub sp@8/@20 | BAR n_poss@8 |")
    lines = [hdr, "|" + "---|" * 14]

    def row(name, r, b):
        d_n, d_t = r["dfl"]["native"], r["dfl"]["trk"]
        mc, me = r["metrica"]["ctrl|v5"]["curve"], r["metrica"]["events|v5"]["curve"]
        bt, bp = _best(d_t["curve"])
        cb, cp = _best(mc)
        return [d_n["team_acc"], _best(d_n["curve"])[0], d_t["team_acc"], bt, bp, _at(d_t["curve"], 20),
                _at(d_t["curve"], 32), cb, cp, _at(mc, 20), _at(mc, 32), _best(me)[0], r["metrica"]["ctrl|team_acc"],
                b.get(8, (np.nan,))[0], b.get(20, (np.nan,))[0], b.get(8, (0, 0, 0))[2]]

    ref = row("base", base, bb)
    lines.append("| base V3 | %.4f | %.4f | %.4f | %.4f (%d) | %.4f | %.4f | %.4f (%d) | %.4f | %.4f | %.4f | %.4f | "
                 "%.4f/%.4f | %d |" % tuple(ref))
    for nm in names:
        tag = tag_of(nm, conv)
        if not os.path.isfile(os.path.join(bench.REPORTS, tag + ".json")):
            continue
        r = row(nm, _load_res(tag), _barath(tag))
        dl = [r[i] - ref[i] if i not in (4, 8, 15) else r[i] for i in range(len(r))]
        lines.append("| %s | %+.4f | %+.4f | %+.4f | %+.4f (%d) | %+.4f | %+.4f | %+.4f (%d) | %+.4f | %+.4f | %+.4f | "
                     "%+.4f | %+.4f/%+.4f | %d |" % ((nm,) + tuple(dl)))
    print("\n".join(lines))
    return lines


def _ph_metrica(tag, version):
    model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    return [bench.predict_team(model, f) for f in bench.features(version, bench.src_name("metrica", VAR))]


def groups(name, pen, conv="native"):
    """Per-match (DFL-trk) and per-game x seed (Metrica-ctrl, v5 decoder) spellF1 deltas at one pen_team."""
    sd, sm = bench.src_name("dfl", VAR), bench.src_name("metrica", VAR)
    pd_d, pd_m = bench.dead_probs(sd, "base"), bench.dead_probs(sm, "base")
    out = {}
    for nm in ("base", name):
        tag = BASE_TAG if nm == "base" else tag_of(nm, conv)
        ph_d = bench._load(os.path.join(bench.CACHE, "oof", tag + ".pkl"))
        ph_m = _ph_metrica(tag, version_of(nm))
        res = {}
        for src, pdl, phl, c, dec in ((sd, pd_d, ph_d, "trk", "dflcv"), (sm, pd_m, ph_m, "ctrl", "v5")):
            preds = [bench.decode(a, b, pen, bench.DECODERS[dec]) for a, b in zip(pdl, phl)]
            r = bench.spell_by_group(bench.blocks(src), bench.labels(src, c), preds)
            res[src] = {k: v[0] for k, v in r.items()}
        out[nm] = res
    for src in (sd, sm):
        keys = sorted(out["base"][src])
        d = np.array([out[name][src][k] - out["base"][src][k] for k in keys])
        print("%s pen %d: mean delta %+.4f, positive %d/%d, per group %s" % (
            src, pen, d.mean(), int((d > 0).sum()), len(d), " ".join("%+.3f" % x for x in d)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        run(args)
    elif cmd == "table":
        table(args)
    elif cmd == "groups":
        groups(args[0], int(args[1]))
