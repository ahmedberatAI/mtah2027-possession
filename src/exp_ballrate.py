"""Does it matter that the ball is looked at on EVERY frame? (user question, 22 Oct)

The pipeline already reads the tracking frame by frame (25 fps): team model, dead-ball model and decoder all work
per frame. This measures what we would lose if the ball were only observed every k frames -- e.g. a video ball
detector run at a stride, linear interpolation in between (bench variants "v3b<k>"): k = 1 (V3 as is), 5 (0.2 s),
10 (0.4 s), 15 (0.6 s), 25 (1 s), 50 (2 s), 250 (10 s). Per variant the team and dead-ball models are retrained
(v7 features, DFL native labels; DFL leave-one-match-out, all-DFL models for Metrica / SkillCorner) and decoded
with the v7 decoder (B2, pen 12). Scored on DFL-trk, Metrica-ctrl and SkillCorner-phase with learned D and true D,
paired against k = 1, plus frame team accuracy and the share of alive frames with a ball. AGGREGATE output only.

  python -I src/exp_ballrate.py [v3 v3b10 ...]     (the first variant is the reference; default: all)
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402

VARS = ("v3", "v3b5", "v3b10", "v3b15", "v3b25", "v3b50", "v3b250")


def var_name(var):
    k = int(var[3:]) if var.startswith("v3b") else 1
    return "every frame" if k == 1 else "every %d (%.1f s)" % (k, k / 25.0)
BASES = ("dfl", "metrica", "sc")
CONV = {"dfl": "trk", "metrica": "ctrl", "sc": "phase"}
DEC = {"dfl": (32.0, 5, 25), "metrica": (30.0, 8, 50), "sc": (30.0, 8, 50)}
OUT = os.path.join(bench.CACHE, "ballrate")
EPS = 1e-4


def probs(var):
    """{base: (team probabilities, dead probabilities)} -- DFL out of fold, the rest from the all-DFL models."""
    path = os.path.join(OUT, "probs_%s.pkl" % var)
    if os.path.isfile(path):
        return bench._load(path)
    sd = bench.src_name("dfl", var)
    fd = bench.features(model_v7.VERSION, sd)
    labs = bench.labels("dfl", "native")
    bl = bench.blocks(sd)
    ph, pdd = [None] * len(bl), [None] * len(bl)
    for m in sorted({b["group"] for b in bl}):
        tr = [i for i, b in enumerate(bl) if b["group"] != m]
        mt = bench.fit_team([fd[i] for i in tr], [labs[i] for i in tr], {"num_threads": bench.THREADS})
        md = bench.fit_dead([fd[i] for i in tr], [labs[i] for i in tr])
        for i, b in enumerate(bl):
            if b["group"] == m:
                ph[i] = bench.predict_team(mt, fd[i])
                pdd[i] = md.predict(fd[i]["XF"], num_threads=bench.THREADS)
        print("  %s LOMO %s done" % (var, m), flush=True)
    mt = bench.fit_team(fd, labs, {"num_threads": bench.THREADS})
    md = bench.fit_dead(fd, labs)
    del fd
    res = {"dfl": (ph, pdd)}
    for base in BASES[1:]:
        fs = bench.features(model_v7.VERSION, bench.src_name(base, var))
        res[base] = ([bench.predict_team(mt, f) for f in fs], [md.predict(f["XF"], num_threads=bench.THREADS)
                                                              for f in fs])
        del fs
    bench._dump(res, path)
    return res


def decode_all(dec, pds, phs):
    sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
    return [sp.decode(td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0)).astype(np.int8)
            for pd_, ph in zip(pds, phs)]


def score(src, conv, preds):
    r = bench.spell_by_group(bench.blocks(src), bench.labels(src, conv), preds)
    return np.array([r[k][0] for k in sorted(r)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    ref = {}
    variants = sys.argv[1:] or VARS
    for var in variants:
        name = var_name(var)
        res = probs(var)
        for base in BASES:
            src = bench.src_name(base, var)
            labs = bench.labels(src, CONV[base])
            ph, pdl = res[base]
            seen = np.mean(np.concatenate([np.isfinite(b["B"][:, 0])[l < 2] for b, l in zip(bench.blocks(src), labs)]))
            acc = np.mean(np.concatenate([((p > 0.5) == (l == 0))[l < 2] for p, l in zip(ph, labs)]))
            cells = []
            for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
                v = score(src, CONV[base], decode_all(DEC[base], pds, ph))
                if var == variants[0]:
                    ref[base, dname] = v
                    cells.append("%s %.4f" % (dname, v.mean()))
                else:
                    r = ref[base, dname]
                    cells.append("%s %.4f (%+.4f, better in %d/%d)" % (dname, v.mean(), v.mean() - r.mean(),
                                                                     int((v > r).sum()), len(v)))
            print("%-17s %-8s ball on %.3f of alive frames | frame team acc %.4f | spellF1 %s" % (
                name, base + "-" + CONV[base], seen, acc, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
