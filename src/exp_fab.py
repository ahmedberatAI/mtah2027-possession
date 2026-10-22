"""Round 8 (research a01 / a04 / a18 / a19): an external test of model_v7's clean_ball option. BAR-ATH's tactical ATD
fills ~3% of the seen ball frames by prediction (over-smooth runs ~1.7 m off); sim.V5 reproduces that on top of V3
(3% of live frames, median error 1.3 m), sim.V5C additionally drops the over-smooth runs (ballclean.fabricated) as
clean_ball does.

  A  v3-trained models on V5 inputs     (= v7 on BAR-ATH: trained without fabrication, fed the raw ATD ball)
  B  v3-trained models on V5C inputs    (= v7c: same models, fabricated runs dropped)
  C  V5-trained models on V5 inputs     (train with fabrication)
  D  V5C-trained models on V5C inputs   (train and run on cleaned balls)
Team + dead-ball models as exp_ballrate.py (v7 features; DFL leave-one-match-out, all-DFL models elsewhere); v7
decoder; DFL-trk, Metrica-ctrl, SkillCorner-phase; learned and true D; paired against A. AGGREGATE output only.

  python -I src/exp_fab.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_ballrate as eb  # noqa: E402
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "fab")
EPS = 1e-4


def cross_probs(train_var, eval_var):
    """{base: (team, dead)} of models trained on train_var's DFL, applied to eval_var's inputs."""
    path = os.path.join(OUT, "probs_%s_on_%s.pkl" % (train_var, eval_var))
    if os.path.isfile(path):
        return bench._load(path)
    fd = bench.features(model_v7.VERSION, bench.src_name("dfl", train_var))
    fe = bench.features(model_v7.VERSION, bench.src_name("dfl", eval_var))
    labs = bench.labels("dfl", "native")
    bl = bench.blocks(bench.src_name("dfl", train_var))
    ph, pdd = [None] * len(bl), [None] * len(bl)
    for m in sorted({b["group"] for b in bl}):
        tr = [i for i, b in enumerate(bl) if b["group"] != m]
        mt = bench.fit_team([fd[i] for i in tr], [labs[i] for i in tr], {"num_threads": bench.THREADS})
        md = bench.fit_dead([fd[i] for i in tr], [labs[i] for i in tr])
        for i, b in enumerate(bl):
            if b["group"] == m:
                ph[i] = bench.predict_team(mt, fe[i])
                pdd[i] = md.predict(fe[i]["XF"], num_threads=bench.THREADS)
        print("  %s->%s LOMO %s done" % (train_var, eval_var, m), flush=True)
    mt = bench.fit_team(fd, labs, {"num_threads": bench.THREADS})
    md = bench.fit_dead(fd, labs)
    del fd, fe
    res = {"dfl": (ph, pdd)}
    for base in eb.BASES[1:]:
        fs = bench.features(model_v7.VERSION, bench.src_name(base, eval_var))
        res[base] = ([bench.predict_team(mt, f) for f in fs], [md.predict(f["XF"], num_threads=bench.THREADS)
                                                              for f in fs])
        del fs
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    runs = [("A v3 models, V5 inputs", "v5", lambda: cross_probs("v3", "v5")),
            ("B v3 models, V5C inputs (clean)", "v5c", lambda: cross_probs("v3", "v5c")),
            ("C V5-trained, V5 inputs", "v5", lambda: eb.probs("v5")),
            ("D V5C-trained, V5C inputs", "v5c", lambda: eb.probs("v5c")),
            ("ref v3 models, V3 inputs", "v3", lambda: eb.probs("v3"))]
    ref = {}
    for name, var, fn in runs:
        res = fn()
        for base in eb.BASES:
            src = bench.src_name(base, var)
            labs = bench.labels(src, eb.CONV[base])
            ph, pdl = res[base]
            acc = np.mean(np.concatenate([((p > 0.5) == (l == 0))[l < 2] for p, l in zip(ph, labs)]))
            cells = []
            for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
                v = eb.score(src, eb.CONV[base], eb.decode_all(eb.DEC[base], pds, ph))
                if name.startswith("A "):
                    ref[base, dname] = v
                    cells.append("%s %.4f" % (dname, v.mean()))
                else:
                    r = ref[base, dname]
                    cells.append("%s %.4f (%+.4f vs A, better in %d/%d)" % (dname, v.mean(), v.mean() - r.mean(),
                                                                          int((v > r).sum()), len(v)))
            print("%-32s %-12s frame team acc %.4f | %s" % (name, base + "-" + eb.CONV[base], acc, " | ".join(cells)),
                  flush=True)


if __name__ == "__main__":
    main()
