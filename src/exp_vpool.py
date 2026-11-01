"""Round 20: more data and less variance for v14's verifier.

v14's verifier is trained, per v10 seed, on that seed's candidates only (both dead channels pooled, 30 matches,
leave-one-match-out). Here (a) one pooled model per match learns from the candidates of all three v10 seeds (three
correlated but different decodes, ~3x the rows), still never from the match it scores; (b) the probabilities of three
verifier seeds are averaged. Same features (G1), thresholds 0.65 / 0.35. Ten contexts. AGGREGATE output only.

  python -I src/exp_vpool.py trainpool VSEED     pooled-over-seeds leave-one-match-out probabilities
  python -I src/exp_vpool.py eval                v14 / v14 seed-averaged / pooled / pooled seed-averaged
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_verify as ev  # noqa: E402

VSEEDS = (0, 1, 2)


def _setup():
    import exp_blend_weight as bw
    bw.use_weight(0.5)


def _rows(seed, dname):
    return [r for src, *_ in ev.SOURCES for r in ev._rows4(src, seed, dname)["rows"]]


def trainpool(vseed):
    _setup()
    blocks, cut = [], [0]
    for seed in es.SEEDS:
        for dname in ev.DEADS:
            blocks += _rows(seed, dname)
            cut.append(len(blocks))
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    pp = ev.predict_lomo(blocks, ev.FEATSETS["G1"], params)
    k = 0
    for seed in es.SEEDS:
        for dname in ev.DEADS:
            bench._dump({"G1": pp[cut[k]:cut[k + 1]]}, ev._path("vpool", seed, dname, "q%d" % vseed))
            k += 1
    print("  q%d pooled-over-seeds verifier written (%d rows)" % (vseed, len(blocks)), flush=True)


def _probs(kind, seed, dname, vseeds):
    return np.mean([bench._load(ev._path(kind, seed, dname, "q%d" % q))["G1"] for q in vseeds], axis=0)


def _apply(src, seed, dname, pp_all):
    n0 = 0
    for s2, *_ in ev.SOURCES:
        if s2 == src:
            break
        n0 += len(ev._rows4(s2, seed, dname)["rows"])
    d = ev._rows4(src, seed, dname)
    pp = pp_all[n0:n0 + len(d["rows"])]
    by_half = {}
    for n, r in enumerate(d["rows"]):
        by_half.setdefault(r["i"], []).append(n)
    return [ev.apply(pr, [d["rows"][n] for n in by_half.get(i, [])], pp[by_half.get(i, [])], True, True, 0.65, 0.35)
            for i, pr in enumerate(d["base"])]


def evaluate():
    import exp_blend_weight as bw
    _setup()
    cells = {}
    names = ["v14", "v14 avg3", "pooled", "pooled avg3"]
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                runs = {"v14": [_probs("probs4", seed, dname, (q,)) for q in (0, 1)],
                        "v14 avg3": [_probs("probs4", seed, dname, VSEEDS)],
                        "pooled": [_probs("vpool", seed, dname, (q,)) for q in (0, 1)],
                        "pooled avg3": [_probs("vpool", seed, dname, VSEEDS)]}
                for name, plist in runs.items():
                    for pp in plist:
                        preds = _apply(src, seed, dname, pp)
                        for c in convs:
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    mean, regret = er.summarize(cells, names, ref="v14")
    for name in names[1:]:
        bw.regret_boot(mean, ["v14", name], "v14", name)
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "trainpool":
        return trainpool(int(sys.argv[2]))
    if sys.argv[1] == "eval":
        return evaluate()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
