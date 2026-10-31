"""Round 19 (analyst ideas, idea 2): edge insertions are cheaper than interior ones.

Inserting a short spell inside a decoded possession adds two predicted spells (the possession is cut in three), while
an insertion that touches the possession's start or end (next to a stoppage or to the other team) adds one. With the
F1 accounting of round 18 (accept when 2 x matched gain > F x added spells) an edge insertion needs about half the
evidence of an interior one, yet v14's verifier uses one threshold (0.65) for both. Here v14's held-out verifier
probabilities (exp_verify.py probs4 at blend weight .5) are reused unchanged; only edge insertions get a lower
threshold, fixed in advance to 0.55 or 0.50. Ten contexts, 3 v10 seeds x 2 verifier seeds. AGGREGATE output only.

  python -I src/exp_edge.py eval VSEED[,VSEED]
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

T_INS, T_DEL = 0.65, 0.35
EDGES = (0.55, 0.50)


def is_edge(r):
    return r["typ"] == "ins" and (r["s"] == r["a"] or r["t"] == r["e"])


def verified(src, seed, dname, vseed, t_edge=None):
    """v14's decisions, with edge insertions accepted from t_edge instead of T_INS (None = v14)."""
    n0 = 0
    for s2, *_ in ev.SOURCES:
        if s2 == src:
            break
        n0 += len(ev._rows4(s2, seed, dname)["rows"])
    d = ev._rows4(src, seed, dname)
    pp = bench._load(ev._path("probs4", seed, dname, "q%d" % vseed))["G1"][n0:n0 + len(d["rows"])].copy()
    if t_edge is not None:
        edge = np.array([is_edge(r) for r in d["rows"]], bool)
        pp[edge] += T_INS - t_edge
    by_half = {}
    for n, r in enumerate(d["rows"]):
        by_half.setdefault(r["i"], []).append(n)
    return [ev.apply(pr, [d["rows"][n] for n in by_half.get(i, [])], pp[by_half.get(i, [])], True, True, T_INS, T_DEL)
            for i, pr in enumerate(d["base"])]


def evaluate(vseeds):
    import exp_blend_weight as bw
    bw.use_weight(0.5)
    cells = {}
    names = ["v14"] + ["edge %.2f" % t for t in EDGES]
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                n_edge = sum(is_edge(r) for r in ev._rows4(src, seed, dname)["rows"])
                if seed == 0:
                    print("  %s %s: %d edge insertion candidates" % (src.split("@")[0], dname, n_edge), flush=True)
                for vseed in vseeds:
                    outs = {"v14": verified(src, seed, dname, vseed)}
                    for t in EDGES:
                        outs["edge %.2f" % t] = verified(src, seed, dname, vseed, t)
                    for c in convs:
                        for name, preds in outs.items():
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    mean, regret = er.summarize(cells, names, ref="v14")
    for name in names[1:]:
        bw.regret_boot(mean, names, "v14", name)
        bw.regret_boot(mean, ["v14", name], "v14", name)
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "eval":
        return evaluate(tuple(int(x) for x in sys.argv[2].split(",")))
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
