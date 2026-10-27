"""Round 9 diagnostic: where does v10 still lose on the closest BAR-ATH proxy? Per spell-duration bin, recall of the
true spells and precision of v10's spells at IoU 0.5 (official greedy matching), for Metrica-ctrl and DFL-trk with
learned and true D; plus whether a missed true spell sits next to a stoppage (starts or ends at D) or inside play.
AGGREGATE output only.

  python -I src/exp_errbins.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402

EPS = 1e-4
BINS = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9))


def next_to_d(lab, a, b):
    return (a > 0 and lab[a - 1] == 2) or (b < len(lab) and lab[b] == 2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src, conv, dec in (("metrica@v3", "ctrl", (30.0, 8, 50)), ("dfl@v3", "trk", (32.0, 5, 25))):
        labs = bench.labels(src, conv)
        _, pdl = exp_oracle.probs(src)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            preds = e2.decode_all(dec, pds, v10[src])
            n_g, hit_g, n_p, hit_p = np.zeros(len(BINS)), np.zeros(len(BINS)), np.zeros(len(BINS)), np.zeros(len(BINS))
            miss_d = miss_play = 0
            kinds = {"after D only": 0, "before D only": 0, "between two D": 0}
            for g, p in zip(labs, preds):
                gs, ps = exp_oracle.spells(g), exp_oracle.spells(p)
                mg, mp = exp_oracle.matched(gs, ps, 0.5)
                for (a, b, _), ok in zip(gs, mg):
                    k = next(i for i, (lo, hi) in enumerate(BINS) if lo <= (b - a) / 25.0 < hi)
                    n_g[k] += 1
                    hit_g[k] += ok
                    if not ok:
                        if next_to_d(g, a, b):
                            miss_d += 1
                            after, before = a > 0 and g[a - 1] == 2, b < len(g) and g[b] == 2
                            kinds["between two D" if after and before else "after D only" if after
                                  else "before D only"] += 1
                        else:
                            miss_play += 1
                for (a, b, _), ok in zip(ps, mp):
                    k = next(i for i, (lo, hi) in enumerate(BINS) if lo <= (b - a) / 25.0 < hi)
                    n_p[k] += 1
                    hit_p[k] += ok
            cells = []
            for i, (lo, hi) in enumerate(BINS):
                name = "%g-%gs" % (lo, hi) if hi < 1e9 else ">%gs" % lo
                cells.append("%s true %d rec %.2f / pred %d prec %.2f" % (
                    name, n_g[i], hit_g[i] / max(n_g[i], 1), n_p[i], hit_p[i] / max(n_p[i], 1)))
            print("%-8s %-5s %-9s | %s | missed true spells: next to D %d (%s), inside play %d" % (
                src.split("@")[0], conv, dname, " | ".join(cells), miss_d,
                ", ".join("%s %d" % kv for kv in kinds.items()), miss_play), flush=True)


if __name__ == "__main__":
    main()
