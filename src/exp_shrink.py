"""Round 19 (analyst idea): correct the known width bias of found short possessions.

Round 15 measured that found short spells are wider than the truth -- in-play ones start 10-36 frames early and end
9-26 frames late, those right after a stoppage end 16-31 frames late (medians) -- while longer spells are unbiased. The
metric matches spells by IoU 0.3-0.7, so for a two-second spell a dozen frames decide whether it matches at the higher
thresholds. Instead of moving boundaries to evidence optima (round 15, lost), the bias is corrected directly: v14's
final labels are post-processed so that every possession shorter than 4 s lying between two possessions of the other
team gives delta frames back on each side, and every possession shorter than 4 s that follows a stoppage and ends in
the other team gives delta frames back at its end (never below 8 frames). delta 6 / 12 fixed in advance. Ten contexts,
3 v10 seeds x 2 verifier seeds. AGGREGATE output only.

  python -I src/exp_shrink.py eval VSEED[,VSEED]
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

DELTAS = (6, 12)
SHORT, MIN_LEN = 100, 8


def shrink(pr, delta):
    out = pr.copy()
    st = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
    en = np.r_[st[1:], len(pr)]
    for j in range(1, len(st) - 1):
        a, e, k = st[j], en[j], pr[st[j]]
        if k > 1 or e - a >= SHORT or pr[e] != 1 - k:
            continue
        prev = pr[st[j - 1]]
        if prev == 1 - k:
            d = min(delta, (e - a - MIN_LEN) // 2)
            if d > 0:
                out[a:a + d] = 1 - k
                out[e - d:e] = 1 - k
        elif prev == 2:
            d = min(delta, e - a - MIN_LEN)
            if d > 0:
                out[e - d:e] = 1 - k
    return out


def evaluate(vseeds):
    import exp_blend_weight as bw
    bw.use_weight(0.5)
    cells = {}
    names = ["v14"] + ["shrink %d" % d for d in DELTAS]
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                for vseed in vseeds:
                    v14 = bw._verified(src, seed, dname, vseed, lambda s2: ev._rows4(s2, seed, dname), "probs4")[1]
                    outs = {"v14": v14}
                    for d in DELTAS:
                        outs["shrink %d" % d] = [shrink(p, d) for p in v14]
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
