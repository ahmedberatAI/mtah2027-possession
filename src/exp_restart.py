"""Round 18 (literature review, agents A6 / A5: snap a possession change to a physical control event): v14's decoded
team switches moved to the opposing team's first physical control onset nearby.

The ceiling is real: moving the end of each spell that starts right after a stoppage to the true switch within +-2 s
adds +0.004..+0.014 spell F1 to v14 (+-4 s: +0.008..+0.024), because these short first spells are found but end
16-31 frames late (round 15). Round 15's re-localisation on less-smoothed probabilities lost everywhere; the
literature (ELASTIC, Vidal-Codina 2022, Link & Hoernig 2017) times a change by a physical event instead. Rules, fixed
before the first run: for a decoded switch k -> o at frame e, search [e - 75, e + 10] (keeping both spells >= 8 frames)
for the first frame from which an o player holds the visible ball (nearest player, within 1.5 m) for 6 frames
("hold"), or for o's first detected touch ("touch"); move the switch there. Applied to the first spell after a
stoppage ("restart") or to every team switch ("all"). AGGREGATE output only.

  python -I src/exp_restart.py eval VSEED[,VSEED]
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_verify as ev  # noqa: E402

BACK, FWD, HOLD, MIN_LEN = 75, 10, 6, 8
VARIANTS = {"restart hold": (True, "hold"), "restart touch": (True, "touch"), "all hold": (False, "hold")}


def onsets(c, o, rule):
    """Candidate frames, sorted, where team o takes physical control."""
    if rule == "touch":
        return np.sort(c["touch_t"][c["touch_k"] == o])
    hold = (~c["miss"]) & (c["holder"] >= 0) & (c["near_team"] == o)
    run = np.convolve(hold.astype(np.int32), np.ones(HOLD, np.int32), "valid") == HOLD    # hold[f:f + HOLD] all
    return np.flatnonzero(run & ~np.r_[False, run[:-1]])


def snap(pr, c, only_restart, rule):
    out = pr.copy()
    st = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
    en = np.r_[st[1:], len(pr)]
    cand = {o: onsets(c, o, rule) for o in (0, 1)}
    for j in range(1, len(st) - 1):
        a, e, k = st[j], en[j], pr[st[j]]
        if k > 1 or pr[e] != 1 - k or (only_restart and pr[st[j - 1]] != 2):
            continue
        o = 1 - k
        lo, hi = max(a + MIN_LEN, e - BACK), min(en[j + 1] - MIN_LEN, e + FWD)
        f = cand[o][(cand[o] >= lo) & (cand[o] < hi)]
        if not len(f):
            continue
        new = int(f[0])
        if new < e:
            out[new:e] = o
        elif new > e:
            out[e:new] = k
    return out


def evaluate(vseeds):
    import exp_blend_weight as bw
    bw.use_weight(0.5)
    cells = {}
    names = ["v14"] + list(VARIANTS)
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        outs = {}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                for vseed in vseeds:
                    outs[(seed, dname, vseed)] = {"v14": bw._verified(src, seed, dname, vseed,
                                                                      lambda s2: ev._rows4(s2, seed, dname), "probs4")[1]}
        for name in VARIANTS:
            for o in outs.values():
                o[name] = [None] * len(blocks)
        for i, b in enumerate(blocks):
            c = ea.frame_cues(b)
            for o in outs.values():
                for name, (only_restart, rule) in VARIANTS.items():
                    o[name][i] = snap(o["v14"][i], c, only_restart, rule)
            del c
        for (seed, dname, vseed), o in outs.items():
            for cv in convs:
                for name in names:
                    cells.setdefault((src, cv, dname, name), []).append(er.by_match(blocks, labs[cv], o[name]))
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
