"""Round 15 diagnostic: recall of true short spells by kind under v10 (3 seeds) and the round-13 blend. AGGREGATE only.

Kinds of true spells < 4 s: between two opponent spells (in-play), after a restart, before a stoppage, between two
stoppages. Recall over the five IoU thresholds and the share "seen" by the +-25 probability (>= 6 frames favouring
the spell's team), learned and true D, shipped decoder (B2 c 48). Also how the decoder treats the stoppage-adjacent
ones: the share of those frames decoded as D (dead-ball error) vs the wrong team.

  python -I src/exp_short_kinds.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402

KINDS = ("between opponents", "after restart", "before stoppage", "between stoppages")


def kind_of(gt, a, e, k):
    prev = gt[a - 1] if a > 0 else 3
    nxt = gt[e] if e < len(gt) else 3
    if prev == 1 - k and nxt == 1 - k:
        return KINDS[0]
    if prev == 2 and nxt == 2:
        return KINDS[3]
    if prev == 2:
        return KINDS[1]
    if nxt == 2:
        return KINDS[2]
    return "other"


def tally(gts, preds, phs):
    out = {kd: np.zeros(5) for kd in KINDS}      # n, hits (avg over thresholds), seen, frames called D, wrong team
    for gt, pr, ph in zip(gts, preds, phs):
        gs, ps = eo.spells(gt), eo.spells(pr)
        hit = np.zeros(len(gs))
        for thr in es.IOUS:
            hit += eo.matched(gs, ps, thr)[0]
        fav = np.where(np.asarray(ph) > 0.5, 0, 1)
        for (a, e, k), h in zip(gs, hit):
            if e - a >= es.SHORT:
                continue
            kd = kind_of(gt, a, e, k)
            if kd not in out:
                continue
            seg = pr[a:e]
            out[kd] += [1, h / len(es.IOUS), (fav[a:e] == k).sum() >= 6, np.mean(seg == 2), np.mean(seg == 1 - k)]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, convs, dec in es.CONTEXTS:
        labs = {c: bench.labels(src, c) for c in convs}
        models = {"v10": [exp_smooth.smooth(es.v10_raw(s, src), 25) for s in es.SEEDS]}
        if not src.startswith("dfl"):
            members = es.bigru_members(src)
            models["blend"] = [exp_smooth.smooth(es.blend(es.v10_raw(s, src), m), 25)
                               for s, m in zip(es.SEEDS, members)]
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        for c in convs:
            gts = labs[c]
            for dname, pds in (("learned D", learned), ("true D", [np.where(g == 2, 1 - es.EPS, es.EPS) for g in gts])):
                for m, phs_list in models.items():
                    tot = {kd: np.zeros(5) for kd in KINDS}
                    for phs in phs_list:
                        t = tally(gts, es.decode(dec, 48.0, pds, phs), phs)
                        for kd in KINDS:
                            tot[kd] += t[kd]
                    print("[%s %s | %s | %s] " % (src.split("@")[0], c, dname, m) + " | ".join(
                        "%s n %d recall %.2f seen %.2f (frames D %.2f, wrong team %.2f)" % (
                            kd, v[0] / len(phs_list), v[1] / max(v[0], 1), v[2] / max(v[0], 1), v[3] / max(v[0], 1),
                            v[4] / max(v[0], 1)) for kd, v in tot.items()), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


if __name__ == "__main__":
    main()
