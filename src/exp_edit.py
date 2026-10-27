"""Round 9 (research a17), step 1: candidate coverage and legal-path oracle for spell EDITS of v7's decoded output.

Candidates (team k inside a decoded spell of the other team): maximal runs where v7's raw team probability, raw or
smoothed over +-6 / +-12 frames, favours k for >= 5 frames. Coverage: share of true spells < 4 s with a same-team
candidate at IoU >= 0.3 / 0.5 / 0.7; candidates per alive minute and their share that is "real" (IoU >= 0.5 with
a same-team true spell). Oracles on top of v7 (learned D): insert the best real candidate per true spell; delete
decoded spells with no same-team true overlap (IoU < 0.3), merging them into the previous spell; both. These bound
what an edit rescorer could reach with this candidate family. AGGREGATE output only.

  python -I src/exp_edit.py
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
from mlfeat import win_mean  # noqa: E402

SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))
MIN_RUN = 5
SHORT = 100


def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def candidates(p, y0):
    """[(a, b, k)] team-k runs inside decoded spells of the other team."""
    out = set()
    for w in (0, 6, 12):
        q = p if w == 0 else win_mean(p, -w, w)
        for k, fav in ((0, q > 0.5), (1, q < 0.5)):
            m = fav & (y0 == 1 - k)
            for a, b in runs(m):
                if b - a >= MIN_RUN:
                    out.add((int(a), int(b), k))
    return sorted(out)


def iou(a, b, c, d):
    inter = max(0, min(b, d) - max(a, c))
    return inter / float(max(b, d) - min(a, c))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, conv, dec in SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        y0s = e2.decode_all(dec, pdl, phl)
        cov = np.zeros(3)
        n_short = n_cand = n_real = alive = 0
        ins, dele, both = [], [], []
        for p, y0, g in zip(phl, y0s, labs):
            alive += int((g < 2).sum())
            cands = candidates(p, y0)
            gt = exp_oracle.spells(g)
            n_cand += len(cands)
            best = {}
            for a, b, k in cands:
                for j, (c, d, kk) in enumerate(gt):
                    if kk == k and d > a and c < b:
                        v = iou(a, b, c, d)
                        if v >= 0.5:
                            n_real += 1
                        if v > best.get(j, (0, None))[0]:
                            best[j] = (v, (a, b, k))
            for j, (c, d, kk) in enumerate(gt):
                if d - c < SHORT:
                    n_short += 1
                    v = best.get(j, (0, None))[0]
                    cov += [v >= 0.3, v >= 0.5, v >= 0.7]
            # oracle insertion: the best real candidate per true spell (IoU >= 0.5)
            yi = y0.copy()
            for j, (v, cand) in best.items():
                if v >= 0.5:
                    a, b, k = cand
                    seg = yi[a:b]
                    seg[seg < 2] = k
            # oracle deletion: decoded spells without a same-team true overlap at IoU >= 0.3 -> previous team
            yd = y0.copy()
            for a, b, k in exp_oracle.spells(y0):
                ok = any(kk == k and iou(a, b, c, d) >= 0.3 for c, d, kk in gt if d > a and c < b)
                if not ok and a > 0 and y0[a - 1] < 2:
                    yd[a:b] = y0[a - 1]
            yb = yd.copy()
            for j, (v, cand) in best.items():
                if v >= 0.5:
                    a, b, k = cand
                    seg = yb[a:b]
                    seg[seg < 2] = k
            ins.append(yi), dele.append(yd), both.append(yb)
        ref = e2.score(src, conv, y0s)
        res = [e2.score(src, conv, x) for x in (ins, dele, both)]
        print("%-10s %-6s short true spells %d | candidates %.1f / alive min, real %.3f | coverage IoU>=0.3 %.3f, "
              ">=0.5 %.3f, >=0.7 %.3f | v7 %.4f, oracle insert %+.4f, oracle delete %+.4f, both %+.4f" % (
                  src, conv, n_short, n_cand / (alive / 1500.0), n_real / max(n_cand, 1), cov[0] / n_short,
                  cov[1] / n_short, cov[2] / n_short, ref.mean(), res[0].mean() - ref.mean(),
                  res[1].mean() - ref.mean(), res[2].mean() - ref.mean()), flush=True)


if __name__ == "__main__":
    main()
