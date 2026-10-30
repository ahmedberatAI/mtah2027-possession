"""Round 15 diagnostic: are short spells found but mis-timed? AGGREGATE output only.

v10 (3 seeds, +-25, shipped B2 decoder), learned and true D. For each true spell, the predicted spell of the same team
with the largest overlap gives start and end offsets (predicted minus true, frames; + = late). Reported by kind and
length (< 4 s by kind, 4-8 s, > 8 s): median and quartiles of both offsets, the share whose best IoU is below 0.5
although the overlap covers >= 80% of the true spell (found, mis-timed), and the same offsets for v10's in-play team
changes against the nearest true change to the same team within +-2 s.

  python -I src/exp_short_timing.py
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
import exp_short_kinds as ek  # noqa: E402
import exp_smooth  # noqa: E402


def offsets(gts, preds):
    rows = []
    for gt, pr in zip(gts, preds):
        ps = eo.spells(pr)
        starts = np.array([p[0] for p in ps])
        for a, e, k in eo.spells(gt):
            lo = max(np.searchsorted(starts, a) - 3, 0)
            best = None
            for c, d, kk in ps[lo:lo + 8]:
                ov = min(d, e) - max(c, a)
                if kk == k and ov > 0 and (best is None or ov > best[0]):
                    best = (ov, c, d)
            kind = ek.kind_of(gt, a, e, k) if e - a < es.SHORT else ("4-8 s" if e - a < 200 else "> 8 s")
            if best is None:
                rows.append((kind, np.nan, np.nan, 0.0, 0.0))
                continue
            ov, c, d = best
            rows.append((kind, c - a, d - e, ov / (max(d, e) - min(c, a)), ov / (e - a)))
    return rows


def change_lags(gts, preds, tol=50):
    lags = []
    for gt, pr in zip(gts, preds):
        tc = eo.changes(gt)
        for t in eo.changes(pr):
            c = tc[(np.abs(tc - t) <= tol) & (gt[tc] == pr[t])]
            if len(c):
                lags.append(int(t - c[np.argmin(np.abs(c - t))]))
    return np.array(lags)


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "n/a"
    return "%+.0f [%+.0f, %+.0f]" % tuple(np.percentile(v, [50, 25, 75]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    kinds = ek.KINDS + ("4-8 s", "> 8 s")
    for src, convs, dec in es.CONTEXTS:
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        phs = [exp_smooth.smooth(es.v10_raw(s, src), 25) for s in es.SEEDS]
        for c in convs[:1]:
            gts = bench.labels(src, c)
            for dname, pds in (("learned D", learned), ("true D", [np.where(g == 2, 1 - es.EPS, es.EPS) for g in gts])):
                rows, lags = [], []
                for ph in phs:
                    pr = es.decode(dec, 48.0, pds, ph)
                    rows += offsets(gts, pr)
                    lags.append(change_lags(gts, pr))
                lags = np.concatenate(lags)
                print("\n[%s %s | %s] in-play changes matched within 2 s: lag %s frames (n %d per seed)" % (
                    src.split("@")[0], c, dname, q(lags), len(lags) // len(phs)))
                for kd in kinds:
                    sel = [r for r in rows if r[0] == kd]
                    if not sel:
                        continue
                    st = [r[1] for r in sel]
                    en = [r[2] for r in sel]
                    mist = np.mean([(r[3] < 0.5) and (r[4] >= 0.8) for r in sel])
                    print("  %-18s n %5d  start offset %-18s end offset %-18s found but IoU < 0.5: %.0f%%" % (
                        kd, len(sel) // len(phs), q(st), q(en), 100 * mist))
                sys.stdout.flush()
        bench._blocks.clear()
        bench._labels.clear()


if __name__ == "__main__":
    main()
