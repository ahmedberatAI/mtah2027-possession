"""Why are ground-truth possession spells missed? AGGREGATE counts only.

For every GT spell (maximal H/A run): best IoU with a same-team predicted spell, and the failure mode when
that IoU is below 0.5 --
  wrong_team  most of the spell is predicted as the other team
  dead        most of it is predicted D
  split       the right team covers most of it but in >= 2 predicted spells (a spurious D or flip inside)
  merged      the right team covers it but the predicted spell runs on over a GT boundary (IoU < 0.5)
  shifted     otherwise (boundaries off)
by duration bucket and by how the spell starts (after D = restart, after the other team = turnover).
Predicted spells without a GT match are counted by duration too.

  python -I src/err_spells.py            # public window: v5 and the bench V3 candidates
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402

pm = ev.pm
BUCKETS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9))


def spells(codes, runs):
    return pm.possession_spells(codes, runs)


def analyse(gt, pred, runs, iou_thr=0.5):
    gs, ps = spells(gt, runs), spells(pred, runs)
    out = []
    for a, b, k in gs:
        best = 0.0
        for c, d, k2 in ps:
            if k2 != k or d <= a or c >= b:
                continue
            inter = min(b, d) - max(a, c)
            best = max(best, inter / (max(b, d) - min(a, c)))
        seg = pred[a:b]
        share = {lab: np.mean(seg == lab) for lab in (0, 1, 2)}
        prev = gt[a - 1] if a > 0 else 3
        start = "restart" if prev == 2 else ("turnover" if prev in (0, 1) else "edge")
        if best >= iou_thr:
            mode = "ok"
        elif share[1 - k] >= 0.5:
            mode = "wrong_team"
        elif share[2] >= 0.5:
            mode = "dead"
        else:
            inside = [(c, d) for c, d, k2 in ps if k2 == k and d > a and c < b]
            if len(inside) >= 2:
                mode = "split"
            elif inside and (inside[0][0] < a - 0.25 * (b - a) or inside[0][1] > b + 0.25 * (b - a)):
                mode = "merged"
            else:
                mode = "shifted"
        out.append(((b - a) / ev.atd.FPS, start, mode, best))
    spur = []
    for c, d, k in ps:
        hit = False
        for a, b, k2 in gs:
            if k2 == k and b > c and a < d and (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) >= iou_thr:
                hit = True
                break
        if not hit:
            spur.append((d - c) / ev.atd.FPS)
    return out, spur


def report(name, labels):
    gt = ev.public_gt()
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(ev.to_codes(labels), runs)
    res, spur = analyse(gt, pred, runs)
    modes = ("ok", "wrong_team", "dead", "split", "merged", "shifted")
    print("\n%s: GT spells %d, predicted %d, unmatched predicted %d (IoU 0.5)" % (
        name, len(res), len(spells(pred, runs)), len(spur)))
    print("  %-12s %s" % ("", " ".join("%10s" % m for m in modes)))
    for lo, hi in BUCKETS:
        row = [r for r in res if lo <= r[0] < hi]
        print("  %-12s %s" % ("%g-%gs" % (lo, hi if hi < 1e9 else 99), " ".join(
            "%10d" % sum(1 for r in row if r[2] == m) for m in modes)))
    for st in ("restart", "turnover", "edge"):
        row = [r for r in res if r[1] == st]
        if row:
            print("  %-12s %s" % (st, " ".join("%10d" % sum(1 for r in row if r[2] == m) for m in modes)))
    print("  unmatched predicted spells by duration: " + ", ".join(
        "%g-%gs %d" % (lo, hi if hi < 1e9 else 99, sum(1 for s in spur if lo <= s < hi)) for lo, hi in BUCKETS))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import model_v5
    from make_submission import merged_features
    f = merged_features()
    report("v5 (current, DFL v1 team model, pen 8)", model_v5.predict(f))
    for tag in ("base_native_v3", "base_trk_v3"):
        path = os.path.join(ev.ROOT, "cache", "bench", "barath_probs_%s.pkl" % tag)
        if not os.path.isfile(path):
            continue
        for pt in (8, 20):
            model_v5._probs.clear()
            model_v5.team_logp(f, path)
            lab = model_v5.predict(f, {"pen_team": float(pt)})
            model_v5._probs.clear()
            report("v5 + %s team model, pen %d" % (tag, pt), lab)


if __name__ == "__main__":
    main()
