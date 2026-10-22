"""Timing of predicted transitions vs the public GT, by transition type (AGGREGATE: counts and quantiles).

For every GT transition, the nearest predicted transition into the same label within +-3 s; offset =
predicted - GT frame (negative = early). Types: alive -> D, D -> alive, in-play team change.

  python -I src/trans_offsets.py [bench barath_probs tag ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402

pm = ev.pm
WIN = 75


def kind(a, b):
    if b == 2:
        return "alive->D"
    if a == 2:
        return "D->alive"
    return "team change"


def offsets(gt, pred):
    runs = pm.valid_runs(gt)
    gtt = pm.transitions(gt, runs)
    prt = pm.transitions(pred, runs)
    out = {}
    for t, a, b in gtt:
        k = kind(a, b)
        cand = [q - t for q, a2, b2 in prt if b2 == b and abs(q - t) <= WIN and (kind(a2, b2) == k)]
        out.setdefault(k, []).append(min(cand, key=abs) if cand else None)
    return out


def report(name, lab):
    gt = ev.public_gt()
    res = offsets(gt, ev.to_codes(lab))
    cells = []
    for k in ("alive->D", "D->alive", "team change"):
        v = res.get(k, [])
        hit = np.array([x for x in v if x is not None], float) / ev.atd.FPS
        if len(hit):
            cells.append("%s n %d found %d: median %+.2f s (p25 %+.2f, p75 %+.2f)" % (
                k, len(v), len(hit), np.median(hit), np.percentile(hit, 25), np.percentile(hit, 75)))
        else:
            cells.append("%s n %d found 0" % (k, len(v)))
    print("%-34s | %s" % (name, " | ".join(cells)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import model_v5
    from make_submission import merged_features
    f = merged_features()
    report("v5 (current)", model_v5.predict(f))
    for tag in sys.argv[1:]:
        path = os.path.join(ev.ROOT, "cache", "bench", "barath_probs_%s.pkl" % tag)
        for pt in (8, 20):
            model_v5._probs.clear()
            model_v5.team_logp(f, path)
            lab = model_v5.predict(f, {"pen_team": float(pt)})
            model_v5._probs.clear()
            report("%s pen %d" % (tag[:26], pt), lab)


if __name__ == "__main__":
    main()
