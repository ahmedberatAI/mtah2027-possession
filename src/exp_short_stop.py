"""Round 15 diagnostic: restart logic around stoppages ("who restarts tells who touched last"). AGGREGATE only.

Football rule: when the ball leaves the pitch over a touchline or goal line, the restart (throw-in, corner, goal kick)
goes to the team that did NOT touch it last; after a foul the fouled team (usually the one with the ball) restarts.
For every true stoppage with alive play on both sides (v10 seed 0, true D so that only the team question remains):
  where the ball left play (degraded input, last visible ball within 1 s before the stoppage): over a touchline
  (within 1.5 m of it or beyond), over a goal line, inside the pitch (foul / offside / other), unknown;
  truth: is the last spell before the stoppage the restarting team's, and is that last spell short (< 4 s);
  v10: same two questions from its decoded labels.
The rule candidate: v10's last team equals the restarting team at an out-of-play stoppage (or differs at an inside
one) -> a short spell of the other team before the stoppage was missed. Its precision and recall against the truth.

  python -I src/exp_short_stop.py
"""
import os
import sys
import warnings
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402

CONTEXTS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))


def where_out(b, d0):
    L, W = b["pitch"]
    bxy = np.asarray(b["B"], np.float32)[max(d0 - 25, 0):d0 + 1, :2] * np.array([L, W], np.float32)
    ok = np.isfinite(bxy[:, 0])
    if not ok.any():
        return "unknown"
    x, y = bxy[ok][-1]
    if min(y, W - y) <= 1.5:
        return "touchline"
    if min(x, L - x) <= 1.5:
        return "goal line"
    return "inside"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, conv, dec in CONTEXTS:
        blocks, gts = bench.blocks(src), bench.labels(src, conv)
        smo = exp_smooth.smooth(es.v10_raw(0, src), 25)
        preds = es.decode(dec, 48.0, [np.where(g == 2, 1 - es.EPS, es.EPS) for g in gts], smo)
        tab = Counter()
        for b, gt, pr in zip(blocks, gts, preds):
            dead = (gt == 2).astype(np.int8)
            d = np.diff(np.concatenate([[0], dead, [0]]))
            for d0, d1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if d0 == 0 or d1 >= len(gt) or gt[d0 - 1] > 1 or gt[d1] > 1:
                    continue
                loc = where_out(b, d0)
                restart = gt[d1]
                sp = [s for s in eo.spells(gt[max(d0 - 500, 0):d0])]
                last_len = sp[-1][1] - sp[-1][0] if sp else 0
                t_last, p_last, p_restart = gt[d0 - 1], pr[d0 - 1], pr[d1]
                short_other = last_len < es.SHORT and len(sp) >= 2 and sp[-2][2] == 1 - t_last
                out_of_play = loc in ("touchline", "goal line")
                flag = (p_last == p_restart) if out_of_play else (p_last != p_restart) if loc == "inside" else False
                missed = short_other and p_last != t_last
                tab[(loc, "n")] += 1
                tab[(loc, "true last = restart")] += int(t_last == restart)
                tab[(loc, "true last short, other before")] += int(short_other)
                tab[(loc, "v10 last = restart")] += int(p_last == p_restart)
                tab[(loc, "v10 last wrong")] += int(p_last != t_last)
                tab[(loc, "missed short")] += int(missed)
                tab[(loc, "flag")] += int(flag)
                tab[(loc, "flag & missed short")] += int(flag and missed)
        print("\n=== %s %s (v10 seed 0, true D) ===" % (src.split("@")[0], conv))
        for loc in ("touchline", "goal line", "inside", "unknown"):
            n = tab[(loc, "n")]
            if not n:
                continue
            f, fm, ms = tab[(loc, "flag")], tab[(loc, "flag & missed short")], tab[(loc, "missed short")]
            print("  %-9s stoppages %4d | truth: last = restart %.2f, last spell short after the other team %.2f | "
                  "v10: last = restart %.2f, last team wrong %.2f, missed short %d | rule flags %d, precision %.2f, "
                  "recall %.2f" % (loc, n, tab[(loc, "true last = restart")] / n,
                                   tab[(loc, "true last short, other before")] / n, tab[(loc, "v10 last = restart")] / n,
                                   tab[(loc, "v10 last wrong")] / n, ms, f, fm / max(f, 1), fm / max(ms, 1)))
        sys.stdout.flush()
        bench._blocks.clear()
        bench._labels.clear()


if __name__ == "__main__":
    main()
