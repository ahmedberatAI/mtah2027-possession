"""Round 19 (analyst idea 1): the interception race -- who reaches the moving ball's path first.

A possession change is a race for the ball: once the ball is played, its path and the players' positions and
velocities largely decide who controls it next (time-to-intercept / pitch control, Spearman 2017-2018). For every frame
with the ball seen and moving (>= 4 m/s over the next 4 frames), the path is extended along that velocity for 2 s
(stopping at the touchlines); a player reaches a path point after a 0.7 s reaction plus the straight distance from his
reaction-time position at 5 m/s; the first point some player reaches no later than the ball names the interceptor and
the time of the interception. Players within 1.5 m of the ball are ignored for the first 0.3 s (the passer). Identity-
free: positions, velocities and team labels only. AGGREGATE output only.

  python -I src/exp_race.py diag     recall of true team changes and false alarms, race vs the nearest player now
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_verify as ev  # noqa: E402

FPS = 25.0
REACT, VMAX, HORIZON, STEP = 0.7, 5.0, 2.0, 0.04
MIN_SPEED, OWN, OWN_T = 4.0, 1.5, 0.3


def kinematics(b):
    L, W = b["pitch"]
    pxy = np.asarray(b["P"], np.float32)[:, :, :2] * np.array([L, W], np.float32)
    bxy = np.asarray(b["B"], np.float32)[:, :2] * np.array([L, W], np.float32)
    pv = np.full_like(pxy, np.nan)
    pv[2:-2] = (pxy[4:] - pxy[:-4]) / (4 / FPS)
    bv = np.full_like(bxy, np.nan)
    bv[:-4] = (bxy[4:] - bxy[:-4]) / (4 / FPS)
    return L, W, pxy, pv, bxy, bv


def race(b, chunk=2048):
    """Per frame: interceptor team (0 / 1; -1 = ball not moving or seen, -2 = leaves the pitch untouched) and the
    time to the interception (s)."""
    L, W, pxy, pv, bxy, bv = kinematics(b)
    team = np.asarray(b["team"])
    n = len(bxy)
    who = np.full(n, -1, np.int8)
    tau = np.full(n, np.nan, np.float32)
    speed = np.linalg.norm(bv, axis=1)
    idx = np.flatnonzero(np.isfinite(speed) & (speed >= MIN_SPEED) & np.isfinite(bxy[:, 0]))
    taus = np.arange(STEP, HORIZON + 1e-9, STEP, dtype=np.float32)                       # (T,)
    ok_team = np.isin(team, (0, 1))
    for a in range(0, len(idx), chunk):
        f = idx[a:a + chunk]
        path = bxy[f, None, :] + bv[f, None, :] * taus[None, :, None]                    # (F, T, 2)
        inside = (path[..., 0] >= 0) & (path[..., 0] <= L) & (path[..., 1] >= 0) & (path[..., 1] <= W)
        start = pxy[f] + np.nan_to_num(pv[f]) * REACT                                    # (F, N, 2)
        d = np.linalg.norm(path[:, :, None, :] - start[:, None, :, :], axis=3)            # (F, T, N)
        arrive = REACT + d / VMAX
        own = np.linalg.norm(pxy[f] - bxy[f, None, :], axis=2) <= OWN                     # (F, N)
        bad = (~ok_team)[None, None, :] | ~np.isfinite(arrive) | (own[:, None, :] & (taus[None, :, None] < OWN_T))
        arrive = np.where(bad, np.inf, arrive)
        reach = (arrive <= taus[None, :, None]) & inside[:, :, None]                     # (F, T, N)
        any_t = reach.any(2)
        first = np.where(any_t.any(1), any_t.argmax(1), -1)
        for k, (fr, t0) in enumerate(zip(f, first)):
            if t0 < 0:
                who[fr] = -2
                continue
            players = np.flatnonzero(reach[k, t0])
            j = players[np.argmin(arrive[k, t0, players])]
            who[fr] = team[j]
            tau[fr] = taus[t0]
    return who, tau


def nearest_team(b):
    L, W, pxy, pv, bxy, bv = kinematics(b)
    team = np.asarray(b["team"])
    d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d = np.where(np.isfinite(d) & np.isin(team, (0, 1))[None, :], d, np.inf)
    j = np.argmin(d, 1)
    return np.where(np.isfinite(d[np.arange(len(d)), j]), team[j], -1).astype(np.int8)


def diag():
    """At every true team -> team change (labels of the source's primary convention): does a signal name the new team
    in the second before it? In stable stretches (>= 2 s from any change, live): how often does it name the other
    team? Race (moving-ball frames, interception within 1.5 s) against the nearest player now (seen-ball frames)."""
    for src, convs, dec, prim in ev.SOURCES:
        labs = bench.labels(src, prim)
        stats = {"race": [0, 0, 0, 0], "near": [0, 0, 0, 0]}        # hits, changes, alarms, stable frames
        short_hit = {"race": 0, "near": 0}
        n_short = 0
        lead = []
        for b, g in zip(bench.blocks(src), labs):
            who, tau = race(b)
            near = nearest_team(b)
            sw = np.flatnonzero((g[1:] != g[:-1]) & (g[1:] < 2) & (g[:-1] < 2)) + 1
            st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            en = np.r_[st[1:], len(g)]
            for c in sw:
                new = g[c]
                lo, hi = max(0, c - 25), min(len(g), c + 5)
                r = (who[lo:hi] == new) & (tau[lo:hi] <= 1.5)
                stats["race"][0] += bool(r.any())
                stats["near"][0] += bool((near[lo:hi] == new).any())
                stats["race"][1] += 1
                stats["near"][1] += 1
                if r.any():
                    k = np.flatnonzero(r)[0]
                    lead.append((lo + k + tau[lo + k] * FPS) - c)
                j = np.searchsorted(st, c, side="right") - 1
                if en[j] - st[j] < 100:
                    n_short += 1
                    short_hit["race"] += bool(r.any())
                    short_hit["near"] += bool((near[lo:hi] == new).any())
            dist = np.full(len(g), 10 ** 6)
            for c in sw:
                dist = np.minimum(dist, np.abs(np.arange(len(g)) - c))
            stable = (dist >= 50) & (g < 2)
            mv = stable & (who >= 0) & (tau <= 1.5)
            stats["race"][2] += int((mv & (who != g)).sum())
            stats["race"][3] += int(mv.sum())
            sn = stable & (near >= 0)
            stats["near"][2] += int((sn & (near != g)).sum())
            stats["near"][3] += int(sn.sum())
        print("%-8s changes %5d | new team named in the second before: race %.2f, nearest %.2f | short spells (%d): "
              "race %.2f, nearest %.2f | stable-frame alarms: race %.3f (of %d moving), nearest %.3f | race lead "
              "(predicted time - change) median %+.0f frames, IQR %+.0f..%+.0f" % (
                  src.split("@")[0], stats["race"][1], stats["race"][0] / stats["race"][1],
                  stats["near"][0] / stats["near"][1], n_short, short_hit["race"] / max(n_short, 1),
                  short_hit["near"] / max(n_short, 1), stats["race"][2] / max(stats["race"][3], 1), stats["race"][3],
                  stats["near"][2] / max(stats["near"][3], 1), np.median(lead), *np.percentile(lead, [25, 75])),
              flush=True)
        bench._blocks.clear()
        bench._labels.clear()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "diag":
        return diag()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
