"""Players-only dead-ball evidence (round 8, research a08 / R-3): what player behaviour alone says about the ball
being out of play, without the ball, its missingness or the broadcast flags (the earlier learned dead-ball models
used the ball and transferred badly). Permutation-invariant per frame -- ATD slots are not identities:

  speed / acceleration quantiles and still-player shares (all players and per team), visible-player counts,
  team centroid speeds, spreads and the distance between the team centroids, players near the touch lines / goal
  lines / corner flags, in the goal areas and the centre circle, players of each team in the opponent's half
  (kick-off shape), each with windowed means over +-0.4 / 0.8 / 2 / 4 s and the change against the last / next
  2-4 s.

po_feats(P, team, gk, pitch) -> (n, F) float32 and the feature names.
"""
import numpy as np

import feat_team2
from mlfeat import velocity, win_mean

WINS = ((-10, 10), (-20, 20), (-50, 50), (-100, 100))


def _q(x, qs):
    with np.errstate(all="ignore"):
        return [np.nanpercentile(x, q, axis=1) for q in qs]


def po_feats(P, team, gk, pitch):
    S = feat_team2._prep(P, np.full((len(P), 2), np.nan, np.float32), team, gk, pitch)
    L, W = S["L"], S["W"]
    pxy = S["pxy"]
    x, y = S["x"], S["y"]
    n = len(pxy)
    seen = np.isfinite(x)
    v = velocity(pxy)
    sp = np.linalg.norm(v, axis=2)
    sp[sp > 12] = np.nan
    acc = np.full_like(sp, np.nan)
    acc[2:-2] = np.abs(sp[4:] - sp[:-4]) * 25.0 / 4.0
    F = {}
    with np.errstate(all="ignore"):
        for name, m in (("all", np.ones(len(team), bool)), ("h", team == 0), ("a", team == 1)):
            s = np.where(m[None, :] & seen, sp, np.nan)
            for q, val in zip((10, 50, 90), _q(s, (10, 50, 90))):
                F["sp_p%d_%s" % (q, name)] = val
            F["sp_mean_%s" % name] = np.nanmean(s, axis=1)
            F["still05_%s" % name] = np.nanmean(np.where(np.isfinite(s), s < 0.5, np.nan), axis=1)
            F["still1_%s" % name] = np.nanmean(np.where(np.isfinite(s), s < 1.0, np.nan), axis=1)
            F["nvis_%s" % name] = (m[None, :] & seen).sum(1).astype(np.float32)
        a50, a90 = _q(acc, (50, 90))
        F["acc_p50"], F["acc_p90"] = a50, a90
        cen = {}
        for k in (0, 1):
            own = S["th"] if k == 0 else S["ta"]
            cx = np.nanmean(np.where(own[None, :], x, np.nan), axis=1)
            cy = np.nanmean(np.where(own[None, :], y, np.nan), axis=1)
            cen[k] = (cx, cy)
            csp = np.full(n, np.nan)
            csp[12:-12] = np.hypot(cx[24:] - cx[:-24], cy[24:] - cy[:-24]) / (24 / 25.0)
            F["cen_sp_%d" % k] = csp
            F["spread_x_%d" % k] = np.nanstd(np.where(own[None, :], x, np.nan), axis=1)
            F["spread_y_%d" % k] = np.nanstd(np.where(own[None, :], y, np.nan), axis=1)
            # players of team k in the opponent's half (kick-off: none)
            att = (x - L / 2) * S["sgn"][k]
            F["in_opp_half_%d" % k] = np.sum(own[None, :] & seen & (att > 1.0), axis=1).astype(np.float32)
        F["cen_dist"] = np.hypot(cen[0][0] - cen[1][0], cen[0][1] - cen[1][1])
        on = seen
        F["near_touch"] = np.sum(on & ((y < 2.0) | (y > W - 2.0)), axis=1).astype(np.float32)
        F["near_goalline"] = np.sum(on & ((x < 2.0) | (x > L - 2.0)), axis=1).astype(np.float32)
        corner = np.min(np.stack([np.hypot(x - cx_, y - cy_) for cx_ in (0.0, L) for cy_ in (0.0, W)]), axis=0)
        F["near_corner"] = np.sum(on & (corner < 4.0), axis=1).astype(np.float32)
        ga = (np.abs(y - W / 2) < 9.16) & ((x < 5.5) | (x > L - 5.5))
        F["in_goal_area"] = np.sum(on & ga, axis=1).astype(np.float32)
        F["in_centre_circle"] = np.sum(on & (np.hypot(x - L / 2, y - W / 2) < 9.15), axis=1).astype(np.float32)
    for key in ("sp_p50_all", "sp_p90_all", "still1_all", "acc_p90", "cen_sp_0", "cen_sp_1", "near_touch",
                "in_centre_circle"):
        base = F[key].astype(np.float64)
        for lo, hi in WINS:
            F["%s_w%d" % (key, hi)] = win_mean(base, lo, hi)
        F["%s_dpast" % key] = base - win_mean(base, -100, -50)
        F["%s_dnext" % key] = win_mean(base, 50, 100) - base
    names = sorted(F)
    return np.stack([np.asarray(F[k], np.float32) for k in names], 1), names
