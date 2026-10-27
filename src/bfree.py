"""Ball-free short-horizon team interaction features (round 9, research a20): signals of which team has the ball
that need neither the ball nor player identities (ATD slots are not people). Per team k (vs opponent o), per frame:

  expand_w     change of k's spread (sqrt of the x-std * y-std area proxy) over the next w minus the last w frames
               (a team that wins the ball spreads out, the defending team compacts), w = 10 / 20 / 40
  line_v_w     speed of k's last outfield line (mean depth of its 4 deepest outfield players, towards the opponent
               goal) over +-w frames: pushing up vs dropping back
  att_rev      reversal of k's centroid velocity towards the opponent goal: mean over the next 1 s minus the last 1 s
  close        mean velocity component of k's outfield players towards their nearest opponent (pressing)
  support      share of k's outfield players moving away from their nearest opponent faster than 1 m/s (offering)
each with its opponent counterpart and the own - opponent difference, windowed (+-0.5 s, last / next 1 s).

bf_feats(P, team, gk, pitch) -> (home view, away view), each (n, F) float32.
"""
import numpy as np

import feat_team2
from mlfeat import velocity, win_mean


def _side(S, k, v):
    L = S["L"]
    x, y = S["x"], S["y"]
    n = len(x)
    own = (S["th"] if k == 0 else S["ta"]) & ~S["gk"]
    opp = (S["ta"] if k == 0 else S["th"]) & ~S["gk"]
    sgn = S["sgn"][k]
    with np.errstate(all="ignore"):
        sx = np.nanstd(np.where(own[None, :], x, np.nan), axis=1)
        sy = np.nanstd(np.where(own[None, :], y, np.nan), axis=1)
        spread = np.sqrt(sx * sy)
        depth = np.where(own[None, :], (x - L / 2) * sgn, np.nan)            # towards the opponent goal
        last_line = np.nanmean(np.sort(np.where(np.isfinite(depth), depth, np.inf), axis=1)[:, :4], axis=1)
        last_line[~np.isfinite(last_line)] = np.nan
        cx = np.nanmean(np.where(own[None, :], x, np.nan), axis=1) * sgn
        catt = np.full(n, np.nan)
        catt[12:-12] = (cx[24:] - cx[:-24]) / (24 / 25.0)
        # nearest opponent per own player: direction and closing speed
        oi, pi = np.flatnonzero(own), np.flatnonzero(opp)
        pxy = S["pxy"]
        d = pxy[:, oi, None, :] - pxy[:, None, pi, :]                        # own -> opp vectors (n, no, np, 2)
        dist = np.linalg.norm(d, axis=-1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        j = np.argmin(dist, axis=2)
        vec = np.take_along_axis(d, j[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0, :]
        nd = np.take_along_axis(dist, j[:, :, None], axis=2)[:, :, 0]
        unit = -vec / np.maximum(nd, 1e-6)[:, :, None]                        # own -> nearest opponent
        vo = v[:, oi, :]
        comp = np.sum(vo * unit, axis=2)
        comp[~np.isfinite(nd) | (nd > 15)] = np.nan
        close = np.nanmean(comp, axis=1)
        support = np.nanmean(np.where(np.isfinite(comp), comp < -1.0, np.nan), axis=1)
    out = {}
    for w in (10, 20, 40):
        out["expand_%d" % w] = win_mean(spread, 1, w) - win_mean(spread, -w, -1)
        lv = np.full(n, np.nan)
        lv[w:-w] = (last_line[2 * w:] - last_line[:-2 * w]) / (2 * w / 25.0)
        out["line_v_%d" % w] = lv
    out["att_rev"] = win_mean(catt, 1, 25) - win_mean(catt, -25, -1)
    out["close"] = close
    out["support"] = support
    return out


def bf_feats2(P, team, gk, pitch):
    """Two more identity-free team signals (round 9): per team the coherence of its outfield players' running
    directions (length of the mean unit velocity of players faster than 2 m/s) and its forward share, and the
    composition of the densest 10 m neighbourhood on the pitch (the busiest player, by players within 10 m: own minus
    opponent players there, and whether the busiest player is own). Windowed (+-0.5 s, last / next 1 s).
    Returns (home view, away view), each (n, F)."""
    S = feat_team2._prep(P, np.full((len(P), 2), np.nan, np.float32), team, gk, pitch)
    v = velocity(S["pxy"])
    v[np.linalg.norm(v, axis=2) > 12] = np.nan
    pxy = S["pxy"]
    n = len(pxy)
    with np.errstate(all="ignore"):
        sp = np.linalg.norm(v, axis=2)
        unit = v / np.maximum(sp, 1e-6)[:, :, None]
        d = np.linalg.norm(pxy[:, :, None, :] - pxy[:, None, :, :], axis=-1)
        near = np.nansum(d < 10.0, axis=2).astype(np.float32)            # players within 10 m (incl. self)
        near[~np.isfinite(pxy[:, :, 0])] = -1
        busiest = np.argmax(near, axis=1)
        inb = d[np.arange(n), busiest] < 10.0                            # players in the busiest neighbourhood
    views = []
    for k in (0, 1):
        own = (S["th"] if k == 0 else S["ta"]) & ~S["gk"]
        opp = (S["ta"] if k == 0 else S["th"]) & ~S["gk"]
        fast = own[None, :] & (sp > 2.0)
        with np.errstate(all="ignore"):
            mu = np.nanmean(np.where(fast[:, :, None], unit, np.nan), axis=1)
            coh = np.linalg.norm(mu, axis=1)
            fwd = np.nanmean(np.where(fast, unit[:, :, 0] * S["sgn"][k] > 0.5, np.nan), axis=1)
        comp = (np.sum(inb & own[None, :], axis=1) - np.sum(inb & opp[None, :], axis=1)).astype(np.float64)
        bown = own[busiest].astype(np.float64)
        cols = []
        for x in (coh, fwd, comp, bown):
            cols.append(x)
            for lo, hi in ((-12, 12), (-25, 0), (0, 25)):
                cols.append(win_mean(x, lo, hi))
        views.append(np.stack(cols, 1).astype(np.float32))
    return views[0], views[1]


def bf_feats(P, team, gk, pitch):
    S = feat_team2._prep(P, np.full((len(P), 2), np.nan, np.float32), team, gk, pitch)
    v = velocity(S["pxy"])
    v[np.linalg.norm(v, axis=2) > 12] = np.nan
    sides = {k: _side(S, k, v) for k in (0, 1)}
    views = []
    for k in (0, 1):
        own, opp = sides[k], sides[1 - k]
        cols = []
        for name in sorted(own):
            a, b = own[name], opp[name]
            cols += [a, b, a - b]
            if name in ("close", "support", "att_rev"):
                for lo, hi in ((-12, 12), (-25, 0), (0, 25)):
                    cols.append(win_mean(a - b, lo, hi))
        views.append(np.stack(cols, 1).astype(np.float32))
    return views[0], views[1]
