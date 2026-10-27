"""Ball-touch events from the tracking (round 9; identity-free). The official definition names "a team completes more
than two consecutive passes" as a proxy of controlled possession, and v10's main remaining loss is short in-play
turnovers. A touch = a local maximum (within +-GAP frames) of the ball's velocity change |v(t+K) - v(t-K)| above DV
m/s with a player within NEAR m of the ball; it goes to that player's team (K = 8 frames: the least sensitive
to BAR-ATH's filtered ball -- detected touches per alive minute Metrica@V3 27, SkillCorner 33, BAR-ATH 24, DFL@V3
41 with its rough ball). Per team view, per frame:
  last_own / next_own        whether the last / next touch is the team's (NaN if none within CLIP frames)
  since_own, since_opp       frames since the team's / the opponent's last touch (capped); until_own, until_opp
  bal_past, bal_next         own minus opponent touches in the last / next 3 s
  chain_own, chain_opp       consecutive touches of the team ending at the last touch (0 if that touch is the other's)
Returns (home view, away view), each (n, 10) float32.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d

from mlfeat import CLIP, since_last, until_next, velocity, win_mean

DV = 2.0
K = 8
NEAR = 2.0
GAP = 6


def touches(pxy, bxy, team):
    """Frames and team (0 / 1) of the detected touches, in time order."""
    n = len(bxy)
    bv = velocity(bxy)
    dv = np.full(n, np.nan, np.float32)
    if n > 2 * K:
        dv[K:n - K] = np.linalg.norm(bv[2 * K:] - bv[:n - 2 * K], axis=1)
    dvf = np.where(np.isfinite(dv), dv, 0.0)
    peak = (dvf > DV) & (dvf >= maximum_filter1d(dvf, size=2 * GAP + 1, mode="nearest"))
    cand = np.flatnonzero(peak)
    if not len(cand):
        return cand, cand
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(pxy[cand] - bxy[cand, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, axis=1)
    keep = (d[np.arange(len(cand)), j] < NEAR) & np.isin(team[j], (0, 1))
    return cand[keep], np.asarray(team)[j[keep]].astype(np.int8)


def _chains(t_team):
    out = np.ones(len(t_team), np.float32)
    for i in range(1, len(t_team)):
        if t_team[i] == t_team[i - 1]:
            out[i] = out[i - 1] + 1
    return out


def touch_feats(P, B, team, gk, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    n = len(P)
    t_idx, t_team = touches(pxy, bxy, np.asarray(team))
    chain = _chains(t_team)
    last = np.searchsorted(t_idx, np.arange(n), side="right") - 1      # index of the last touch at or before t
    views = []
    for k in (0, 1):
        own = np.zeros(n, bool)
        opp = np.zeros(n, bool)
        own[t_idx[t_team == k]] = True
        opp[t_idx[t_team == 1 - k]] = True
        s_own, s_opp, u_own, u_opp = since_last(own), since_last(opp), until_next(own), until_next(opp)
        last_own = np.where((s_own >= CLIP) & (s_opp >= CLIP), np.nan, (s_own < s_opp).astype(np.float32))
        next_own = np.where((u_own >= CLIP) & (u_opp >= CLIP), np.nan, (u_own < u_opp).astype(np.float32))
        bal_p = (win_mean(own.astype(np.float64), -75, 0) - win_mean(opp.astype(np.float64), -75, 0)) * 76
        bal_n = (win_mean(own.astype(np.float64), 0, 75) - win_mean(opp.astype(np.float64), 0, 75)) * 76
        c = np.where(last >= 0, chain[np.maximum(last, 0)], 0.0)
        lt = np.where(last >= 0, t_team[np.maximum(last, 0)], -1)
        chain_own = np.where(lt == k, c, 0.0)
        chain_opp = np.where(lt == 1 - k, c, 0.0)
        views.append(np.stack([last_own, next_own, s_own, s_opp, u_own, u_opp, bal_p, bal_n, chain_own, chain_opp],
                              1).astype(np.float32))
    return views[0], views[1]


def touch_feats2(P, B, team, gk, pitch):
    """touch_feats plus, per team view: the velocity change of the team's / the opponent's last touch (strong kick =
    pass / clearance, soft = dribble), frames since the team / the opponent last won the touch sequence (a touch
    right after one of the other team), and the longest own / opponent touch chain of the last 6 s.
    Returns (home view, away view), each (n, 16) float32."""
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    n = len(P)
    t_idx, t_team = touches(pxy, bxy, np.asarray(team))
    bv = velocity(bxy)
    dv = np.zeros(len(t_idx), np.float32)
    ok = (t_idx >= K) & (t_idx < n - K)
    dv[ok] = np.linalg.norm(bv[t_idx[ok] + K] - bv[t_idx[ok] - K], axis=1)
    chain = _chains(t_team)
    gain = np.r_[False, t_team[1:] != t_team[:-1]] if len(t_team) else np.zeros(0, bool)
    base = touch_feats(P, B, team, gk, pitch)
    views = []
    for k, x0 in zip((0, 1), base):
        cols = [x0]
        for side in (k, 1 - k):
            sel = t_team == side
            last = np.searchsorted(t_idx[sel], np.arange(n), side="right") - 1
            v = np.where(last >= 0, dv[sel][np.maximum(last, 0)], np.nan)
            age = np.arange(n) - np.where(last >= 0, t_idx[sel][np.maximum(last, 0)], -10 ** 9)
            cols.append(np.where(age < CLIP, v, np.nan)[:, None])
        for side in (k, 1 - k):
            g = np.zeros(n, bool)
            g[t_idx[gain & (t_team == side)]] = True
            cols.append(since_last(g)[:, None])
        for side in (k, 1 - k):
            c = np.zeros(n, np.float32)
            sel = t_team == side
            c[t_idx[sel]] = chain[sel]
            cols.append(maximum_filter1d(c, size=151, origin=75, mode="constant")[:, None])
        views.append(np.hstack(cols).astype(np.float32))
    return views[0], views[1]


def rate(P, B, team, pitch):
    """Detected touches per minute (diagnostic)."""
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    return len(touches(pxy, bxy, np.asarray(team))[0]) / (len(P) / 1500.0)
