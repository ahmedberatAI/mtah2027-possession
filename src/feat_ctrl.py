"""Pressure-aware control cues per team (idea C6, docs/RAPOR.md), as a bench feature plugin.

Error analysis (src/err_spells.py): most missed ground-truth possessions are short spells right after a
turnover in which BOTH teams are near the ball -- the nearest player is often the presser, not the player
in control. The player in control moves WITH the ball; a presser closes in on it. Per team, over its
players near the ball:
  comove      min |v_ball - v_player| among players within 3 m (low = carrying the ball)
  closing     max approach speed towards the ball among players within 6 m (high = pressing)
  front       share of own players within 2 m whose running direction points at the ball (dribbling)
  dwell       frames the team has had a player within 2 m without interruption (capped at 5 s)
each windowed (+-0.25 s, +-1 s, last 1 s, next 1 s), seen from both sides (own / opp keys) for the
side-swap training. Realistic only with sim V3 kinematics (V2's per-frame noise swamps velocities).

  bench.run("base+feat_ctrl:control", "native", var="v3")
"""
import numpy as np

from mlfeat import CLIP, velocity, win_mean

WINS = ((-6, 6), (-25, 25), (-25, 0), (0, 25))


def _dwell(mask, cap=125):
    """Length of the current uninterrupted run of True (0 where False), capped."""
    out = np.zeros(len(mask), np.float32)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = min(run, cap)
    return out


def _team_cues(pxy, pv, bxy, bv, sel):
    """Per-frame cues for the players in `sel` (bool per slot)."""
    to = bxy[:, None, :] - pxy[:, sel]
    dist = np.linalg.norm(to, axis=2)
    v = pv[:, sel]
    rel = np.linalg.norm(bv[:, None, :] - v, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        near3 = dist < 3.0
        comove = np.nanmin(np.where(near3 & np.isfinite(rel), rel, np.nan), axis=1)
        unit = to / np.maximum(dist, 1e-6)[:, :, None]
        approach = np.sum(v * unit, axis=2) - np.sum(bv[:, None, :] * unit, axis=2)   # -d(dist)/dt
        closing = np.nanmax(np.where((dist < 6.0) & np.isfinite(approach), approach, np.nan), axis=1)
        sp = np.linalg.norm(v, axis=2)
        cosd = np.sum(v * to, axis=2) / np.maximum(sp * dist, 1e-6)
        near2 = dist < 2.0
        front = np.nanmean(np.where(near2 & (sp > 1.0), cosd > 0.5, np.where(near2, False, np.nan)), axis=1)
        any2 = np.any(near2, axis=1)
    return comove, closing, front, any2


def control(P, B, team, gk, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    pv = velocity(pxy)
    pv[np.linalg.norm(pv, axis=2) > 12] = np.nan
    bv = velocity(bxy)
    bv[np.linalg.norm(bv, axis=1) > 45] = np.nan
    cues = {}
    for k in (0, 1):
        comove, closing, front, any2 = _team_cues(pxy, pv, bxy, bv, team == k)
        c = {}
        for nm, x in (("comove", comove), ("closing", closing), ("front", front)):
            for a, b in WINS:
                c["%s_w%d_%d" % (nm, a, b)] = win_mean(x, a, b)
        c["dwell"] = _dwell(any2)
        c["dwell_w-25_0"] = win_mean(c["dwell"], -25, 0)
        c["dwell_w0_25"] = win_mean(c["dwell"], 0, 25)
        cues[k] = c
    T = {}
    for own, opp in ((0, 1), (1, 0)):
        t = {}
        for nm in cues[own]:
            t["ctl_own_" + nm] = cues[own][nm]
            t["ctl_opp_" + nm] = cues[opp][nm]
        T[own] = t
    for t in T.values():
        for nm in list(t):
            t[nm] = np.clip(np.asarray(t[nm], np.float32), -CLIP, CLIP)
    return {}, T[0], T[1]
