"""Player-level controller model (round 7, docs/RAPOR.md; research R1/R2: "which candidate controls the ball",
with the presser as a hard negative, instead of team-pooled nearest-player cues).

Candidates per frame: the 3 players of each team nearest to the ball. Features follow the SAME player through time
(distance, relative ball-player velocity, approach speed, dwell near the ball, share of the last / next second as
the nearest player, opponents around him, team-mates around the ball), plus ball cues. Supervision: SkillCorner
open data -- a candidate is positive while he is the player in a SkillCorner "player possession" (controlled:
able to pass, shoot or carry; blocks and uncontrolled touches are not possessions); everyone else near the ball,
including SkillCorner's "on-ball engagement" pressers, is negative. Team evidence per frame:
q_team = 1 - prod(1 - p_candidate), z_ctrl = log(q_H / q_A).

Works on bench blocks (degraded inputs, any source); labels only for SkillCorner.
"""
import os

import numpy as np
import pandas as pd

import skillcorner as scr
from mlfeat import velocity, win_mean

K = 3            # candidates per team
STEP = 5         # rows every STEP frames
EPS = 1e-4


def slot_tracks(b):
    """(n, slots) per-player tracks: distance to the ball, relative speed, approach speed, dwell, nearest flags."""
    L, W = b["pitch"]
    pxy = b["P"][:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = b["B"][:, :2].astype(np.float32) * np.array([L, W], np.float32)
    pv = velocity(pxy)
    pv[np.linalg.norm(pv, axis=2) > 12] = np.nan
    bv = velocity(bxy)
    bv[np.linalg.norm(bv, axis=1) > 45] = np.nan
    to = bxy[:, None, :] - pxy
    dist = np.linalg.norm(to, axis=2)
    rel = np.linalg.norm(bv[:, None, :] - pv, axis=2)
    unit = to / np.maximum(dist, 1e-6)[:, :, None]
    approach = np.sum((pv - bv[:, None, :]) * unit, axis=2)
    sp = np.linalg.norm(pv, axis=2)
    cos = np.sum(pv * to, axis=2) / np.maximum(sp * dist, 1e-6)
    d_inf = np.where(np.isfinite(dist), dist, np.inf)
    nearest = np.argmin(d_inf, axis=1)
    near_flag = np.zeros_like(dist, dtype=np.float32)
    ok = np.isfinite(d_inf.min(1))
    near_flag[np.flatnonzero(ok), nearest[ok]] = 1.0
    within2 = (np.nan_to_num(dist, nan=99.0) < 2.0).astype(np.float32)
    dwell = np.zeros_like(within2)
    for t in range(1, len(within2)):
        dwell[t] = (dwell[t - 1] + 1) * within2[t]
    team = b["team"]
    opp_near = np.zeros_like(dist, dtype=np.float32)
    for k in (0, 1):
        own, oth = team == k, team != k
        dd = np.linalg.norm(pxy[:, own, None, :] - pxy[:, None, oth, :], axis=-1)
        opp_near[:, own] = np.nansum(dd < 3.0, axis=2)
    return dict(dist=dist, rel=rel, approach=approach, speed=sp, cos=cos, near=near_flag, dwell=np.minimum(dwell, 125),
                opp3=opp_near, seen=np.isfinite(bxy[:, 0]), bspeed=np.linalg.norm(bv, axis=1))


def _win(x, lo, hi):
    """win_mean applied column-wise to (n, slots)."""
    return np.stack([win_mean(x[:, j], lo, hi) for j in range(x.shape[1])], 1)


def rows(b, step=STEP, tr=None):
    """Feature rows for candidates at frames 0, step, 2 step, ...: X (m, F), frame (m,), slot (m,), team (m,)."""
    tr = tr or slot_tracks(b)
    n, s = tr["dist"].shape
    team = b["team"]
    W = {}
    for name in ("dist", "rel", "approach", "near"):
        x = tr[name].astype(np.float64)
        for lo, hi in ((-6, 6), (-25, 0), (0, 25)):
            W["%s_w%d_%d" % (name, lo, hi)] = _win(x, lo, hi)
    frames = np.arange(0, n, step)
    Xs, fr, sl, tm = [], [], [], []
    d = np.where(np.isfinite(tr["dist"]), tr["dist"], np.inf)
    for k in (0, 1):
        cols = np.flatnonzero(team == k)
        order = np.argsort(d[frames][:, cols], axis=1)[:, :K]
        for r in range(K):
            j = cols[order[:, r]]
            ok = np.isfinite(d[frames, j])
            f_ok, j_ok = frames[ok], j[ok]
            dk = d[f_ok][:, cols]
            dopp = np.where(np.isfinite(d[f_ok][:, team != k]), d[f_ok][:, team != k], np.inf).min(1)
            feats = [tr["dist"][f_ok, j_ok], np.full(len(f_ok), r, np.float32), dopp, dopp - tr["dist"][f_ok, j_ok],
                     tr["rel"][f_ok, j_ok], tr["approach"][f_ok, j_ok], tr["speed"][f_ok, j_ok], tr["cos"][f_ok, j_ok],
                     tr["dwell"][f_ok, j_ok], tr["opp3"][f_ok, j_ok], tr["seen"][f_ok].astype(np.float32),
                     tr["bspeed"][f_ok], np.sort(dk, axis=1)[:, 1] if dk.shape[1] > 1 else np.full(len(f_ok), np.nan)]
            feats += [W[name][f_ok, j_ok] for name in sorted(W)]
            Xs.append(np.stack(feats, 1).astype(np.float32))
            fr.append(f_ok), sl.append(j_ok), tm.append(np.full(len(f_ok), k, np.int8))
    return np.concatenate(Xs), np.concatenate(fr), np.concatenate(sl), np.concatenate(tm)


def sc_controller(match, half, n):
    """(n,) slot index of SkillCorner's player in possession at each 25 fps frame of a half, -1 if none."""
    d = scr.parse(match)
    pid = list(d["pid"])
    f0 = int(d["h%d_f0" % half][0])
    ev = pd.read_csv(os.path.join(scr.DATA, match, "%s_dynamic_events.csv" % match), low_memory=False,
                     usecols=["event_type", "period", "frame_start", "frame_end", "player_id"])
    pp = ev[(ev["event_type"] == "player_possession") & (ev["period"] == half)]
    out = np.full(n, -1, np.int64)
    for r in pp.itertuples(index=False):
        if r.player_id not in pid:
            continue
        a = int(round((r.frame_start - f0) * 2.5))
        b_ = int(round((r.frame_end - f0) * 2.5)) + 3
        out[max(0, a):max(0, min(n, b_))] = pid.index(r.player_id)
    return out


def team_evidence(p, fr, tm, n):
    """Per-frame z_ctrl = log(q_H / q_A) and q_none, interpolated from the row frames to all n frames."""
    frames = np.unique(fr)
    pos = np.searchsorted(frames, fr)
    logn = np.zeros((len(frames), 2))
    np.add.at(logn, (pos, tm.astype(int)), np.log(np.clip(1 - p, EPS, 1)))
    q = 1 - np.exp(logn)
    q = np.clip(q, EPS, 1 - EPS)
    z = np.log(q[:, 0] / q[:, 1])
    none = np.exp(logn.sum(1))
    t = np.arange(n)
    return np.interp(t, frames, z), np.interp(t, frames, none)
