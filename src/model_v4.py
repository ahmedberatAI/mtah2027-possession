"""v4: v3 alive/dead decoding + an explicit, event-based possession tracker inside alive spells.

Control frames: the nearest player is within r_ctrl of the ball, the nearest opponent is at least
excl metres further, and the ball moves with the player (relative speed < ctrl_rel) or slowly.
Control episodes: runs of control frames of one team (gaps <= join frames bridged).
A change of possession to team Y happens at the start of a Y episode that is CONFIRMED:
  - it lasts >= confirm frames ("keeps the ball"), or
  - another Y episode follows within chain_s seconds before any X episode ("consecutive passes").
Unconfirmed opponent touches are deflections and do not change possession.
After a dead ball, the first episode (confirmed or not) decides the restarting team.
"""
import numpy as np

import atd
import model_v3
from features import runs_of

DEFAULT = dict(model_v3.DEFAULT)
DEFAULT.update(r_ctrl=1.25, excl=0.5, ctrl_rel=3.0, slow_ball=3.0, join=3, min_ep=3,
               confirm=12, chain_s=3.0, lead=0)


def control_frames(f, p):
    seen = f["seen"]
    dh, da = f["dh"], f["da"]
    home = seen & (dh <= p["r_ctrl"]) & (da >= dh + p["excl"])
    away = seen & (da <= p["r_ctrl"]) & (dh >= da + p["excl"])
    rel = np.nan_to_num(f["rel_speed"], nan=99)
    bs = np.nan_to_num(f["bspeed"], nan=99)
    calm = (rel < p["ctrl_rel"]) | (bs < p["slow_ball"])
    c = np.full(f["n"], -1, np.int8)
    c[home & calm] = 0
    c[away & calm] = 1
    return c


def episodes(c, p):
    """[(start, stop, team)] with same-team gaps <= join bridged and length >= min_ep."""
    out = []
    for k in (0, 1):
        m = c == k
        r = runs_of(m)
        merged = []
        for a, b in r:
            if merged and a - merged[-1][1] <= p["join"]:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out += [(a, b, k) for a, b in merged if b - a >= p["min_ep"]]
    out.sort()
    return out


def team_track(eps, lo, hi, prior, p):
    """Possession team per frame in [lo, hi) from episodes; prior = team before lo (or -1)."""
    eps = [e for e in eps if lo <= e[0] < hi]
    lab = np.full(hi - lo, -1, np.int8)
    cur = prior
    cur_start = lo
    chain = int(p["chain_s"] * atd.FPS)
    changes = []
    for i, (a, b, k) in enumerate(eps):
        if cur == -1:
            cur = k
            changes.append((lo, k))
            continue
        if k == cur:
            continue
        confirmed = (b - a) >= p["confirm"]
        if not confirmed:
            for a2, b2, k2 in eps[i + 1:]:
                if a2 - b > chain:
                    break
                if k2 != k:
                    break
                confirmed = True
                break
        if confirmed:
            cur = k
            changes.append((max(lo, a - p["lead"]), k))
    if not changes:
        lab[:] = prior  # -1 -> caller falls back to the HMM's team
        return lab
    first = changes[0][1] if prior < 0 else prior
    lab[:] = first
    for t, k in changes:
        lab[t - lo:] = k
    return lab


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    base = model_v3.predict(f, p)
    c = control_frames(f, p)
    eps = episodes(c, p)
    out = base.copy()
    for _, h0, h1 in atd.HALVES:
        seg_lab = base[h0:h1 + 1]
        alive = (seg_lab == 0) | (seg_lab == 1)
        prior = -1
        for a, b in runs_of(alive):
            A, B = h0 + a, h0 + b
            # a restart after a dead ball: the first touch decides, no prior
            lab = team_track(eps, A, B, -1 if (A > h0 and base[A - 1] == 2) else prior, p)
            # fall back to the HMM team where the tracker had no evidence at all
            out[A:B] = np.where(lab >= 0, lab, base[A:B])
            prior = out[B - 1]
    return out
