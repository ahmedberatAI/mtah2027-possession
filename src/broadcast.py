"""Broadcast source mapped onto the tactical timeline.

Measured with src/align_ball.py: the broadcast is a real-time recording cut at halftime, so
  1st half: tactical_frame = broadcast_frame + 4014
  2nd half: tactical_frame = broadcast_frame + 26920
(the per-2-minute best lag is constant to +-1-2 frames, median ball disagreement ~0.5-1 m).

Per tactical frame this gives the broadcast ball (where its homography is valid), whether the
broadcast shows live pitch action that agrees with the tactical tracking, or something else
(replay, close-up, crowd) -- the latter mostly happens while the ball is dead.
"""
import numpy as np
import pandas as pd

import atd

LAG = {1: 4014, 2: 26920}


def tactical_to_broadcast(n_tac=atd.N_FRAMES["tactical"], n_bro=atd.N_FRAMES["broadcast"]):
    """Index array: broadcast frame for each tactical frame, -1 where none."""
    out = np.full(n_tac, -1, np.int64)
    half = atd.half_of_frame(n_tac)
    t = np.arange(n_tac)
    # assign by half; outside the graded halves use the nearer half's lag
    lag = np.where(t < (atd.HALVES[0][2] + atd.HALVES[1][1]) // 2, LAG[1], LAG[2])
    b = t - lag
    ok = (b >= 0) & (b < n_bro)
    out[ok] = b[ok]
    return out


def build(tac_f=None):
    bro = atd.load("broadcast")
    hom = pd.read_csv(atd.paths("broadcast")["homography"], usecols=["valid", "reprojection_error"])
    valid = hom["valid"].values.astype(bool)
    err = hom["reprojection_error"].values
    idx = tactical_to_broadcast()
    n = len(idx)
    have = idx >= 0
    bidx = np.where(have, idx, 0)
    b_valid = have & valid[bidx]
    bball = atd.to_metres(bro["ball"][bidx, :2])
    bball[~b_valid] = np.nan
    b_seen = np.isfinite(bball[:, 0])
    # live agreement: broadcast players vs tactical players (median nearest same-team distance)
    tac = atd.load("tactical")
    Pt = atd.to_metres(tac["players"][:, :, :2])
    Pb = atd.to_metres(bro["players"][bidx, :, :2])
    tt, tb = tac["team"], bro["team"]
    agree = np.full(n, np.nan)
    step = 10000
    with np.errstate(invalid="ignore"):
        for s in range(0, n, step):
            sl = slice(s, min(n, s + step))
            dmins = []
            for k in (0, 1):
                A = Pt[sl][:, tt == k]          # (m, a, 2)
                Bp = Pb[sl][:, tb == k]         # (m, b, 2)
                dd = np.linalg.norm(Bp[:, :, None, :] - A[:, None, :, :], axis=-1)
                dmin = np.min(np.where(np.isnan(dd), np.inf, dd), axis=2)   # (m, b)
                dmin[np.isinf(dmin)] = np.nan
                dmins.append(dmin)
            agree[sl] = np.nanmedian(np.concatenate(dmins, axis=1), axis=1)
    agree[~b_valid] = np.nan
    n_vis = np.where(b_valid, (~np.isnan(Pb[:, :, 0])).sum(1), 0)
    live = b_valid & (agree < 2.5) & (n_vis >= 4)
    return {"b_have": have, "b_valid": b_valid, "b_err": np.where(have, err[bidx], np.nan),
            "b_ball": bball, "b_seen": b_seen, "b_agree": agree, "b_live": live, "b_nvis": n_vis}
