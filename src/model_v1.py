"""v1: transparent evidence model + min-duration Viterbi. No learning.

Per-frame rewards for H / A / D are built from physically motivated cues, then decoded into
segments with switch penalties, which is what the segment-level ranking metric rewards.
"""
import numpy as np

import atd
from decode import decode_halves, H, A, D

DEFAULT = dict(
    r_sigma=1.0,        # m, decay of team evidence with ball-player distance
    margin_scale=1.0,   # m, how decisive (da - dh) is
    ctrl_rel=2.5,       # m/s, ball moving with the nearest player
    ctrl_boost=1.0,
    w_team=1.0,
    slow_lo=0.9, slow_hi=1.4,   # m/s smoothed mean player speed: below lo dead-ish
    fast_lo=1.6, fast_hi=2.4,
    w_slow=0.6, w_fast=0.4,
    out_margin=0.5, w_out=3.0,
    ball_fast=6.0, w_ballmove=0.5,
    w_gap_slow=0.3,
    pen_team=30.0, pen_dead=40.0,
    min_poss=10, min_dead=50,
)


def evidence(f, p):
    n = f["n"]
    S = np.zeros((n, 3))
    seen = f["seen"]
    dh, da = f["dh"], f["da"]
    dmin = np.minimum(dh, da)
    ok = seen & np.isfinite(dmin)
    w = np.where(ok, np.exp(-np.where(ok, dmin, 0) / p["r_sigma"]), 0.0)
    margin = np.tanh(np.clip(np.where(ok, da - dh, 0), -50, 50) / p["margin_scale"])
    rel = f["rel_speed"]
    ctrl = 1.0 + p["ctrl_boost"] * (np.isfinite(rel) & (rel < p["ctrl_rel"]))
    e = w * margin * ctrl
    S[:, H] += p["w_team"] * np.clip(e, 0, None)
    S[:, A] += p["w_team"] * np.clip(-e, 0, None)

    spd = f["spd_all_s25"]
    spd = np.where(np.isfinite(spd), spd, 1.5)
    slow = np.clip((p["slow_hi"] - spd) / (p["slow_hi"] - p["slow_lo"]), 0, 1)
    fast = np.clip((spd - p["fast_lo"]) / (p["fast_hi"] - p["fast_lo"]), 0, 1)
    S[:, D] += p["w_slow"] * slow
    S[:, H] += p["w_fast"] * fast
    S[:, A] += p["w_fast"] * fast

    out = np.nan_to_num(f["ball_out"], nan=-1) > p["out_margin"]
    S[out, D] += p["w_out"]
    inside = np.nan_to_num(f["ball_out"], nan=1) <= 0
    moving = seen & inside & (np.nan_to_num(f["bspeed"]) > p["ball_fast"])
    S[moving, H] += p["w_ballmove"]
    S[moving, A] += p["w_ballmove"]
    S[(~seen) & (spd < p["slow_hi"]), D] += p["w_gap_slow"]
    # FAQ: second half graded from 110808 but kicks off at 113742 -> all D before it
    S[atd.HALVES[1][1]:atd.KICKOFF_2ND, D] += 1e6
    return S


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = evidence(f, p)
    pen = np.array([[0, p["pen_team"], p["pen_dead"]],
                    [p["pen_team"], 0, p["pen_dead"]],
                    [p["pen_dead"], p["pen_dead"], 0]], dtype=np.float64)
    min_dur = np.array([p["min_poss"], p["min_poss"], p["min_dead"]])
    halves = [(a, b) for _, a, b in atd.HALVES]
    path = decode_halves(S, pen, min_dur, halves)
    labels = np.full(f["n"], 3, np.int8)  # X outside the graded halves
    m = path >= 0
    labels[m] = path[m]
    return labels
