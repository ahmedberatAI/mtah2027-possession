"""v2: factorised evidence -- alive/dead log-odds and H/A log-odds are separate, then decoded.

  S[H] =  a/2 + e/2      S[A] =  a/2 - e/2      S[D] = -a/2

a = aliveness log-odds (players' tempo, ball outside the lines, ball movement, ball absence),
e = home-vs-away log-odds (who is in control of the ball), zero-sum so that a player standing
next to a dead ball cannot make the ball look alive (the main v1 failure).
"""
import numpy as np

import atd
from decode import decode_halves
from features import rolling_mean, rolling_frac

DEFAULT = dict(
    # aliveness
    v0=1.55, kv=2.2, v_clip=2.5,          # tempo: (mean player speed - v0) * kv, +-1 s window
    v0_long=1.55, kv_long=1.2,            # +-3 s window
    out_margin=0.5, w_out=4.0,            # ball beyond a line
    w_missing=0.25, missing_after=25,     # ball unseen for > 1 s
    still_disp=0.6, w_still=0.8,          # ball moved < still_disp m over +-0.5 s
    ball_fast=5.0, w_ballmove=0.6,
    # team
    r_ctrl=1.5, r_sigma=1.0, ctrl_rel=2.5, ctrl_boost=1.0, contest_scale=1.0, w_team=2.0,
    # decoding
    pen_team=12.0, pen_dead=30.0, min_poss=8, min_dead=50,
)


def aliveness(f, p):
    n = f["n"]
    seen = f["seen"]
    sh = int(p.get("spd_shift", 0))   # >0: look ahead -- players react to the referee late
    v1 = np.nan_to_num(np.roll(f["spd_all_s25"], -sh), nan=p["v0"])
    v3 = np.nan_to_num(np.roll(f["spd_all_s100"], -sh), nan=p["v0_long"])
    a = np.clip((v1 - p["v0"]) * p["kv"], -p["v_clip"], p["v_clip"])
    a += np.clip((v3 - p["v0_long"]) * p["kv_long"], -p["v_clip"], p["v_clip"])
    out = np.nan_to_num(f["ball_out"], nan=-1) > p["out_margin"]
    a[out] -= p["w_out"]
    a[(~seen) & (f["since_seen"] > p["missing_after"])] -= p["w_missing"]
    # ball stillness over +-12 frames using observed positions only
    bxy = f["bxy"]
    lo = np.clip(np.arange(n) - 12, 0, n - 1)
    hi = np.clip(np.arange(n) + 12, 0, n - 1)
    ok = seen & seen[lo] & seen[hi]
    disp = np.where(ok, np.linalg.norm(bxy[hi] - bxy[lo], axis=1), np.nan)
    still = ok & (disp < p["still_disp"])
    a[still] -= p["w_still"]
    inside = np.nan_to_num(f["ball_out"], nan=1) <= 0
    moving = seen & inside & (np.nan_to_num(f["bspeed"]) > p["ball_fast"])
    a[moving] += p["w_ballmove"]
    return a


def team_margin(f, p):
    seen = f["seen"]
    dh, da = f["dh"], f["da"]
    dmin = np.minimum(dh, da)
    ok = seen & np.isfinite(dmin) & (dmin <= p["r_ctrl"])
    w = np.where(ok, np.exp(-np.where(ok, dmin, 0) / p["r_sigma"]), 0.0)
    gap = np.clip(np.where(ok, da - dh, 0), -50, 50)
    contest = np.tanh(np.abs(gap) / p["contest_scale"])
    rel = f["rel_speed"]
    ctrl = 1.0 + p["ctrl_boost"] * (np.isfinite(rel) & (rel < p["ctrl_rel"]))
    return p["w_team"] * w * contest * ctrl * np.sign(gap)


def scores(f, p):
    a = aliveness(f, p)
    e = team_margin(f, p)
    S = np.stack([a / 2 + e / 2, a / 2 - e / 2, -a / 2], axis=1)
    S[atd.HALVES[1][1]:atd.KICKOFF_2ND, 2] += 1e6
    return S


def predict(f, params=None, return_scores=False):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = scores(f, p)
    pen = np.array([[0, p["pen_team"], p["pen_dead"]],
                    [p["pen_team"], 0, p["pen_dead"]],
                    [p["pen_dead"], p["pen_dead"], 0]], dtype=np.float64)
    min_dur = np.array([p["min_poss"], p["min_poss"], p["min_dead"]])
    path = decode_halves(S, pen, min_dur, [(a, b) for _, a, b in atd.HALVES])
    labels = np.full(f["n"], 3, np.int8)
    labels[path >= 0] = path[path >= 0]
    return (labels, S) if return_scores else labels
