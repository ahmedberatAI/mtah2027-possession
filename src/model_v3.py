"""v3 = v2 + aligned-broadcast cues (shot validity / live agreement) in the aliveness term.
Use with features_merged.pkl to also get broadcast ball fill-in."""
import numpy as np
import model_v2
from model_v2 import team_margin
import atd
from decode import decode_halves

DEFAULT = dict(model_v2.DEFAULT)
DEFAULT.update(spd_shift=37, w_binvalid=1.5, w_blive=0.4, dis_lo=1.0, dis_hi=2.5, w_dis_mid=0.7, w_dis_hi=0.35)


def scores(f, p):
    a = model_v2.aliveness(f, p)
    have = f["b_have"]
    a[have & ~f["b_valid"]] -= p["w_binvalid"]
    a[have & f["b_live"]] += p["w_blive"]
    e = team_margin(f, p)
    # two-camera ball disagreement: the ball is probably in the air (or one detection is wrong),
    # so the nearest player to its ground projection is a weak cue
    both = f["tac_seen"] & f["b_seen"] & f["b_live"]
    dis = np.full(f["n"], np.nan)
    dis[both] = np.linalg.norm(f["bxy"][both] - f["b_ball"][both], axis=1)
    e = np.where(both & (dis >= p["dis_lo"]) & (dis < p["dis_hi"]), e * p["w_dis_mid"], e)
    e = np.where(both & (dis >= p["dis_hi"]), e * p["w_dis_hi"], e)
    S = np.stack([a / 2 + e / 2, a / 2 - e / 2, -a / 2], axis=1)
    S[atd.HALVES[1][1]:atd.KICKOFF_2ND, 2] += 1e6
    return S


def predict(f, params=None, return_scores=False):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = scores(f, p)
    pen = np.array([[0, p["pen_team"], p["pen_dead"]], [p["pen_team"], 0, p["pen_dead"]],
                    [p["pen_dead"], p["pen_dead"], 0]], dtype=np.float64)
    min_dur = np.array([p["min_poss"], p["min_poss"], p["min_dead"]])
    path = decode_halves(S, pen, min_dur, [(a, b) for _, a, b in atd.HALVES])
    labels = np.full(f["n"], 3, np.int8)
    labels[path >= 0] = path[path >= 0]
    return (labels, S) if return_scores else labels
