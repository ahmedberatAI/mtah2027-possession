"""v5 hybrid: alive/dead evidence from v3 (rules + aligned broadcast), H/A evidence from the
team model learned on degraded open DFL data (cache/dfl/barath_probs.pkl, written by
`train_dfl.py apply`).

  S[H] = a/2 + k*log p_home,  S[A] = a/2 + k*log(1 - p_home),  S[D] = -a/2
(log-probabilities are <= 0, so a confident team call gives no bonus to "alive" over "dead";
the earlier +-e/2 form did, and it pushed the dead-ball share down.)
"""
import os
import pickle

import numpy as np

import atd
import model_v3
from decode import decode_halves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = dict(model_v3.DEFAULT)
DEFAULT.update(k_team=1.0, mix_rule=0.0, pen_team=8.0)
_probs = {}


def team_logp(f, path=os.path.join(ROOT, "cache", "dfl", "barath_probs.pkl")):
    if "lp" not in _probs:
        with open(path, "rb") as fh:
            probs = pickle.load(fh)
        lh = np.full(f["n"], np.log(0.5))
        la = np.full(f["n"], np.log(0.5))
        for a, (pd_, ph) in probs.items():
            ph = np.clip(ph, 1e-4, 1 - 1e-4)
            lh[a:a + len(ph)] = np.log(ph)
            la[a:a + len(ph)] = np.log(1 - ph)
        _probs["lp"] = (lh, la)
    return _probs["lp"]


def scores(f, p):
    S3 = model_v3.scores(f, p)
    # recover v3's a and e: S[H] - S[A] = e ; S[H] + S[A] = a ; (ignore the 1e6 pre-kickoff term)
    a = S3[:, 0] + S3[:, 1]
    e_rule = S3[:, 0] - S3[:, 1]
    lh, la = team_logp(f)
    k = p["k_team"]
    sh = a / 2 + k * lh + p["mix_rule"] * np.minimum(e_rule, 0)
    sa = a / 2 + k * la + p["mix_rule"] * np.minimum(-e_rule, 0)
    return np.stack([sh, sa, S3[:, 2]], axis=1)


def predict(f, params=None):
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
    return labels
