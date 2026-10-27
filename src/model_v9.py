"""v9 = v8 (team model trained on DFL + the Metrica sample games) with the ball-free short-horizon team interaction
features (src/bfree.py) added to the team model, and its team probabilities smoothed in logit space over +-SMOOTH
frames before the v7 decoder (round 9; src/exp_combo.py). On held-out matches against v7 (learned / true D):
DFL-trk +0.008 / +0.012 (6/7), Metrica-ctrl +0.007 / +0.009, SkillCorner phase +0.017 / +0.016, player-possession
+0.011 / +0.009; worst-case regret 0.0038 (A2 alone 0.0118). Everything else (dead ball, B2 decoder, options) is
model_v7's.

Team probabilities are cached in cache/v9/barath_probs.pkl (deterministic: fixed seeds, LightGBM deterministic mode,
pinned thread count).
"""
import os

import numpy as np

import bench
import bfree
import model_v7
from mlfeat import win_mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = os.path.join(ROOT, "cache", "v9", "barath_probs.pkl")
SMOOTH = 25
DEFAULT = dict(model_v7.DEFAULT)
EPS = 1e-4


def _aug(src):
    import exp_bfree
    out = []
    for b, f, (h, a) in zip(bench.blocks(src), bench.features(model_v7.VERSION, src), exp_bfree.extra(src)):
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
    return out


def smooth(p, w=SMOOTH):
    p = np.clip(p, EPS, 1 - EPS)
    z = win_mean(np.log(p / (1 - p)), -w, w)
    return 1 / (1 + np.exp(-z))


def ensure_probs(refresh=False):
    if os.path.isfile(PROBS) and not refresh:
        return PROBS
    import lightgbm as lgb
    Xd, yd, _ = bench.team_rows(_aug("dfl@v3"), bench.labels("dfl", "native"))
    Xm, ym, _ = bench.team_rows(_aug("metrica@v3"), bench.labels("metrica@v3", "ctrl"))
    X = np.concatenate([Xd, Xm])
    y = np.concatenate([yd, ym])
    w = np.concatenate([np.ones(len(yd), np.float32), np.full(len(ym), 1.0 / 3.0, np.float32)])
    del Xd, Xm
    # the thread count is pinned: LightGBM's deterministic mode is only reproducible for a fixed num_threads
    model = lgb.train(dict(bench.LGB_TEAM, num_threads=8), lgb.Dataset(X, y, weight=w), bench.ROUNDS)
    probs = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        fa = dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))
        probs[b["a"]] = (None, smooth(bench.predict_team(model, fa)))
    os.makedirs(os.path.dirname(PROBS), exist_ok=True)
    bench._dump(probs, PROBS)
    return PROBS


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    return model_v7.predict(f, p, probs_path=ensure_probs())
