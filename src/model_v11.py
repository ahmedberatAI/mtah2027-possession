"""v11 = v10 + ball-touch event features (round 9, src/touch.py, src/exp_touch.py): a touch is a sharp change of the
ball's velocity with a player within 2 m, credited to that player's team; per team view the last / next touch's
team, time since / until each team's touch, the touch balance over 3 s and the current same-team touch chain. Team
model trained as v10 (DFL native + Metrica ctrl weight 2), +-25-frame smoothing, v7's decoder and dead ball.

Team probabilities are cached in cache/v11/barath_probs.pkl (deterministic).
"""
import os

import numpy as np

import bench
import bfree
import model_v7
import model_v9
import model_v10
import touch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = os.path.join(ROOT, "cache", "v11", "barath_probs.pkl")
DEFAULT = dict(model_v7.DEFAULT)


def _aug(src):
    import exp_touch
    return [dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))
            for f, (h, a) in zip(model_v9._aug(src), exp_touch.extra(src))]


def ensure_probs(refresh=False):
    if os.path.isfile(PROBS) and not refresh:
        return PROBS
    import lightgbm as lgb
    Xd, yd, _ = bench.team_rows(_aug("dfl@v3"), bench.labels("dfl", "native"))
    Xm, ym, _ = bench.team_rows(_aug("metrica@v3"), bench.labels("metrica@v3", "ctrl"))
    X = np.concatenate([Xd, Xm])
    y = np.concatenate([yd, ym])
    w = np.concatenate([np.ones(len(yd), np.float32), np.full(len(ym), model_v10.W_M / 3.0, np.float32)])
    del Xd, Xm
    # the thread count is pinned: LightGBM's deterministic mode is only reproducible for a fixed num_threads
    model = lgb.train(dict(bench.LGB_TEAM, num_threads=8), lgb.Dataset(X, y, weight=w), bench.ROUNDS)
    probs = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        th, ta = touch.touch_feats(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        fa = dict(XF=f["XF"], XH=np.hstack([f["XH"], h, th]), XA=np.hstack([f["XA"], a, ta]))
        probs[b["a"]] = (None, model_v9.smooth(bench.predict_team(model, fa)))
    os.makedirs(os.path.dirname(PROBS), exist_ok=True)
    bench._dump(probs, PROBS)
    return PROBS


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    return model_v7.predict(f, p, probs_path=ensure_probs())
