"""v8 = v7 with the team model trained on open DFL (native labels) PLUS the three Metrica sample games (controlled-
possession "ctrl" labels; same vendor as BAR-ATH's ATD), both degraded with sim.V3; Metrica rows weighted 1/3 per
degradation seed (= one game). Idea A2 (docs/RAPOR.md, round 9): on held-out matches DFL-trk +0.010 / +0.014,
Metrica-ctrl (leave one game out) +0.004 / +0.005, SkillCorner phase +0.009 / +0.008, player-possession +0.003 / 0.000
(learned / true D; src/exp_a2.py). Everything else (features, dead ball, B2 decoder, options) is model_v7's.

Team probabilities are cached in cache/v8/barath_probs.pkl (deterministic: fixed seeds, LightGBM deterministic mode,
pinned thread count).
"""
import os

import numpy as np

import bench
import model_v7

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = os.path.join(ROOT, "cache", "v8", "barath_probs.pkl")
DEFAULT = dict(model_v7.DEFAULT)


def ensure_probs(refresh=False):
    if os.path.isfile(PROBS) and not refresh:
        return PROBS
    import lightgbm as lgb
    fd = bench.features(model_v7.VERSION, "dfl@v3")
    fm = bench.features(model_v7.VERSION, "metrica@v3")
    Xd, yd, _ = bench.team_rows(fd, bench.labels("dfl", "native"))
    Xm, ym, _ = bench.team_rows(fm, bench.labels("metrica@v3", "ctrl"))
    X = np.concatenate([Xd, Xm])
    y = np.concatenate([yd, ym])
    w = np.concatenate([np.ones(len(yd), np.float32), np.full(len(ym), 1.0 / 3.0, np.float32)])
    # the thread count is pinned: LightGBM's deterministic mode is only reproducible for a fixed num_threads
    model = lgb.train(dict(bench.LGB_TEAM, num_threads=8), lgb.Dataset(X, y, weight=w), bench.ROUNDS)
    probs = {b["a"]: (None, bench.predict_team(model, f))
             for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath"))}
    bench._dump(probs, PROBS)
    return PROBS


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    return model_v7.predict(f, p, probs_path=ensure_probs())
