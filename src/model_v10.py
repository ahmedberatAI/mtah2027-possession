"""v10 = v9 with the Metrica games weighted W_M = 2 (one DFL match = 1) in the team model's training (round 9,
src/exp_combo.py COMBO_WM): averaged over two LightGBM seeds, against v7 (learned / true D) DFL-trk +0.011 / +0.018,
Metrica-ctrl +0.008 / +0.011, SkillCorner phase +0.015 / +0.013, player-possession +0.009 / +0.006 -- better than v9
on DFL and Metrica (the tracking vendor and label convention closest to BAR-ATH), within noise on SkillCorner; lower
worst-case regret. Everything else (features, smoothing, dead ball, decoder, options) is v9's / v7's.

The team classifier is trained on open data only and kept in models/v10_team.txt (committed), so the submission can be
regenerated from the provided data alone; MTAH_RETRAIN=1 (or a missing file) retrains it from the open data and
rewrites the file. Team probabilities are cached in cache/v10/barath_probs.pkl (deterministic).
"""
import os

import numpy as np

import bench
import bfree
import model_v7
import model_v9

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = os.path.join(ROOT, "cache", "v10", "barath_probs.pkl")
MODEL = os.path.join(ROOT, "models", "v10_team.txt")
W_M = 2.0
DEFAULT = dict(model_v7.DEFAULT)


def team_model(retrain=None):
    """LightGBM team classifier: DFL native labels + the Metrica games' controlled-possession labels (weight W_M per
    game), all degraded with sim.V3; loaded from MODEL unless it is missing or a retrain is asked for."""
    import lightgbm as lgb
    if retrain is None:
        retrain = os.environ.get("MTAH_RETRAIN") == "1"
    if os.path.isfile(MODEL) and not retrain:
        return bench.load_booster(MODEL)
    Xd, yd, _ = bench.team_rows(model_v9._aug("dfl@v3"), bench.labels("dfl", "native"))
    Xm, ym, _ = bench.team_rows(model_v9._aug("metrica@v3"), bench.labels("metrica@v3", "ctrl"))
    X = np.concatenate([Xd, Xm])
    y = np.concatenate([yd, ym])
    w = np.concatenate([np.ones(len(yd), np.float32), np.full(len(ym), W_M / 3.0, np.float32)])
    del Xd, Xm
    # the thread count is pinned: LightGBM's deterministic mode is only reproducible for a fixed num_threads
    model = lgb.train(dict(bench.LGB_TEAM, num_threads=8), lgb.Dataset(X, y, weight=w), bench.ROUNDS)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    model.save_model(MODEL)
    return model


def ensure_probs(refresh=False):
    if os.path.isfile(PROBS) and not refresh:
        return PROBS
    model = team_model()
    bench._barath_inputs()                 # BAR-ATH blocks (built here when the model was not retrained)
    probs = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        fa = dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))
        probs[b["a"]] = (None, model_v9.smooth(bench.predict_team(model, fa)))
    os.makedirs(os.path.dirname(PROBS), exist_ok=True)
    bench._dump(probs, PROBS)
    return PROBS


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    return model_v7.predict(f, p, probs_path=ensure_probs())
