"""v16 = v15 + dead-ball boundary timing from the learned dead model (round 29, docs/RAPOR.md: DT).

v15's labels keep their dead stretches (v3's rules + the aligned broadcast, then the verifier); each stop and restart
moves onto the nearest boundary of the same kind in the learned dead model's own decode (v15's team probability,
BAR-ATH's decoder), within +-2 s and never across another boundary (src/dead_timing.py). Judged with BAR-ATH-like dead
ball on the open sets (src/exp_dtime.py, before the verifier): DFL evctrl +0.016, Metrica ctrl +0.005, SkillCorner
phase / player possession +0.003 / +0.003, SoccerTrack v2 +0.005 / +0.003 (not used to design the round). Here it is
the last step.

The learned dead model is the all-DFL LightGBM of src/bench.py (DFL's own ball status, sim.V3, "base" features), kept
in models/dead_base_v3.txt (committed; the model the evaluation used on Metrica, SkillCorner and SoccerTrack), so the
submission can be regenerated from the provided data alone. MTAH_RETRAIN_DEAD=1 (or a missing file) retrains it from
the open DFL data; a retrain today is not byte-identical to the committed file (written on 21 October), so
MTAH_RETRAIN=1 (the team classifier) leaves it alone. Its BAR-ATH probabilities are cached in cache/v16/barath_dead.pkl
(deterministic).
"""
import os

import numpy as np

import bench
import dead_timing
import model_v15

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "dead_base_v3.txt")
DEAD = os.path.join(ROOT, "cache", "v16", "barath_dead.pkl")
VERSION = "base"
DEFAULT = dict(model_v15.DEFAULT)


def dead_model(retrain=None):
    if retrain is None:
        retrain = os.environ.get("MTAH_RETRAIN_DEAD") == "1"
    if os.path.isfile(MODEL) and not retrain:
        return bench.load_booster(MODEL)
    model = bench.fit_dead(bench.features(VERSION, "dfl@v3"), bench.labels("dfl", "native"))
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    model.save_model(MODEL)
    return model


def ensure_dead(refresh=False):
    """{block start: P(dead) per frame} on BAR-ATH's halves."""
    if os.path.isfile(DEAD) and not refresh:
        return bench._load(DEAD)
    model = dead_model()
    bench._barath_inputs()
    out = {b["a"]: model.predict(f["XF"]) for b, f in zip(bench.blocks("barath"), bench.features(VERSION, "barath"))}
    os.makedirs(os.path.dirname(DEAD), exist_ok=True)
    bench._dump(out, DEAD)
    return out


def predict(f, params=None, cfg=model_v15.V15):
    labels = np.asarray(model_v15.predict(f, params, cfg=cfg)).copy()
    probs = bench._load(model_v15.ensure_probs(cfg=cfg))
    dead = ensure_dead()
    for b in bench.blocks("barath"):
        sl = slice(b["a"], b["b"] + 1)
        labels[sl] = dead_timing.retime(labels[sl], dead_timing.learned_decode(dead[b["a"]], probs[b["a"]][1]))
    return labels
