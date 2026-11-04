"""v18 = v17 with fused dead-ball evidence (round 40, docs/RAPOR.md).

The decoder's alive log-odds are v3's rules + the aligned broadcast + 0.5 x the learned dead model's alive log-odds
(models/dead_base_v3.txt, the all-DFL LightGBM that times v16's boundaries); the weight was set on 23 October (idea N21)
and not tuned again. The verifier follows as in v17 (its models, its thresholds), on this decode. The boundary timing
step of v16 is gone: on top of the fused evidence it adds less than 0.005.

Why: the open sets' rules used to lack the broadcast, so they understated BAR-ATH's. SkillCorner's real view flag was
transplanted onto DFL and Metrica (src/exp_deadmix.py), so that the rule formula is BAR-ATH's on every set. Held out against
v17's recipe, spell F1 before the verifier: DFL evctrl +0.097, DFL trk +0.095, Metrica ctrl +0.070, SkillCorner evctrl /
phase / player possession +0.060 / +0.054 / +0.053; worst-case regret on the target convention's contexts +0.097.

Cached: nothing of its own (cache/v16/barath_dead.pkl, cache/v17/*); deterministic.
"""
import numpy as np

import atd
import bench
import dead_timing
import model_v7
import model_v15
import model_v16
import model_v17

DEFAULT = dict(model_v15.DEFAULT)
WEIGHT = 0.5


def alive_evidence(n):
    """WEIGHT x the learned dead model's alive log-odds per frame, zero outside the halves."""
    a = np.zeros(n)
    dead = model_v16.ensure_dead()
    for b in bench.blocks("barath"):
        a[b["a"]:b["b"] + 1] = WEIGHT * dead_timing.alive_logit(dead[b["a"]])
    a[atd.HALVES[1][1]:atd.KICKOFF_2ND] = 0.0
    return a


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    cfg = model_v17.CONFIG
    labels = model_v7.predict(f, p, probs_path=model_v15.ensure_probs(cfg=cfg), alive=alive_evidence(f["n"]))
    return model_v15.verify(labels, **model_v15.VERIFIER, cfg=cfg)
