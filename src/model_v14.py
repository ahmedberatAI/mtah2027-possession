"""v14 = the round-13 BiGRU blend + the short-possession verifier retrained on the blend's candidates (round 17).

Team probability: 0.25 x v10's raw P(home) + 0.75 x the shared BiGRU's equal mean (models/bigru_dfl_pff_v1, trained on
open DFL + PFF data; BAR-ATH probabilities rebuilt by src/reproduce_shared_bigru.py when the cache is missing), then
v10's +-25-frame logit smoothing, dead ball and B2 decoder. The verifier (src/short_verifier.py, base="blend") then
re-judges the short decisions with v13's thresholds; its classifiers are trained on the blend's candidates on the open
sets, DFL with the leave-one-match-out BiGRU (src/exp_bigru_lomo.py) so that its margins are held out as on BAR-ATH.
Candidate until the DFL validation of round 17 (docs/RAPOR.md); v13 stays the default until then.

Team probabilities are cached in cache/v14/barath_probs.pkl (deterministic).
"""
import os

import numpy as np

import bench
import bfree
import exp_short as es
import model_v7
import model_v9
import model_v10
import model_v13
import short_verifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = os.path.join(ROOT, "cache", "v14", "barath_probs.pkl")
DEFAULT = dict(model_v10.DEFAULT)
VERIFIER = dict(model_v13.VERIFIER, base="blend")


def ensure_probs(refresh=False):
    if os.path.isfile(PROBS) and not refresh:
        return PROBS
    gru = bench._load(model_v13.ensure_gru())
    model = model_v10.team_model()
    bench._barath_inputs()
    probs = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw = bench.predict_team(model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
        probs[b["a"]] = (None, model_v9.smooth(es.blend([raw], [gru[b["a"]][1]])[0]))
    os.makedirs(os.path.dirname(PROBS), exist_ok=True)
    bench._dump(probs, PROBS)
    return PROBS


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    labels = model_v7.predict(f, p, probs_path=ensure_probs())
    return short_verifier.apply_barath(labels, **VERIFIER)
