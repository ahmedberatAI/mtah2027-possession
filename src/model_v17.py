"""v17 = v16 with the short-possession specialist J1 in C1's place (round 37, docs/RAPOR.md: J1abETmS).

J1 is C1 (models/kpu_c1/) fine-tuned on the ten target-convention games: DFL evctrl and Metrica ctrl, 2,000 steps at lr 3e-4, every batch half those windows and half C1's own PFF windows (src/short_specialist.py finetune). Metrica's sample ball sits on its
owner, so its windows use bench variant v3g (ball noise 0.8 m), which puts the ball as far from the players as an automatic tracker's. Held out on the open sets, its Metrica outputs on v3g input: against C1 on the same input Metrica ctrl +0.0059 / +0.0051 (learned /
true dead ball), DFL evctrl +0.0020 / +0.0017, primary regret +0.0059, worst guard -0.0050. The model here saw all ten games; BAR-ATH uses seed 20261102, as v15 uses C1's. Everything else is v16's recipe: mix weight 0.25, theta 0.7, the verifier retrained on
J1abETmS's rows (models/short_verifier_*_j1m25s.txt, v13's thresholds), and last the dead-ball boundary timing.

Cached: cache/v17/j1_barath.pkl (J1's BAR-ATH outputs), cache/v17/barath_probs.pkl (team probabilities); deterministic. MTAH_RETRAIN_VERIFIER=1 (or a missing file) retrains the verifier from the research caches of src/exp_kpu.py.
"""
import os

import model_v15
import model_v16

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = dict(model_v15.DEFAULT)
CONFIG = dict(model=os.path.join(ROOT, "models", "kpu_j1", "kpu_J1_s20261102.pt"), out=os.path.join(ROOT, "cache", "v17", "j1_barath.pkl"),
              probs=os.path.join(ROOT, "cache", "v17", "barath_probs.pkl"), rows="J1abETmS", tag="j1m25s")


def predict(f, params=None):
    return model_v16.predict(f, params, cfg=CONFIG)
