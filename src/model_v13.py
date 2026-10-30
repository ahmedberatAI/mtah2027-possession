"""v13 = v10 + the short-possession verifier (round 15, src/short_verifier.py, src/exp_verify.py).

v10 decides the dead ball, the team probabilities and the decoded possessions exactly as before. The verifier then takes
a second look at v10's short decisions: a flip the decoder suppressed is inserted when the verifier gives it p >= 0.65,
and a decoded possession shorter than 4 s is handed to the other team when p < 0.35. Its features are identity-free
tracking cues, v10's margins and the shared BiGRU's margin (models/bigru_dfl_pff_v1, trained on open DFL + PFF data;
its BAR-ATH probabilities are rebuilt by src/reproduce_shared_bigru.py when the cache is missing). On the external sets
(three v10 seeds, ten contexts) it lowered the worst-case regret from 0.0202 to 0.0037; see docs/RAPOR.md round 15.
"""
import os
import subprocess
import sys

import model_v10
import short_verifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = dict(model_v10.DEFAULT)
VERIFIER = dict(use_g=True, use_ins=True, use_del=True, t_ins=0.65, t_del=0.35)
GRU = os.path.join(ROOT, "cache", "shared_bigru", "barath_equal_probabilities.pkl")


def ensure_gru():
    if not os.path.isfile(GRU):
        subprocess.run([sys.executable, "-I", os.path.join(ROOT, "src", "reproduce_shared_bigru.py"),
                        "--device", "cpu"], check=True)
    return GRU


def predict(f, params=None):
    labels = model_v10.predict(f, params)
    ensure_gru()
    return short_verifier.apply_barath(labels, **VERIFIER)
