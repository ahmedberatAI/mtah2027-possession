"""v20 (rounds 45, 46, 49, 51-55, docs/RAPOR.md, the user's decision): the specialist's weight 0.5 and a sequence model in the dead-ball evidence, every dead-ball model trained on sim.V6's masks.

Team: 0.5 x (0.5 v10 + 0.5 shared BiGRU) + 0.5 x J1, +-1 s smoothing. Round 45: with Metrica's ball de-glued for every channel J1's weight 0.5 passes the decision rule (0.35-0.65 is a plateau); the old 0.25 had been chosen with the glued ball.

Dead-ball evidence, the decoder's alive log-odds per frame:

    0.25 x open-data rules + 0.375 x multi-source LightGBM + 0.375 x sequence model + view terms

  open-data rules   model_v19's form; thresholds and bounds from models/dead_rules_open_v1.json, weights refitted on V6 (models/dead_rules_open_v2.json)
  multi-source      as model_v19's, trained on V6 (models/dead_multi_v2.txt)
  sequence model    src/exp_tdead.py: convolution + bidirectional GRU at 5 Hz on the same tracking features, each feature scaled by this half's own median and spread (so a match's absolute tempo does not move it),
                    auxiliary stop and restart heads, equal provider weight, trained on V6, mean of three seeds (models/tdead_v1/)
  view terms        as model_v19's, both weights from SkillCorner's real flag on V6

sim.V6 (round 49): V3 cut every dead-ball gap exactly at the restart; real tracking does not. V6 draws the ball around stops and restarts from the target's label-free, whole-match aggregate shares (no public window); see src/sim.py.

The decoder's settings are v17's; no verifier (+0.0002 on the primary contexts on v19's evidence, not retrained for this base) and no boundary timing. Frames before the second kick-off (official FAQ) are dead.

Cached: cache/v20/barath_td.pkl, cache/v20/barath_m.pkl, cache/v20/barath_probs.pkl; deterministic on CPU.
"""
import json
import os

import numpy as np

import atd
import bench
import bfree
import dead_timing
import exp_short as es
import model_v7
import model_v9
import model_v10
import model_v13
import model_v15
import model_v17
import model_v19
import short_specialist as ss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M_MODEL = os.path.join(ROOT, "models", "dead_multi_v2.txt")
RULES = os.path.join(ROOT, "models", "dead_rules_open_v2.json")
TD_DIR = os.path.join(ROOT, "models", "tdead_v1")
TD_SEEDS = (20261104, 20261105, 20261106)
OUT = os.path.join(ROOT, "cache", "v20")
DEFAULT = dict(model_v15.DEFAULT)
W_RULES, W_M, W_TD = 0.25, 0.375, 0.375
W_J1 = 0.5


def ensure_m():
    path = os.path.join(OUT, "barath_m.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    model = bench.load_booster(M_MODEL)
    bench._barath_inputs()
    out = {b["a"]: model.predict(f["XF"]) for b, f in zip(bench.blocks("barath"), bench.features(model_v19.VERSION, "barath"))}
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def ensure_td():
    """{block start: mean P(dead) of the three sequence models}, on the CPU so that the numbers do not depend on a GPU."""
    path = os.path.join(OUT, "barath_td.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    os.environ["TD_CPU"] = "1"
    import torch
    import exp_tdead as td
    torch.set_num_threads(4)
    nets = []
    for seed in TD_SEEDS:
        ck = torch.load(os.path.join(TD_DIR, "tdead_v6n_s%d.pt" % seed), map_location="cpu", weights_only=False)
        net = td.Net(ck["n_in"]).to(td.DEVICE)
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append((net, ck["mean"], ck["scale"]))
    bench._barath_inputs()
    os.environ["TD_NORM"] = "1"
    out = {b["a"]: np.mean([td.score_half(n, m, s, f["XF"]) for n, m, s in nets], axis=0).astype(np.float32)
           for b, f in zip(bench.blocks("barath"), bench.features(model_v19.VERSION, "barath"))}
    os.environ.pop("TD_NORM")
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def ensure_probs():
    """Team probability: J1 at weight 0.5 (model_v15's recipe otherwise)."""
    path = os.path.join(OUT, "barath_probs.pkl")
    if os.path.isfile(path):
        return path
    gru = bench._load(model_v13.ensure_gru())
    j1 = model_v15.ensure_c1(cfg=model_v17.CONFIG)
    model = model_v10.team_model()
    bench._barath_inputs()
    out = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw = bench.predict_team(model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
        blend = es.blend([raw], [gru[b["a"]][1]], model_v15.W_BIGRU)[0]
        out[b["a"]] = (None, model_v9.smooth((1 - W_J1) * blend + W_J1 * ss.pk(j1[b["a"]], 0)))
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return path


def alive_evidence(f):
    with open(RULES) as fh:
        conf = json.load(fh)
    beta, gamma = np.asarray(conf["beta"]), np.asarray(conf["gamma"])
    m, td_ = ensure_m(), ensure_td()
    valid, nvis = np.asarray(f["b_valid"]), np.asarray(f["b_nvis"])
    a = np.zeros(f["n"])
    for b in bench.blocks("barath"):
        sl = slice(b["a"], b["b"] + 1)
        rules = model_v19.rule_terms(model_v19.rule_fields(b), conf) @ beta
        view = gamma[0] * (~valid[sl]) + gamma[1] * (valid[sl] & (nvis[sl] >= model_v19.MIN_VISIBLE))
        a[sl] = W_RULES * rules + W_M * dead_timing.alive_logit(m[b["a"]]) + W_TD * dead_timing.alive_logit(td_[b["a"]]) + view
    return a


def scores(f, probs_path):
    a = alive_evidence(f)
    lh = np.full(f["n"], np.log(0.5))
    la = np.full(f["n"], np.log(0.5))
    for s, (_, ph) in bench._load(probs_path).items():
        ph = np.clip(ph, 1e-4, 1 - 1e-4)
        lh[s:s + len(ph)] = np.log(ph)
        la[s:s + len(ph)] = np.log(1 - ph)
    S = np.stack([a / 2 + lh, a / 2 + la, -a / 2], 1)
    S[atd.HALVES[1][1]:atd.KICKOFF_2ND, 2] += 1e6
    return S


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = scores(f, ensure_probs())
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for _, a, e in atd.HALVES:
        labels[a:e + 1] = sp.decode(S[a:e + 1])
    return labels
