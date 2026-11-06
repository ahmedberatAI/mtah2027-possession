"""v21 candidate (rounds 61, 65; docs/RAPOR.md; not the default): v20's channels and dead-ball evidence, and a stacker on top of them.

The stacker (src/exp_stack.py / src/exp_crf.py) reads, at 5 Hz, each channel's logit, J1's gain / lose / touch heads, each team's distance to the ball and the alive evidence through a small convolution and a bidirectional GRU, and outputs P(H), P(A), P(D). It was trained on the same channels' held-out outputs on the open games only (DFL, Metrica, SkillCorner, target-convention labels); it has never seen BAR-ATH.

    q      = logit mix 0.5 x stacker + 0.5 x v20's smoothed blend
    alive  = 0.5 x v20's evidence + 0.5 x the stacker's alive log-odds
    decoder: pen_team 6, B2 c 24 (the stacker's fewer false flips allow lower penalties); the rest as v20

Cached: cache/v21/barath_stack_<tag>.pkl; deterministic on CPU.
"""
import os

import numpy as np

import atd
import bench
import bfree
import model_v7
import model_v10
import model_v13
import model_v15
import model_v17
import model_v20
import short_specialist as ss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "v21")
STACK_DIR = os.path.join(ROOT, "models", "stack_v1")
DEFAULT = dict(model_v20.DEFAULT)
DEFAULT.update(pen_team=6.0, b2_c=24.0)
W_Q, W_A = 0.5, 0.5
TAG = os.environ.get("STACK_TAG", "HC")
CK = os.environ.get("STACK_CK", "")           # research: a checkpoint outside models/ while the candidate is open


def _channels():
    """Per half {block start: (v10 raw, BiGRU, J1's outputs, alive evidence)} exactly as v20 computes them."""
    gru = bench._load(model_v13.ensure_gru())
    j1 = model_v15.ensure_c1(cfg=model_v17.CONFIG)
    model = model_v10.team_model()
    bench._barath_inputs()
    out = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw = bench.predict_team(model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
        out[b["a"]] = (raw, gru[b["a"]][1], j1[b["a"]])
    return out


def ensure_stack(f):
    """{block start: (q, alive log-odds) at 25 fps}, on the CPU so that the numbers do not depend on a GPU."""
    path = os.path.join(OUT, "barath_stack_%s.pkl" % TAG)
    if os.path.isfile(path):
        return bench._load(path)
    import sys
    import torch
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_crf as ec
    import exp_stack as es_
    sys.argv = _argv
    torch.set_num_threads(4)
    ck = torch.load(CK or os.path.join(STACK_DIR, "stack_%s.pt" % TAG), map_location="cpu", weights_only=False)
    if TAG == "HC":
        net = ec.build_net(ck["d_in"])
    else:
        net = es_.build_net(ck["d_in"], e=ck.get("e", 0))
    net.load_state_dict(ck["state"])
    net.eval()
    a_all = model_v20.alive_evidence(f)
    chans = _channels()
    res = {}
    for b in bench.blocks("barath"):
        v, g, o = chans[b["a"]]
        a = a_all[b["a"]:b["b"] + 1]
        x = ec.pool(ec.frame_inputs(b, v, g, o, a)).astype(np.float32)
        if TAG == "HC":
            q, alive, _ = ec.score_half(net, x, len(v), "HC", "cpu")
        else:
            q, alive = es_.score_half(net, x, len(v), "cpu")
        res[b["a"]] = (q, alive)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def scores(f, probs_path):
    a = model_v20.alive_evidence(f)
    st = ensure_stack(f)
    lh = np.full(f["n"], np.log(0.5))
    la = np.full(f["n"], np.log(0.5))
    for s, (_, ph) in bench._load(probs_path).items():
        qs, alive = st[s]
        n = min(len(ph), len(qs))
        lq = np.log(np.clip(qs[:n], 1e-4, 1 - 1e-4) / np.clip(1 - qs[:n], 1e-4, 1 - 1e-4))
        ph = np.clip(ph[:n], 1e-4, 1 - 1e-4)
        lb = np.clip(np.log(ph / (1 - ph)), -6.0, 6.0)
        lq = np.clip(lq, -6.0, 6.0)
        mix = np.clip(1 / (1 + np.exp(-(W_Q * lq + (1 - W_Q) * lb))), 1e-4, 1 - 1e-4)
        lh[s:s + n] = np.log(mix)
        la[s:s + n] = np.log(1 - mix)
        a[s:s + n] = (1 - W_A) * a[s:s + n] + W_A * alive[:n]
    S = np.stack([a / 2 + lh, a / 2 + la, -a / 2], 1)
    S[atd.HALVES[1][1]:atd.KICKOFF_2ND, 2] += 1e6
    return S


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = scores(f, model_v20.ensure_probs())
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for _, a, e in atd.HALVES:
        labels[a:e + 1] = sp.decode(S[a:e + 1])
    return labels
