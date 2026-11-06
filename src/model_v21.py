"""v21 (rounds 61, 65-69, docs/RAPOR.md): v20's channels and dead-ball evidence, a fourth channel that never sees the ball, and a stacker on top of them.

Ball-free channel (src/exp_ballfree.py, models/ballfree_v1/): every player of both teams as a set per step, 5 Hz, a bidirectional GRU over a minute; trained on open PFF data only (controlled reading, sim.V3). Mean of three seeds. It reads no ball at all, so its errors are its own.

Stacker (src/stack_model.py, models/stack_v1/): reads the channels' outputs and gives a path model's posterior; trained on the channels' held-out outputs on the open games only.

    team     logit mix 0.5 x stacker + 0.5 x v20's smoothed blend
    alive    0.5 x v20's evidence + 0.5 x the stacker's alive log-odds
    decoder  pen_team 6, B2 c 24; the stacker's fewer false flips allow the lower penalties; the rest as v20

Frames before the second kick-off (official FAQ) are dead. Cached: cache/v21/barath_bf.pkl, cache/v21/barath_stack.pkl; deterministic on CPU. Nothing here reads a label of the match.
"""
import os

os.environ["BF_WIN"] = "300"       # the ball-free channel's window at its full settings
os.environ.pop("BF_TAG", None)

import numpy as np  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import bfree  # noqa: E402
import exp_ballfree as bf  # noqa: E402
import model_v7  # noqa: E402
import model_v10  # noqa: E402
import model_v13  # noqa: E402
import model_v15  # noqa: E402
import model_v17  # noqa: E402
import model_v20  # noqa: E402
import stack_model as sm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "v21")
BF_DIR = os.path.join(ROOT, "models", "ballfree_v1")
BF_SEEDS = (20261106, 20261107, 20261108)
STACK = os.path.join(ROOT, "models", "stack_v1", "stack_s0.pt")
DEFAULT = dict(model_v20.DEFAULT)
DEFAULT.update(pen_team=6.0, b2_c=24.0)
W_Q, W_A = 0.5, 0.5


def ensure_bf():
    """{block start: (q, alive log-odds)}: the mean of the three ball-free seeds, on the CPU."""
    path = os.path.join(OUT, "barath_bf.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    import torch
    torch.set_num_threads(4)
    nets = []
    for seed in BF_SEEDS:
        net = bf.build_net()
        net.load_state_dict(torch.load(os.path.join(BF_DIR, "bf_s%d.pt" % seed), map_location="cpu", weights_only=False)["state"])
        nets.append(net.eval())
    bench._barath_inputs()
    out = {}
    for b in bench.blocks("barath"):
        tok, _, signs = bf.tokens(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        res = [bf.predict(net, tok, signs, len(b["P"]), "cpu") for net in nets]
        lq = np.mean([sm.logit(z[0]) for z in res], axis=0)
        out[b["a"]] = ((1 / (1 + np.exp(-lq))).astype(np.float32), np.mean([z[1] for z in res], axis=0).astype(np.float32))
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def _channels():
    """Per half {block start: (v10 raw, BiGRU, J1's outputs)} exactly as v20 computes them."""
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
    path = os.path.join(OUT, "barath_stack.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    import torch
    torch.set_num_threads(4)
    net = sm.load(STACK)
    a_all = model_v20.alive_evidence(f)
    chans = _channels()
    bfs = ensure_bf()
    res = {}
    for b in bench.blocks("barath"):
        v, g, o = chans[b["a"]]
        q_b, a_b = bfs[b["a"]]
        x = sm.inputs(b, v, g, o, a_all[b["a"]:b["b"] + 1], q_b, a_b)
        res[b["a"]] = sm.score_half(net, x, len(v))
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
        mix = np.clip(1 / (1 + np.exp(-(W_Q * sm.logit(qs[:n]) + (1 - W_Q) * sm.logit(ph[:n])))), 1e-4, 1 - 1e-4)
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
