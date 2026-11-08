"""v22 (rounds 87-88, docs/RAPOR.md): v21's channels and stacker over three seeds, and the labelling chosen by the metric's
Bayes decision on the stackers' joint posterior (src/stack_decode.py) instead of a Viterbi path over marginals.

    seeds      three of everything that was trained with a seed: v10 (LightGBM), J1, the stacker; the shared BiGRU
               (mean of its three members) and the ball-free channel (mean of three seeds) as in v21
    reference  v21's recipe on the seed mean: team = logit mean over seeds of 0.5 x stacker + 0.5 x v20's smoothed
               blend; alive = 0.5 x v20's evidence + 0.5 x the stackers' mean alive log-odds; v21's decoder
    decision   samples from each stacker's path model, an equal share each; candidates = their segments and the
               reference's; each possession is worth its mean chance of matching over the IoU thresholds - lam

Frames before the second kick-off (official FAQ) stay dead. Every model was trained on open data only; nothing here
reads a label of the match. Deterministic on the CPU (fixed seeds of numba's generator).
"""
import os

os.environ["BF_WIN"] = "300"
os.environ.pop("BF_TAG", None)

import numpy as np  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import bfree  # noqa: E402
import model_v7  # noqa: E402
import model_v9  # noqa: E402
import model_v10  # noqa: E402
import model_v13  # noqa: E402
import model_v15  # noqa: E402
import model_v17  # noqa: E402
import model_v20  # noqa: E402
import model_v21  # noqa: E402
import short_specialist as ss  # noqa: E402
import stack_decode as sd  # noqa: E402
import stack_model as sm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "v22")
MODELS = os.path.join(ROOT, "models")
V10 = (os.path.join(MODELS, "v10_team.txt"), os.path.join(MODELS, "v10_seeds", "v10_team_s1.txt"), os.path.join(MODELS, "v10_seeds", "v10_team_s2.txt"))
J1 = tuple(os.path.join(MODELS, "kpu_j1", "kpu_J1_s%d.pt" % s) for s in (20261102, 20261103, 20261104))
STACKS = tuple(os.path.join(MODELS, "stack_v1", "stack_s%d.pt" % s) for s in (0, 1, 2))
DEFAULT = dict(model_v21.DEFAULT)
DEFAULT.update(lam=0.5, samples=3000, seed=20261108)
W_Q, W_A = model_v21.W_Q, model_v21.W_A


def _j1(s):
    """J1 of seed index s on BAR-ATH's halves, {block start: (T, 9) float16}."""
    if s == 0:
        return model_v15.ensure_c1(cfg=model_v17.CONFIG)
    path = os.path.join(OUT, "j1_s%d.pkl" % s)
    if os.path.isfile(path):
        return bench._load(path)
    model = ss.load(J1[s])
    bench._barath_inputs()
    out = {b["a"]: ss.predict(model, ss.features(b["P"], b["B"], b["team"], b["gk"], b["pitch"])).astype(np.float16) for b in bench.blocks("barath")}
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def _v10(s):
    """v10's raw team probability of seed index s, {block start: array}."""
    path = os.path.join(OUT, "v10_s%d.pkl" % s)
    if os.path.isfile(path):
        return bench._load(path)
    model = model_v10.team_model() if s == 0 else bench.load_booster(V10[s])
    bench._barath_inputs()
    out = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        out[b["a"]] = np.asarray(bench.predict_team(model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))), np.float64)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def ensure_members(f):
    """Per seed and half: (v20-style smoothed blend, stacker q, stacker alive, emissions, moves); on the CPU."""
    path = os.path.join(OUT, "members.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    import torch
    torch.set_num_threads(4)
    a_all = model_v20.alive_evidence(f)
    gru = bench._load(model_v13.ensure_gru())
    bfs = model_v21.ensure_bf()
    bench._barath_inputs()
    res = {}
    for s in range(3):
        v10 = _v10(s)
        j1 = _j1(s)
        net = sm.load(STACKS[s])
        for b in bench.blocks("barath"):
            v, g, o = v10[b["a"]], gru[b["a"]][1], j1[b["a"]]
            blend = model_v9.smooth((1 - model_v20.W_J1) * ((1 - model_v15.W_BIGRU) * v + model_v15.W_BIGRU * np.asarray(g, np.float64)) + model_v20.W_J1 * ss.pk(o, 0))
            q_b, a_b = bfs[b["a"]]
            x = sm.inputs(b, v, g, o, a_all[b["a"]:b["b"] + 1], q_b, a_b)
            q, alive = sm.score_half(net, x, len(v))
            with torch.no_grad():
                oh, oa = net(torch.as_tensor(x[None], dtype=torch.float32))
                d = 0.5 * (oh[..., 2] + oa[..., 2])
                e = torch.stack([oh[..., 0], oh[..., 1], oa[..., 0], oa[..., 1], d], -1)[0].numpy()
                A = sm.moves(net.t).numpy()
            res.setdefault(b["a"], []).append((np.asarray(blend, np.float32), q, alive, e.astype(np.float32), A.astype(np.float32)))
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    a_all = model_v20.alive_evidence(f)
    mem = ensure_members(f)
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for h, a, e in atd.HALVES:
        ms = mem[a]
        n = min(min(len(x[0]), len(x[1]), len(x[2])) for x in ms)
        n = min(n, e + 1 - a)
        lq = np.mean([W_Q * sm.logit(x[1][:n]) + (1 - W_Q) * sm.logit(x[0][:n]) for x in ms], axis=0)
        q = np.clip(1 / (1 + np.exp(-lq)), 1e-4, 1 - 1e-4)
        al = (1 - W_A) * a_all[a:a + n] + W_A * np.mean([x[2][:n] for x in ms], axis=0)
        S = np.stack([al / 2 + np.log(q), al / 2 + np.log(1 - q), -al / 2], 1)
        if a == atd.HALVES[1][1]:
            S[:atd.KICKOFF_2ND - a, 2] += 1e6
        ref = sp.decode(S).astype(np.int8)
        out = sd.decide([(x[3], x[4]) for x in ms], ref, p["lam"], int(p["samples"]), int(p["seed"]) + 1000 * h)
        if p.get("keep_dead"):
            out = sd.keep_reference_dead(out, ref)
        if a == atd.HALVES[1][1]:
            out[:atd.KICKOFF_2ND - a] = 2
        labels[a:a + len(out)] = out
    return labels
