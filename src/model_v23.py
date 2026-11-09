"""v23 (rounds 94 and 96, docs/RAPOR.md): v22 with wider path models (width 128 instead of 64) as the source of the
decision's samples. Everything else is v22's.

    reference  as v22: v21's recipe on the seed mean of the width-64 stackers (models/stack_v1/), v21's decoder
    samples    from the three width-128 stackers (models/stack_v2/), same input, same training recipe and seeds,
               an equal share from each, 3,000 paths per half
    decision   as v22 (src/stack_decode.py): each possession is worth its mean chance of matching over the IoU
               thresholds - 0.5

Frames before the second kick-off (official FAQ) stay dead. Every model was trained on open data only; nothing here
reads a label of the match. Deterministic on the CPU (fixed seeds of numba's generator).
"""
import os

import numpy as np

import atd
import bench
import model_v7
import model_v13
import model_v20
import model_v21
import model_v22
import stack_decode as sd
import stack_model as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "v23")
WIDE = tuple(os.path.join(ROOT, "models", "stack_v2", "stack_w128_s%d.pt" % s) for s in (0, 1, 2))
DEFAULT = dict(model_v22.DEFAULT)


def ensure_wide(f):
    """Per half and seed: the width-128 stacker's emissions (T, 5) and moves (5, 5), on v22's inputs; on the CPU."""
    path = os.path.join(OUT, "wide.pkl")
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
        v10 = model_v22._v10(s)
        j1 = model_v22._j1(s)
        net = sm.load(WIDE[s])
        for b in bench.blocks("barath"):
            q_b, a_b = bfs[b["a"]]
            x = sm.inputs(b, v10[b["a"]], gru[b["a"]][1], j1[b["a"]], a_all[b["a"]:b["b"] + 1], q_b, a_b)
            with torch.no_grad():
                oh, oa = net(torch.as_tensor(x[None], dtype=torch.float32))
                d = 0.5 * (oh[..., 2] + oa[..., 2])
                e = torch.stack([oh[..., 0], oh[..., 1], oa[..., 0], oa[..., 1], d], -1)[0].numpy()
                A = sm.moves(net.t).numpy()
            res.setdefault(b["a"], []).append((e.astype(np.float32), A.astype(np.float32)))
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    a_all = model_v20.alive_evidence(f)
    mem = model_v22.ensure_members(f)
    wide = ensure_wide(f)
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for h, a, e in atd.HALVES:
        ms = mem[a]
        n = min(min(len(x[0]), len(x[1]), len(x[2])) for x in ms)
        n = min(n, e + 1 - a)
        lq = np.mean([model_v22.W_Q * sm.logit(x[1][:n]) + (1 - model_v22.W_Q) * sm.logit(x[0][:n]) for x in ms], axis=0)
        q = np.clip(1 / (1 + np.exp(-lq)), 1e-4, 1 - 1e-4)
        al = (1 - model_v22.W_A) * a_all[a:a + n] + model_v22.W_A * np.mean([x[2][:n] for x in ms], axis=0)
        S = np.stack([al / 2 + np.log(q), al / 2 + np.log(1 - q), -al / 2], 1)
        if a == atd.HALVES[1][1]:
            S[:atd.KICKOFF_2ND - a, 2] += 1e6
        ref = sp.decode(S).astype(np.int8)
        out = sd.decide(list(wide[a]), ref, p["lam"], int(p["samples"]), int(p["seed"]) + 1000 * h)
        if p.get("keep_dead"):
            out = sd.keep_reference_dead(out, ref)
        if a == atd.HALVES[1][1]:
            out[:atd.KICKOFF_2ND - a] = 2
        labels[a:a + len(out)] = out
    return labels
