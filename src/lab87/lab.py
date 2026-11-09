"""Round 87 harness: decode and score arms over cached held-out outputs, no training. Open data, aggregate output only.
Every set is packed once: per half the truth of each reading, the dead-ball evidence, v20's smoothed blend per seed.
"""
import os
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
PACKS = os.path.join(REPO, "cache", "lab87")
THREADS = int(os.environ.get("LAB_THREADS", "20"))
SETS = ("dfl", "metrica", "sc", "st")
CTX = [("dfl", "evctrl"), ("dfl", "trk"), ("metrica", "ctrl"), ("sc", "evctrl"), ("sc", "phase"), ("sc", "pp"), ("st", "evctrl"), ("st", "native")]
_cache = {}


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def xs():
    """exp_stack and friends, imported only when something must be built."""
    os.chdir(REPO)
    import exp_stack
    return exp_stack


def pack(K):
    """Built once per set; the tracking is not kept."""
    path = os.path.join(PACKS, "pack_%s.pkl" % K)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    os.makedirs(PACKS, exist_ok=True)
    m = xs()
    from exp_stack import OUT, bench, ee, eg2, exp_smooth, ss
    if K == "st":
        import exp_st
        blocks = bench.blocks("st@v6")
        truth = {c: [np.asarray(g) for g in bench.labels(exp_st.SRC, c)] for c in exp_st.CONVS}
        ev = bench._load(os.path.join(OUT, "evidence_st.pkl"))["ev"]
        s50 = {s: bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s))["s50"] for s in (0, 1, 2)}
    else:
        import exp_deadmix as dm
        fsrc, src = ee.V6[K][0], ee.V3[K][1]
        blocks = bench.blocks(fsrc)
        truth = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in dm.CONVS[src]}
        ev = bench._load(os.path.join(OUT, "evidence_%s.pkl" % K))["ev"]
        s50 = {}
        for s in (0, 1, 2):
            v, g, o = eg2.parts(src, s, True)
            o = [ss.pk(np.asarray(z, np.float32), 0) for z in o]
            s50[s] = [np.asarray(z, np.float32) for z in exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * np.asarray(z, np.float64) for x, y, z in zip(v, g, o)], 25)]
    out = dict(groups=[{"group": b["group"], "seed": b.get("seed", 0)} for b in blocks], truth={c: [t.astype(np.int8) for t in v] for c, v in truth.items()}, ev=[np.asarray(e, np.float32) for e in ev], s50=s50)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    bench._blocks.clear()
    bench._labels.clear()
    return out


def get(K):
    if K not in _cache:
        _cache[K] = pack(K)
    return _cache[K]


def stacker(tag, s, K):
    """Held-out (q, alive) of one stacker, per half. SoccerTrack: the all-data model on its inputs (cached)."""
    path = os.path.join(PACKS, "stk_%s_%d_%s.pkl" % (tag, s, K))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    src = os.path.join(REPO, "cache", "bench", "crf", "sheld_%s_%d_%s.pkl" % (tag, s, K))
    if K != "st":
        with open(src, "rb") as fh:
            h = pickle.load(fh)
        out = ([np.asarray(z, np.float32) for z in h["q"]], [np.asarray(z, np.float32) for z in h["alive"]])
    else:
        m = xs()
        import torch
        from exp_stack import OUT, bench, ec
        d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s))
        ck = torch.load(os.path.join(OUT, "sall_%s_%d.pt" % (tag, s)), map_location="cpu", weights_only=False)
        net = m.make_net(tag, ck["d_in"], ck.get("e", 0))
        net.load_state_dict(ck["state"])
        net.eval()
        torch.set_num_threads(8)
        rows = []
        c = m.canon(tag)
        if c in ("Gk", "Gg", "Gj", "Gi", "G3"):
            pb = m.bf_mean("st@v3")
            ps = m.bs_mean("st@v6", [s] if c in ("Gk", "Gj", "G3") else [0, 1, 2], {"Gj": "J", "Gi": "J", "G3": "_t"}.get(c, ""))
            for r, a, b in zip(d["rows"], pb, ps):
                z = ec.pool(m._bs_cols(a, b).astype(np.float64)).astype(np.float32)
                n = min(len(r[1]), len(z))
                rows.append((r[0], np.hstack([r[1][:n], z[:n]]), None, r[3]))
        else:
            if c in ("Gy", "Gr"):
                pairs = m.bf_mean("st@v3", (0, 1, 2) if c == "Gy" else (3, 4, 5))
            elif c == "Go":
                pairs = m.bs_mean("st@v6", [0, 1, 2])
            elif c == "Gx":
                h = bench._load(os.path.join(REPO, "cache", "ballfree", "held_f_st-v3_%d.pkl" % s))
                pairs = list(zip(h["q"], h["alive"]))
            elif c == "Gf":
                h = bench._load(m._bf_path("st@v3", s))
                pairs = list(zip(h["q"], h["alive"]))
            elif c == "Gz":
                pairs = None
            else:
                raise SystemExit("no SoccerTrack input for " + tag)
            for r, p in zip(d["rows"], pairs if pairs is not None else [None] * len(d["rows"])):
                if p is None:
                    rows.append(r)
                    continue
                z = ec.pool(m._bf_cols(p[0], p[1]).astype(np.float64)).astype(np.float32)
                n = min(len(r[1]), len(z))
                rows.append((r[0], np.hstack([r[1][:n], z[:n]]), None, r[3]))
        zs = [m.score_half(net, r[1], r[3], "cpu", m.step_of(tag), tag) for r in rows]
        out = ([np.asarray(z[0], np.float32) for z in zs], [np.asarray(z[1], np.float32) for z in zs])
    os.makedirs(PACKS, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def _spec(pen, c, pen_dead=30.0, min_poss=8, min_dead=50, ramp_len=175):
    import decode2
    import exp_v7
    return decode2.duration_spec(pen, pen_dead, min_poss, min_dead, exp_v7.ramp(ramp_len, c))


def decode(a_list, q_list, pen=6.0, c=24.0, **kw):
    sp = _spec(pen, c, **kw)

    def one(i):
        q = np.clip(q_list[i], 1e-4, 1 - 1e-4)
        a = a_list[i]
        return sp.decode(np.stack([a / 2 + np.log(q), a / 2 + np.log(1 - q), -a / 2], 1)).astype(np.int8)
    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(one, range(len(a_list))))


def score(K, preds, convs=None):
    """{reading: {match: spellF1}}"""
    import exp_refine as er
    p = get(K)
    out = {}
    for c, tr in p["truth"].items():
        if convs is not None and c not in convs:
            continue
        ns = [min(len(t), len(x)) for t, x in zip(tr, preds)]
        out[c] = er.by_match(p["groups"], [t[:n] for t, n in zip(tr, ns)], [np.asarray(x)[:n] for x, n in zip(preds, ns)])
    return out


def mix(K, s, hq, alive, wq=0.5, wa=0.5):
    """The mix of v21 on one seed's blend: (alive evidence, q) per half, lengths aligned."""
    p = get(K)
    A, Q = [], []
    for e, b, q, x in zip(p["ev"], p["s50"][s], hq, alive):
        n = min(len(e), len(b), len(q), len(x))
        A.append((1 - wa) * np.asarray(e[:n], np.float64) + wa * np.asarray(x[:n], np.float64))
        Q.append(sg(wq * lg(q[:n]) + (1 - wq) * lg(b[:n])))
    return A, Q


def mean_paths(paths):
    """Logit mean of q, mean of the alive evidence over members [(A, Q)], per half."""
    h = len(paths[0][0])
    A, Q = [], []
    for i in range(h):
        n = min(len(p[0][i]) for p in paths)
        A.append(np.mean([p[0][i][:n] for p in paths], axis=0))
        Q.append(sg(np.mean([lg(p[1][i][:n]) for p in paths], axis=0)))
    return A, Q


def row(res):
    """One line of means per context, res: {K: {reading: {match: f1}}}."""
    return {(K, c): float(np.mean(list(res[K][c].values()))) for K, c in CTX if K in res and c in res[K]}


def fmt(vals, base=None):
    out = []
    for key in CTX:
        if key not in vals:
            continue
        if base is None:
            out.append("%s %s %.4f" % (key[0], key[1], vals[key]))
        else:
            out.append("%s %s %+.4f" % (key[0], key[1], vals[key] - base[key]))
    return " | ".join(out)


def setup():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(REPO)
