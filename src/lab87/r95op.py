"""Round 95: the target-condition yardstick (OP). On the target every component works out of provider: v10, J1, the BiGRU,
the dead-ball models and the stacker never saw BAR-ATH's provider. In the lab, DFL's and Metrica's channels were trained
on the provider's other matches. Here each provider is read with components that never saw it:

  Metrica-OP  v10 trained on DFL only; J1 -> D1 (C1 fine-tuned on DFL only) on the de-glued input; the shared BiGRU (never
              saw Metrica); BF (PFF only); dead-ball evidence and stacker without Metrica
  DFL-OP      v10 trained on Metrica only; J1 -> H1 (C1 fine-tuned on de-glued Metrica only); the BiGRU stays DFL's match
              hold-out (no DFL-free BiGRU exists: the one in-provider part left); BF; dead-ball evidence and stacker without DFL

"LP" = the same provider-held-out stacker and dead-ball evidence with the lab's in-provider channels (round 88's hold-out).
Open data only, aggregate output.

    python -I r95op.py prep            v10 without the provider (3 LightGBM seeds), D1 on metrica@v3g
    python -I r95op.py eval TAG [M]    v21's recipe and the decision on LP and OP rows (stacker TAG's provider-held-out nets)
"""
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402

OUTD = os.path.join(lab.REPO, "cache", "lab87", "foldnets")
OPD = os.path.join(lab.REPO, "cache", "lab87", "op")
SEEDS = (20261102, 20261103, 20261104)


def v10_op(seed):
    """v10's recipe trained on one provider: Metrica only (read on DFL), DFL only (read on de-glued Metrica)."""
    path = os.path.join(OPD, "v10op_s%d.pkl" % seed)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import lightgbm as lgb
    import bench
    import exp_deglue as eg
    params = dict(bench.LGB_TEAM, num_threads=bench.THREADS, seed=seed)

    def rows(feats, labs, w):
        Xt, yt, _ = bench.team_rows(feats, labs)
        return Xt, yt, np.full(len(yt), w, np.float32)

    def fit(parts):
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        w = np.concatenate([p[2] for p in parts])
        return lgb.train(params, lgb.Dataset(X, y, weight=w), bench.ROUNDS)
    t0 = time.time()
    fd, fm, fg = eg._aug("dfl@v3"), eg._aug("metrica@v3"), eg._aug(eg.G)
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    m_only = fit([rows(fm, lm, eg.W_M / 3.0)])           # the shipped v10 learned Metrica on the glued input
    d_only = fit([rows(fd, ld, 1.0)])
    out = {"dfl": [np.asarray(bench.predict_team(m_only, f), np.float32) for f in fd],
           "metrica": [np.asarray(bench.predict_team(d_only, f), np.float32) for f in fg]}
    os.makedirs(OPD, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print("  v10 without the provider, seed %d: %.0f s" % (seed, time.time() - t0), flush=True)
    return out


def prep():
    lab.setup()
    import short_specialist as ss
    for s in (0, 1, 2):
        v10_op(s)
    if not os.path.isfile(os.path.join(ss.OUT, "D1_metrica-v3g.pkl")):
        ss.infer("metrica@v3g", "D1")
    print("prep done", flush=True)


def op_parts(K, s):
    """(v10, BiGRU, J1-like) per half for provider K with components that never saw it."""
    xs = lab.xs()
    import exp_kpu
    src = xs.ee.V3[K][1]
    _, g, _ = xs.eg2.parts(src, s, True)
    v = v10_op(s)[K]
    o = exp_kpu.outputs("H1", "dfl@v3")[SEEDS[s]] if K == "dfl" else exp_kpu.outputs("D1", "metrica@v3g")[SEEDS[s]]
    return v, g, o


def build(K, s, op):
    """Stacker rows (Hy's input) and v20's smoothed blend per half, with the dead-ball evidence held out of provider K."""
    xs = lab.xs()
    import bench
    import exp_smooth
    import short_specialist as ss
    from exp_stack import ec
    blocks = bench.blocks(xs.ee.V6[K][0])
    ev = xs.lopo_evidence(K)
    v, g, o = op_parts(K, s) if op else xs.eg2.parts(xs.ee.V3[K][1], s, True)
    bf = xs.extras("Gy", K, s)
    rows, s50, nfr = [], [], []
    for i, b in enumerate(blocks):
        x = ec.pool(ec.frame_inputs(b, v[i], g[i], o[i], ev[i]))
        z = ec.pool(np.asarray(bf[i], np.float64).reshape(len(bf[i]), -1)).astype(np.float32)
        n = min(len(x), len(z))
        rows.append(np.hstack([x[:n], z[:n]]).astype(np.float32))
        oo = np.asarray(o[i], np.float32)
        m = min(len(v[i]), len(g[i]), len(oo))
        bl = 0.25 * np.asarray(v[i][:m], np.float64) + 0.25 * np.asarray(g[i][:m], np.float64) + 0.5 * np.asarray(ss.pk(oo[:m], 0), np.float64)
        s50.append(np.asarray(exp_smooth.smooth([bl], 25)[0], np.float32))
        nfr.append(min(len(b["P"]), len(ev[i])))
    bench._blocks.clear()
    return rows, s50, nfr, [np.asarray(a, np.float32) for a in ev]


def load_net(tag, s, K):
    import torch
    xs = lab.xs()
    ck = torch.load(os.path.join(OUTD, "lnet_%s_%d_%s.pt" % (tag, s, K)), map_location="cpu", weights_only=False)
    net = xs.make_net(tag, ck["d_in"], ck["e"])
    net.load_state_dict(ck["state"])
    return net.eval()


def evaluate(tag, M):
    lab.setup()
    import torch
    import stack_decode as sd
    import foldnets as fn
    xs = lab.xs()
    fn._opts(xs)
    torch.set_num_threads(8)
    t0 = time.time()
    res = {}
    for K in ("dfl", "metrica"):
        nets = {s: load_net(tag, s, K) for s in (0, 1, 2)}
        for kind in ("LP", "OP"):
            E, mixes, singles = {}, {}, {}
            for s in (0, 1, 2):
                rows, s50, nfr, ev = build(K, s, kind == "OP")
                E[s] = [fn.emit_half(nets[s], x) for x in rows]
                qa = [xs.score_half(nets[s], x, n, "cpu", xs.step_of(tag), tag) for x, n in zip(rows, nfr)]
                A, Q = [], []
                for e_, b_, (q_, a_) in zip(ev, s50, qa):
                    n = min(len(e_), len(b_), len(q_), len(a_))
                    A.append(0.5 * np.asarray(e_[:n], np.float64) + 0.5 * np.asarray(a_[:n], np.float64))
                    Q.append(lab.sg(0.5 * lab.lg(q_[:n]) + 0.5 * lab.lg(b_[:n])))
                mixes[s] = (A, Q)
                singles[s] = [np.asarray(x) for x in lab.decode(A, Q)]
                if kind == "LP" and tag == "Hy":
                    with open(os.path.join(OUTD, "emitlopo_Hy_%d_%s.pkl" % (s, K)), "rb") as fh:
                        old = pickle.load(fh)
                    d = max(float(np.abs(a[0][:len(b[0])] - b[0][:len(a[0])]).max()) for a, b in zip(E[s], old))
                    print("  check %s seed %d: max |emission - round 88's hold-out| %.2e" % (K, s, d), flush=True)
            bagA, bagQ = lab.mean_paths([mixes[s] for s in (0, 1, 2)])
            ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
            with ThreadPoolExecutor(8) as pool:
                dec = list(pool.map(lambda i: sd.decide([E[s][i] for s in (0, 1, 2)], ref[i], 0.5, M, 890000 + i), range(len(ref))))
            res[(K, kind)] = {"v21": [lab.score(K, singles[s]) for s in (0, 1, 2)], "v22": lab.score(K, dec)}
            print("%s %s done %.0f s" % (K, kind, time.time() - t0), flush=True)
    with open(os.path.join(OPD, "r95_%s.pkl" % tag), "wb") as fh:
        pickle.dump(res, fh)
    for K in ("dfl", "metrica"):
        for kind in ("LP", "OP"):
            r = res[(K, kind)]
            line = []
            for c in r["v22"]:
                a = float(np.mean([np.mean(list(x[c].values())) for x in r["v21"]]))
                b = float(np.mean(list(r["v22"][c].values())))
                line.append("%s v21 %.4f v22 %.4f (%+.4f)" % (c, a, b, b - a))
            print("%-8s %s: %s" % (K, kind, " | ".join(line)), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


def fold_nets(tag, s):
    """The stacker's match-held-out nets of one seed: [(net, held-out groups)]."""
    import torch
    xs = lab.xs()
    out = []
    for name in sorted(os.listdir(OUTD)):
        if name.startswith("%s_%d_" % (tag, s)) and name.endswith(".pt") and not name.endswith("_all.pt"):
            ck = torch.load(os.path.join(OUTD, name), map_location="cpu", weights_only=False)
            net = xs.make_net(tag, ck["d_in"], ck["e"])
            net.load_state_dict(ck["state"])
            out.append((net.eval(), set(ck["hold"])))
    return out


def evaluate_folds(tag, M):
    """The stacker that knows the provider's convention (match held out, as in the lab) on rows whose channels and / or
    dead-ball evidence never saw the provider: OPs = OP channels + OP dead-ball evidence; OPc = OP channels + the lab's
    evidence; OPd = the lab's channels + OP evidence. Emissions and marginals averaged over the nets that held the half out."""
    lab.setup()
    import torch
    import stack_decode as sd
    import foldnets as fn
    xs = lab.xs()
    fn._opts(xs)
    torch.set_num_threads(8)
    t0 = time.time()
    res = {}
    for K in ("dfl", "metrica"):
        P = lab.get(K)
        groups = [str(g["group"]) for g in P["groups"]]
        for kind in ("OPs", "OPc", "OPd"):
            E, mixes, singles = {}, {}, {}
            for s in (0, 1, 2):
                nets = fold_nets(tag, s)
                if kind == "OPs":
                    rows, s50, nfr, ev = build(K, s, True)
                elif kind == "OPc":
                    rows, s50, nfr, _ = build(K, s, True)
                    ev = [np.asarray(a, np.float32) for a in P["ev"]]
                    rows = [r.copy() for r in rows]
                    from exp_stack import ec
                    for r, a in zip(rows, ev):
                        col = ec.pool(np.clip(np.asarray(a, np.float64), -20, 20)[:, None] / 5.0).astype(np.float32)
                        n = min(len(r), len(col))
                        r[:n, 2 * ec.N_OWN] = col[:n, 0]
                else:
                    rows, _, nfr, ev = build(K, s, False)
                    s50 = P["s50"][s]
                E[s], qas = [], []
                for i, x in enumerate(rows):
                    hold = [net for net, h in nets if groups[i] in h]
                    assert hold, groups[i]
                    em = [fn.emit_half(net, x) for net in hold]
                    E[s].append((np.mean([a for a, _ in em], axis=0).astype(np.float32), np.mean([b for _, b in em], axis=0).astype(np.float32)))
                    qa = [xs.score_half(net, x, nfr[i], "cpu", xs.step_of(tag), tag) for net in hold]
                    n = min(len(q) for q, _ in qa)
                    qas.append((lab.sg(np.mean([lab.lg(np.asarray(q[:n], np.float64)) for q, _ in qa], axis=0)), np.mean([np.asarray(a[:n], np.float64) for _, a in qa], axis=0)))
                A, Q = [], []
                for e_, b_, (q_, a_) in zip(ev, s50, qas):
                    n = min(len(e_), len(b_), len(q_), len(a_))
                    A.append(0.5 * np.asarray(e_[:n], np.float64) + 0.5 * np.asarray(a_[:n], np.float64))
                    Q.append(lab.sg(0.5 * lab.lg(q_[:n]) + 0.5 * lab.lg(b_[:n])))
                mixes[s] = (A, Q)
                singles[s] = [np.asarray(x) for x in lab.decode(A, Q)]
            bagA, bagQ = lab.mean_paths([mixes[s] for s in (0, 1, 2)])
            ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
            with ThreadPoolExecutor(8) as pool:
                dec = list(pool.map(lambda i: sd.decide([E[s][i] for s in (0, 1, 2)], ref[i], 0.5, M, 890000 + i), range(len(ref))))
            res[(K, kind)] = {"v21": [lab.score(K, singles[s]) for s in (0, 1, 2)], "v22": lab.score(K, dec)}
            print("%s %s done %.0f s" % (K, kind, time.time() - t0), flush=True)
    with open(os.path.join(OPD, "r95f_%s.pkl" % tag), "wb") as fh:
        pickle.dump(res, fh)
    for (K, kind), r in res.items():
        line = []
        for c in r["v22"]:
            a = float(np.mean([np.mean(list(x[c].values())) for x in r["v21"]]))
            b = float(np.mean(list(r["v22"][c].values())))
            line.append("%s v21 %.4f v22 %.4f (%+.4f)" % (c, a, b, b - a))
        print("%-8s %s: %s" % (K, kind, " | ".join(line)), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    {"prep": lambda: prep(), "eval": lambda: evaluate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000),
     "folds": lambda: evaluate_folds(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000)}[sys.argv[1]]()
