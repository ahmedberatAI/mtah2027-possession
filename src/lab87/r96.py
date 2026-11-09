"""Round 96 (pre-registered in docs/RAPOR.md): teach the stacker the channels' out-of-provider behaviour (Hop) and read the
candidates where the target sits (OPs: the stacker knows the convention, channels and dead-ball models never saw the
provider). Open data only, aggregate output.

    python -I r96.py rows                 OPs rows per provider and seed (cache/lab87/op/ops_<set>_<seed>.pkl)
    python -I r96.py train SEEDS          Hop's match-held-out nets (+ the all-data net), e.g. 012
    python -I r96.py lnets SEEDS          Hop's provider-held-out nets
    python -I r96.py eval                 v22 / W3 / Hop in the lab and in OPs, SoccerTrack, the rule; Hop's hold-out
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
KS = ("dfl", "metrica", "sc")


def ops_rows(K, s):
    """[(group, x, y, frames)] of provider K with out-of-provider channels (DFL, Metrica) and held-out dead-ball evidence,
    plus v20's blend, frame counts and the evidence at 25 fps."""
    path = os.path.join(OPD, "ops_%s_%d.pkl" % (K, s))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import r95op
    xs = lab.xs()
    ip = xs.load_rows(s, "Hy")[K]
    rows, s50, nfr, ev = r95op.build(K, s, K != "sc")
    out = []
    for r, x in zip(ip, rows):
        L = min(len(x), len(r[2]))
        out.append((r[0], x[:L], r[2][:L], r[3]))
    d = dict(rows=out, s50=s50, nfr=nfr, ev=ev)
    os.makedirs(OPD, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh)
    return d


def _data(s):
    xs = lab.xs()
    ip = xs.load_rows(s, "Hy")
    return {K: list(ip[K]) + list(ops_rows(K, s)["rows"]) for K in KS}


def _job(job):
    import torch
    xs = lab.xs()
    s, name, hold, seed = job
    path = os.path.join(OUTD, name)
    if os.path.isfile(path):
        return path
    torch.set_num_threads(1)
    t0 = time.time()
    data = _data(s)
    e = xs.e_of("Hy")
    net, tail = xs.fit(data, hold, seed, xs.ec.pick_device(), e=e, step_=xs.step_of("Hy"), tag="Hy")
    d_in = next(iter(data.values()))[0][1].shape[1] - xs.ec.N_OWN - e
    torch.save(dict(state={a: v.cpu() for a, v in net.state_dict().items()}, d_in=d_in, e=e, hold=sorted(str(h) for h in hold)), path)
    print("  %s: loss %.4f, %.0f s" % (name, tail, time.time() - t0), flush=True)
    return path


def _pool(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_job, jobs))


def train(seeds, workers):
    """Hy's folds and seeds (foldnets.train), training rows doubled with their OPs copies; a fold holds both copies out."""
    xs = lab.xs()
    jobs = []
    for s in seeds:
        data = xs.load_rows(s, "Hy")
        dfl = sorted({r[0] for r in data["dfl"]}, key=str)
        met = sorted({r[0] for r in data["metrica"]}, key=str)
        sc = sorted({r[0] for r in data["sc"]}, key=str)
        base = 20261107 + 100 * s
        jobs += [(s, "Hop_%d_%d.pt" % (s, i), {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)]), base + i) for i in range(len(dfl))]
        jobs.append((s, "Hop_%d_all.pt" % s, set(), base + 50))
    _pool(jobs, workers)
    print("trained", flush=True)


def lnets(seeds, workers):
    xs = lab.xs()
    jobs = []
    for s in seeds:
        data = xs.load_rows(s, "Hy")
        for j, K in enumerate(xs.SETS):
            jobs.append((s, "lnet_Hop_%d_%s.pt" % (s, K), {r[0] for r in data[K]}, 20261207 + 100 * s + j))
    _pool(jobs, workers)
    print("lnets done", flush=True)


def _load(path):
    import torch
    xs = lab.xs()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = xs.build_path(ck["d_in"], width=int(ck["state"]["front.0.weight"].shape[0]), e=ck["e"])      # Hyw's are 128 wide
    net.load_state_dict(ck["state"])
    return net.eval(), set(ck["hold"])


def folds(tag, s):
    return [_load(os.path.join(OUTD, n)) for n in sorted(os.listdir(OUTD)) if n.startswith("%s_%d_" % (tag, s)) and n.endswith(".pt") and not n.endswith("_all.pt")]


def read(nets, rows, nfr, want_q):
    """Per half: emissions and moves averaged over the nets that held it out; optionally (q, alive) averaged in logit."""
    import foldnets as fn
    xs = lab.xs()
    E, QA = [], []
    for (g, x, _, _), n in zip(rows, nfr):
        hold = [net for net, h in nets if str(g) in h]
        assert hold, g
        em = [fn.emit_half(net, x) for net in hold]
        E.append((np.mean([a for a, _ in em], axis=0).astype(np.float32), np.mean([b for _, b in em], axis=0).astype(np.float32)))
        if want_q:
            qa = [xs.score_half(net, x, n, "cpu", xs.step_of("Hy"), "Hy") for net in hold]
            m = min(len(q) for q, _ in qa)
            QA.append((lab.sg(np.mean([lab.lg(np.asarray(q[:m], np.float64)) for q, _ in qa], axis=0)), np.mean([np.asarray(a[:m], np.float64) for _, a in qa], axis=0)))
    return E, QA


def ops_reference(K):
    """v21's recipe in OPs: Hy's held-out marginals on the OPs rows, the OPs blend and evidence; bag of three seeds."""
    path = os.path.join(OPD, "ref_ops_%s.pkl" % K)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    mixes = {}
    for s in (0, 1, 2):
        d = ops_rows(K, s)
        _, QA = read(folds("Hy", s), d["rows"], d["nfr"], True)
        A, Q = [], []
        for e_, b_, (q_, a_) in zip(d["ev"], d["s50"], QA):
            n = min(len(e_), len(b_), len(q_), len(a_))
            A.append(0.5 * np.asarray(e_[:n], np.float64) + 0.5 * np.asarray(a_[:n], np.float64))
            Q.append(lab.sg(0.5 * lab.lg(q_[:n]) + 0.5 * lab.lg(b_[:n])))
        mixes[s] = (A, Q)
    bagA, bagQ = lab.mean_paths([mixes[s] for s in (0, 1, 2)])
    ref = [np.asarray(x, np.int8) for x in lab.decode(bagA, bagQ)]
    with open(path, "wb") as fh:
        pickle.dump(ref, fh)
    return ref


def members(tag, K, cond):
    """{seed: [(e, A) per half]} of a stacker family in a condition."""
    from r88 import emit
    out = {}
    for s in (0, 1, 2):
        if cond == "IP" and tag in ("Hy", "Hyw"):
            out[s] = emit(K, s, tag)
        elif cond == "IP":
            xs = lab.xs()
            rows = xs.load_rows(s, "Hy")[K]
            out[s] = read(folds(tag, s), rows, [r[3] for r in rows], False)[0]
        else:
            d = ops_rows(K, s)
            out[s] = read(folds(tag, s), d["rows"], d["nfr"], False)[0]
    return out


def st_members(tag):
    """SoccerTrack (never seen by any component): the all-data net on its rows, as foldnets.emit."""
    from r88 import emit
    if tag in ("Hy", "Hyw"):
        return {s: emit("st", s, tag) for s in (0, 1, 2)}
    import foldnets as fn
    xs = lab.xs()
    from exp_stack import OUT, bench, ec
    out = {}
    pb = xs.bf_mean("st@v3", (0, 1, 2))
    for s in (0, 1, 2):
        net, _ = _load(os.path.join(OUTD, "%s_%d_all.pt" % (tag, s)))
        d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s))
        res = []
        for r, (q_, a_) in zip(d["rows"], pb):
            z = ec.pool(xs._bf_cols(q_, a_).astype(np.float64)).astype(np.float32)
            n_ = min(len(r[1]), len(z))
            res.append(fn.emit_half(net, np.hstack([r[1][:n_], z[:n_]])))
        out[s] = res
    return out


def evaluate():
    lab.setup()
    import torch
    import stack_decode as sd
    from r93 import v22_decisions
    torch.set_num_threads(8)
    t0 = time.time()
    res = {}
    cells = {}
    for K in KS + ("st",):
        conds = ("IP",) if K == "st" else ("IP", "OPs")
        for cond in conds:
            if cond == "IP":
                v = v22_decisions(K)
                ref, dec22 = v["ref"], v["dec"]
            else:
                ref = ops_reference(K)
                dec22 = None
            arms = {}
            for tag in ("Hy", "Hyw", "Hop"):
                if tag == "Hy" and dec22 is not None:
                    arms["v22"] = dec22
                    continue
                M_ = st_members(tag) if K == "st" else members(tag, K, cond)
                with ThreadPoolExecutor(8) as pool:
                    arms[{"Hy": "v22", "Hyw": "W3", "Hop": "Hop"}[tag]] = list(pool.map(lambda i: sd.decide([M_[s][i] for s in (0, 1, 2)], ref[i], 0.5, 3000, 890000 + i), range(len(ref))))
            for name, d in arms.items():
                cells[(K, cond, name)] = lab.score(K, d)
            print("%s %s done %.0f s" % (K, cond, time.time() - t0), flush=True)
    with open(os.path.join(OPD, "r96_cells.pkl"), "wb") as fh:
        pickle.dump(cells, fh)
    for (K, cond, name), sc in sorted(cells.items()):
        if name == "v22":
            continue
        base = cells[(K, cond, "v22")]
        print("%-3s vs v22, %-7s %-3s: %s" % (name, K, cond, " | ".join("%s %+.4f" % (c, np.mean(list(sc[c].values())) - np.mean(list(base[c].values()))) for c in sc)), flush=True)
    for K in KS:
        for cond in ("IP", "OPs"):
            b = cells[(K, cond, "v22")]
            print("v22 level %-7s %-3s: %s" % (K, cond, " | ".join("%s %.4f" % (c, np.mean(list(b[c].values()))) for c in b)), flush=True)
    lab.xs()
    import exp_blend_weight as bw
    src = {"dfl": "dfl@v3", "metrica": "metrica@v3", "sc": "sc@v3"}
    srco = {"dfl": "dfl@ops", "metrica": "metrica@ops", "sc": "sc@ops"}
    for name in ("W3", "Hop"):
        print("-- %s against v22, the lab only (the old rule's primary)" % name, flush=True)
        t = {}
        for K in KS:
            for c in cells[(K, "IP", "v22")]:
                t[(src[K], c, "edges D", "v22")] = dict(cells[(K, "IP", "v22")][c])
                t[(src[K], c, "edges D", name)] = dict(cells[(K, "IP", name)][c])
        bw.decide(t, "v22", name)
        print("-- %s against v22, the lab and OPs (round 96's primary)" % name, flush=True)
        prim, guards = set(bw.PRIMARY), set(bw.GUARDS)
        bw.PRIMARY |= {("dfl@ops", "evctrl"), ("metrica@ops", "ctrl")}
        bw.GUARDS |= {("dfl@ops", "trk"), ("sc@ops", "phase"), ("sc@ops", "pp")}
        for K in KS:
            for c in cells[(K, "OPs", "v22")]:
                t[(srco[K], c, "edges D", "v22")] = dict(cells[(K, "OPs", "v22")][c])
                t[(srco[K], c, "edges D", name)] = dict(cells[(K, "OPs", name)][c])
        bw.decide(t, "v22", name)
        bw.PRIMARY.clear()
        bw.PRIMARY.update(prim)
        bw.GUARDS.clear()
        bw.GUARDS.update(guards)
    print("done %.0f s" % (time.time() - t0), flush=True)


def lopo_hop():
    """Information: Hop's provider-held-out nets on the lab's hold-out rows (in-provider channels, held-out dead-ball
    evidence), two-seed mixture against Hy's in the same form (r94.lopo's form)."""
    lab.setup()
    import stack_decode as sd
    import r95op
    t0 = time.time()
    seeds = (0, 1)
    out = {}
    for K in ("dfl", "metrica", "sc"):
        P = lab.get(K)
        Es = {"Hy": {}, "Hop": {}}
        mixes = {}
        for s in seeds:
            rows, s50, nfr, ev = r95op.build(K, s, False)
            import foldnets as fn
            for tag in ("Hy", "Hop"):
                net, _ = _load(os.path.join(OUTD, "lnet_%s_%d_%s.pt" % (tag, s, K)))
                Es[tag][s] = [fn.emit_half(net, x) for x in rows]
            with open(os.path.join(lab.REPO, "cache", "bench", "crf", "slopo_Hy_%d_%s.pkl" % (s, K)), "rb") as fh:
                h = pickle.load(fh)
            A, Q = [], []
            for e_, b_, q_, a_ in zip(ev, P["s50"][s], h["q"], h["alive"]):
                n = min(len(e_), len(b_), len(q_), len(a_))
                A.append(0.5 * np.asarray(e_[:n], np.float64) + 0.5 * np.asarray(a_[:n], np.float64))
                Q.append(lab.sg(0.5 * lab.lg(q_[:n]) + 0.5 * lab.lg(b_[:n])))
            mixes[s] = (A, Q)
        bagA, bagQ = lab.mean_paths([mixes[s] for s in seeds])
        ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
        for tag in ("Hy", "Hop"):
            with ThreadPoolExecutor(8) as pool:
                d = list(pool.map(lambda i: sd.decide([Es[tag][s][i] for s in seeds], ref[i], 0.5, 2000, 890000 + i), range(len(ref))))
            for c, v in lab.score(K, d).items():
                out[(K, c, tag)] = float(np.mean(list(v.values())))
        print("%s done %.0f s" % (K, time.time() - t0), flush=True)
    keys = [k for k in lab.CTX if k[0] != "st"]
    print("Hop against Hy, provider held out: %s" % " | ".join("%s %s %+.4f" % (k[0], k[1], out[(k[0], k[1], "Hop")] - out[(k[0], k[1], "Hy")]) for k in keys), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "rows":
        lab.setup()
        for s in (0, 1, 2):
            for K in KS:
                ops_rows(K, s)
                print("  rows %s seed %d" % (K, s), flush=True)
        print("rows done", flush=True)
    elif cmd == "train":
        lab.setup()
        train([int(c) for c in sys.argv[2]], int(os.environ.get("CRF_WORKERS", "12")))
    elif cmd == "lnets":
        lab.setup()
        lnets([int(c) for c in sys.argv[2]], int(os.environ.get("CRF_WORKERS", "9")))
    elif cmd == "eval":
        evaluate()
    elif cmd == "lopo":
        lopo_hop()
