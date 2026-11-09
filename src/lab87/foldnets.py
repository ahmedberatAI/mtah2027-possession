"""Round 88, step 1: the stacker's folds again, same input and same seeds as Hy (src/exp_stack.py train), but keeping each fold's net, so that the path model's joint posterior can be read on held-out halves: per half the emissions and moves, not only the marginals. Open data only.

    python -I foldnets.py train SEED_INDEX [TAG]
    python -I foldnets.py emit SEED_INDEX [TAG]     per half, held out: emissions (T, 5), moves (5, 5); SoccerTrack: the all-data net
"""
import os
import pickle
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
OUTD = os.path.join(REPO, "cache", "lab87", "foldnets")


def _opts(xs):
    """Round 88's capacity arms: STACK_OPTS (JSON, for this tag's first two letters), STACK_WIN."""
    import json
    o = os.environ.get("STACK_OPTS")
    if o:
        xs.OPTS["Hy"] = json.loads(o)
    if os.environ.get("STACK_WIN"):
        xs.WIN = int(os.environ["STACK_WIN"])


def _job(job):
    import torch
    import exp_stack as xs
    _opts(xs)
    s_i, tag, i, hold, seed = job
    path = os.path.join(OUTD, "%s_%d_%s.pt" % (tag, s_i, i))
    if os.path.isfile(path):
        return path
    torch.set_num_threads(1)
    t0 = time.time()
    device = xs.ec.pick_device()
    data = xs.load_rows(s_i, tag)
    e = xs.e_of(tag)
    net, tail = xs.fit(data, hold, seed, device, e=e, step_=xs.step_of(tag), tag=tag)
    d_in = next(iter(data.values()))[0][1].shape[1] - xs.ec.N_OWN - e
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, d_in=d_in, e=e, hold=sorted(str(h) for h in hold)), path)
    print("  %s seed %d fold %s: loss %.4f, %.0f s" % (tag, s_i, i, tail, time.time() - t0), flush=True)
    return path


def train(s_i, tag, workers):
    import exp_stack as xs
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUTD, exist_ok=True)
    data = xs.load_rows(s_i, tag)
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    base = 20261107 + 100 * s_i + 1000 * (ord(tag[-1]) - ord("a") + 1 if tag[-1] in "abc" else 0)
    jobs = [(s_i, tag, i, {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)]), base + i) for i in range(len(dfl))] + [(s_i, tag, "all", set(), base + 50)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_job, jobs))
    print("trained", flush=True)


def emit_half(net, x, device="cpu"):
    """Per step: emissions of the five states; the moves."""
    import torch
    import exp_stack as xs
    with torch.no_grad():
        oh, oa = net(torch.as_tensor(x[None], dtype=torch.float32, device=device))
        e = xs.ec.emissions(oh, oa, "HC")[0].cpu().numpy().astype("float32")
        A = xs.ec.transitions(net, "HC").cpu().numpy().astype("float32")
    return e, A


def emit(s_i, tag):
    import numpy as np
    import torch
    import exp_stack as xs
    from exp_stack import OUT, bench, ec
    _opts(xs)
    torch.set_num_threads(8)
    data = xs.load_rows(s_i, tag)
    nets = []
    for name in sorted(os.listdir(OUTD)):
        if name.startswith("%s_%d_" % (tag, s_i)) and name.endswith(".pt"):
            ck = torch.load(os.path.join(OUTD, name), map_location="cpu", weights_only=False)
            net = xs.make_net(tag, ck["d_in"], ck["e"])
            net.load_state_dict(ck["state"])
            net.eval()
            nets.append((net, set(ck["hold"]), name))
    for k in xs.SETS:
        out = []
        for r in data[k]:
            es, As = [], []
            for net, hold, _ in nets:
                if str(r[0]) in hold:
                    e, A = emit_half(net, r[1])
                    es.append(e)
                    As.append(A)
            assert es, r[0]
            out.append((np.mean(es, axis=0).astype(np.float32), np.mean(As, axis=0).astype(np.float32)))
        with open(os.path.join(OUTD, "emit_%s_%d_%s.pkl" % (tag, s_i, k)), "wb") as fh:
            pickle.dump(out, fh)
        print("  %s emissions done" % k, flush=True)
    d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s_i))
    net_all = [n for n, _, name in nets if name.endswith("_all.pt")][0]
    out = []
    c = xs.canon(tag)
    pb = xs.bf_mean("st@v3", (0, 1, 2))
    if c in ("Gk", "Gg", "Gj", "Gi", "G3"):
        ps = xs.bs_mean("st@v6", [s_i] if c in ("Gk", "Gj", "G3") else [0, 1, 2], {"Gj": "J", "Gi": "J", "G3": "_t"}.get(c, ""))
        cols = [xs._bs_cols(a_, b_) for a_, b_ in zip(pb, ps)]
    else:
        cols = [xs._bf_cols(q_, a_) for q_, a_ in pb]
    for r, z0 in zip(d["rows"], cols):
        z = ec.pool(z0.astype(np.float64)).astype(np.float32)
        n_ = min(len(r[1]), len(z))
        e, A = emit_half(net_all, np.hstack([r[1][:n_], z[:n_]]))
        out.append((e, A))
    with open(os.path.join(OUTD, "emit_%s_%d_st.pkl" % (tag, s_i)), "wb") as fh:
        pickle.dump(out, fh)
    print("done", flush=True)


def _lopo_job(job):
    """Provider k out of the stacker and out of the dead-ball models behind its input (exp_stack._lopo_job, same seed), keeping emissions."""
    import numpy as np
    import torch
    import exp_stack as xs
    from exp_stack import ec
    _opts(xs)                      # round 94: the hold-out fits ignored STACK_OPTS before (Hyw's were Hy's width)
    s_i, tag, j, k = job
    path = os.path.join(OUTD, "emitlopo_%s_%d_%s.pkl" % (tag, s_i, k))
    if os.path.isfile(path):
        return path
    torch.set_num_threads(1)
    t0 = time.time()
    device = ec.pick_device()
    data = xs.load_rows(s_i, tag)
    hold = {r[0] for r in data[k]}
    net, tail = xs.fit(data, hold, 20261207 + 100 * s_i + j, device, e=xs.e_of(tag), step_=xs.step_of(tag), tag=tag)
    ev = xs.lopo_evidence(k)
    out = []
    for r, a in zip(data[k], ev):
        x = r[1].copy()
        col = ec.pool(np.clip(np.asarray(a, np.float64), -20, 20)[:, None] / 5.0).astype(np.float32)
        n = min(len(x), len(col))
        x = x[:n]
        x[:, 2 * ec.N_OWN] = col[:n, 0]
        out.append(emit_half(net, x, device))
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print("  %s seed %d without %s: loss %.4f, %.0f s" % (tag, s_i, k, tail, time.time() - t0), flush=True)
    return path


def lopo(s_i, tag, workers):
    import exp_stack as xs
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUTD, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_lopo_job, [(s_i, tag, xs.SETS.index(k), k) for k in xs.SETS]))
    print("lopo done", flush=True)


def _lnet_job(job):
    """Round 95: the provider-held-out net itself, kept (same data, hold-out and seed as _lopo_job), so that it can be read
    on any input of that provider (the target-condition rows)."""
    import torch
    import exp_stack as xs
    _opts(xs)
    s_i, tag, j, k = job
    path = os.path.join(OUTD, "lnet_%s_%d_%s.pt" % (tag, s_i, k))
    if os.path.isfile(path):
        return path
    torch.set_num_threads(1)
    t0 = time.time()
    device = xs.ec.pick_device()
    data = xs.load_rows(s_i, tag)
    hold = {r[0] for r in data[k]}
    e = xs.e_of(tag)
    net, tail = xs.fit(data, hold, 20261207 + 100 * s_i + j, device, e=e, step_=xs.step_of(tag), tag=tag)
    d_in = next(iter(data.values()))[0][1].shape[1] - xs.ec.N_OWN - e
    torch.save(dict(state={a: v.cpu() for a, v in net.state_dict().items()}, d_in=d_in, e=e, hold=sorted(str(h) for h in hold)), path)
    print("  %s seed %d net without %s: loss %.4f, %.0f s" % (tag, s_i, k, tail, time.time() - t0), flush=True)
    return path


def lnets(seeds, tag, workers):
    import exp_stack as xs
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUTD, exist_ok=True)
    jobs = [(s, tag, xs.SETS.index(k), k) for s in seeds for k in xs.SETS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_lnet_job, jobs))
    print("lnets done", flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.chdir(REPO)
    cmd, s = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "Hy"
    if cmd == "train":
        train(s, tag, int(os.environ.get("CRF_WORKERS", "8")))
    elif cmd == "lopo":
        lopo(s, tag, int(os.environ.get("CRF_WORKERS", "3")))
    elif cmd == "lnets":                      # python -I foldnets.py lnets 012 TAG  (seed digits)
        lnets([int(c) for c in sys.argv[2]], tag, int(os.environ.get("CRF_WORKERS", "9")))
    else:
        emit(s, tag)
