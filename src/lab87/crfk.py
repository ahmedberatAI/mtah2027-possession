"""Round 89: if the joint posterior decides, its structure matters in a way the marginals never showed. The path model today has two latent states per team (control, free) with fixed roles. Here K latent states per team and free moves among them, tied between the teams: phase-type durations, so that "just won it, may lose it at once" and "settled possession" can be different states. Same input, folds, seeds and augmentation as Hy. Open data only.

    python -I crfk.py train SEED K [TAG]
    python -I crfk.py emit SEED K [TAG]
"""
import os
import pickle
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
OUTD = os.path.join(REPO, "cache", "lab87", "foldnets")


def build(d_in, K, e, width=64, drop=0.1):
    import torch
    import exp_stack as xs
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Conv1d(d_in, width, 5, padding=2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, K + 1)
            self.tw = nn.Parameter(torch.full((K, K), -3.0))
            ts = torch.full((K, K), -8.0)
            ts[:, 0] = -5.0
            self.ts = nn.Parameter(ts)
            self.td = nn.Parameter(torch.full((K,), -5.0))
            tr = torch.full((K,), -6.0)
            tr[0] = -3.0
            self.tr = nn.Parameter(tr)
            self.K = K

        def one(self, x):
            h = self.front(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.gru(h)
            return self.head(h)

        def forward(self, x):
            return self.one(xs.view(x, 0, e)), self.one(xs.view(x, 1, e))

        def moves(self):
            K = self.K
            tw = self.tw - torch.diag(torch.diagonal(self.tw))
            A = torch.zeros((2 * K + 1, 2 * K + 1), device=tw.device)
            A[:K, :K] = tw
            A[K:2 * K, K:2 * K] = tw
            A[:K, K:2 * K] = self.ts
            A[K:2 * K, :K] = self.ts
            A[:K, 2 * K] = self.td
            A[K:2 * K, 2 * K] = self.td
            A[2 * K, :K] = self.tr
            A[2 * K, K:2 * K] = self.tr
            return A

        def emissions(self, oh, oa):
            K = self.K
            return torch.cat([oh[..., :K], oa[..., :K], 0.5 * (oh[..., K:K + 1] + oa[..., K:K + 1])], -1)
    return Net()


def nll(net, x, y):
    import torch
    import exp_stack as xs
    oh, oa = net(x)
    e = net.emissions(oh, oa)
    A = net.moves()
    K = net.K
    m = torch.full(y.shape + (2 * K + 1,), -1e4, device=x.device)
    m[..., :K][y == 0] = 0.0
    m[..., K:2 * K][y == 1] = 0.0
    m[..., 2 * K][y == 2] = 0.0
    m[y < 0] = 0.0
    B = len(x)
    z = xs.ec.forward(torch.cat([e, e + m]), A)
    return (z[:B] - z[B:]).sum() / (B * x.shape[1])


def fit(data, hold, seed, K, e):
    import numpy as np
    import torch
    import exp_stack as xs
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = {k: [r for r in rows if r[0] not in hold] for k, rows in data.items()}
    keys = [k for k in pools if pools[k]]
    d_in = next(iter(data.values()))[0][1].shape[1] - xs.ec.N_OWN - e
    net = build(d_in, K, e)
    opt = torch.optim.AdamW(net.parameters(), lr=xs.LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, xs.LR, total_steps=xs.STEPS, pct_start=0.1)
    net.train()
    tail = []
    for step in range(xs.STEPS):
        X, Y = [], []
        for b in range(xs.BATCH):
            rows = pools[keys[b % len(keys)]]
            _, x, y, _ = rows[int(rng.integers(len(rows)))]
            a = int(rng.integers(0, max(1, len(x) - xs.WIN)))
            X.append(xs.augment(x[a:a + xs.WIN], rng))
            Y.append(y[a:a + xs.WIN])
        L = min(len(v) for v in X)
        X = torch.as_tensor(np.stack([v[:L] for v in X]), dtype=torch.float32)
        Y = torch.as_tensor(np.stack([v[:L] for v in Y]), dtype=torch.long)
        loss = nll(net, X, Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        tail.append(float(loss))
    net.eval()
    return net, float(np.mean(tail[-200:])), d_in


def _job(job):
    import torch
    import exp_stack as xs
    s_i, tag, K, i, hold, seed = job
    path = os.path.join(OUTD, "%s_%d_%s.pt" % (tag, s_i, i))
    if os.path.isfile(path):
        return path
    torch.set_num_threads(1)
    t0 = time.time()
    data = xs.load_rows(s_i, "Hy")
    net, tail, d_in = fit(data, hold, seed, K, xs.e_of("Hy"))
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, d_in=d_in, e=xs.e_of("Hy"), K=K, hold=sorted(str(h) for h in hold)), path)
    print("  %s seed %d fold %s: loss %.4f, %.0f s" % (tag, s_i, i, tail, time.time() - t0), flush=True)
    return path


def train(s_i, K, tag, workers):
    import exp_stack as xs
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUTD, exist_ok=True)
    data = xs.load_rows(s_i, "Hy")
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    base = 20261107 + 100 * s_i
    jobs = [(s_i, tag, K, i, {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)]), base + i) for i in range(len(dfl))] + [(s_i, tag, K, "all", set(), base + 50)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_job, jobs))
    print("trained", flush=True)


def emit(s_i, tag):
    import numpy as np
    import torch
    import exp_stack as xs
    from exp_stack import OUT, bench, ec
    torch.set_num_threads(8)
    data = xs.load_rows(s_i, "Hy")
    nets = []
    for name in sorted(os.listdir(OUTD)):
        if name.startswith("%s_%d_" % (tag, s_i)) and name.endswith(".pt"):
            ck = torch.load(os.path.join(OUTD, name), map_location="cpu", weights_only=False)
            net = build(ck["d_in"], ck["K"], ck["e"])
            net.load_state_dict(ck["state"])
            net.eval()
            nets.append((net, set(ck["hold"]), name))

    def one(net, x):
        with torch.no_grad():
            oh, oa = net(torch.as_tensor(x[None], dtype=torch.float32))
            return net.emissions(oh, oa)[0].numpy().astype("float32"), net.moves().numpy().astype("float32")
    for k in xs.SETS:
        out = []
        for r in data[k]:
            es, As = [], []
            for net, hold, _ in nets:
                if str(r[0]) in hold:
                    e, A = one(net, r[1])
                    es.append(e)
                    As.append(A)
            if len(es) > 1:
                # latent states are not aligned between fold models: keep the first fold's model for this half
                es, As = es[:1], As[:1]
            out.append((es[0], As[0]))
        with open(os.path.join(OUTD, "emit_%s_%d_%s.pkl" % (tag, s_i, k)), "wb") as fh:
            pickle.dump(out, fh)
        print("  %s emissions done" % k, flush=True)
    d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s_i))
    net_all = [n for n, _, name in nets if name.endswith("_all.pt")][0]
    out = []
    for r, (q_, a_) in zip(d["rows"], xs.bf_mean("st@v3", (0, 1, 2))):
        z = ec.pool(xs._bf_cols(q_, a_).astype(np.float64)).astype(np.float32)
        n_ = min(len(r[1]), len(z))
        out.append(one(net_all, np.hstack([r[1][:n_], z[:n_]])))
    with open(os.path.join(OUTD, "emit_%s_%d_st.pkl" % (tag, s_i)), "wb") as fh:
        pickle.dump(out, fh)
    print("done", flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.chdir(REPO)
    cmd, s, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "HK%d" % K
    if cmd == "train":
        train(s, K, tag, int(os.environ.get("CRF_WORKERS", "8")))
    else:
        emit(s, tag)
