"""Round 46 (docs/RAPOR.md): a sequence model of the dead ball. The multi-source dead model is a frame-level LightGBM on windowed features; a stoppage is a sequence (ball out,
players slow down, set-up, release). TD: small convolution + bidirectional GRU over the same 'base' features at 5 Hz, trained on DFL, de-glued Metrica and SkillCorner with
equal provider weight in every batch, with two auxiliary heads for stops and restarts to sharpen the boundaries. Seven folds, each holding out one DFL match, one Metrica game and a seventh of
SkillCorner; every half scored by models that never saw its match. AGGREGATE output only. Runs on the local GPU.

  python -I src/exp_tdead.py train SEED      the seven folds and the all-data model
  python -I src/exp_tdead.py infer SEED SRC  the all-data model on another source (st@v3, barath)
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bench  # noqa: E402

SOURCES = {"dfl": ("dfl@v3", "dfl", "native"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}
OUT = os.path.join(bench.CACHE, "tdead")
if os.environ.get("TD_V6"):         # round 49: the same model on sim.V6's masks
    SOURCES = {"dfl": ("dfl@v6", "dfl", "native"), "metrica": ("metrica@v6g", "metrica@v3", "ctrl"), "sc": ("sc@v6", "sc@v3", "evctrl")}
    OUT = os.path.join(bench.CACHE, "tdead6")
STEP = 5            # frames per model step (5 Hz)
WIN = 600           # steps per training window (2 minutes)
STEPS = 4000
BATCH = 24
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, n_in, width=128, drop=0.1):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(n_in, width, 5, padding=2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.gru = nn.GRU(width, width, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.head = nn.Linear(2 * width, 3)

    def forward(self, x):
        h = self.front(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h)


def pool(x, step=STEP):
    """Mean over step frames, NaNs ignored."""
    n = len(x) // step
    x = x[:n * step].reshape(n, step, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis=1)


def targets(lab, step=STEP):
    """Per model step: dead share, mask (known frames), stop and restart heat (Gaussian of about 0.4 s)."""
    lab = np.asarray(lab)
    n = len(lab) // step
    y = lab[:n * step].reshape(n, step)
    known = y < 3
    dead = (y == 2).sum(1) / np.maximum(known.sum(1), 1)
    mask = known.mean(1) > 0.5
    d = (lab == 2).astype(np.int8)
    stops = (np.flatnonzero((d[1:] == 1) & (d[:-1] == 0) & (lab[1:] < 3) & (lab[:-1] < 3)) + 1) // step
    rest = (np.flatnonzero((d[1:] == 0) & (d[:-1] == 1) & (lab[1:] < 3) & (lab[:-1] < 3)) + 1) // step
    heat = np.zeros((n, 2), np.float32)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    for col, idx in ((0, stops), (1, rest)):
        for i in idx:
            a, b = max(0, i - 6), min(n, i + 7)
            heat[a:b, col] = np.maximum(heat[a:b, col], kernel[a - i + 6:b - i + 6])
    return dead.astype(np.float32), mask, heat


def load_all():
    data = {}
    for k, (fsrc, lsrc, conv) in SOURCES.items():
        fs = bench.features("base", fsrc)
        ls = bench.labels(lsrc, conv)
        gs = [b["group"] for b in bench.blocks(fsrc)]
        rows = []
        for f, l, g in zip(fs, ls, gs):
            x = pool(f["XF"])
            rows.append((g, x.astype(np.float32),) + targets(l))
        data[k] = rows
        print("  %s: %d halves" % (k, len(rows)), flush=True)
    return data


def fit(data, hold, seed):
    """Train on everything except the groups in hold; returns (net, mean, scale)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = {k: [r for r in rows if r[0] not in hold] for k, rows in data.items()}
    xs = np.concatenate([r[1][::5] for rows in pools.values() for r in rows])
    mean, scale = np.nanmean(xs, axis=0), np.nanstd(xs, axis=0) + 1e-6
    n_in = xs.shape[1]
    net = Net(n_in).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    keys = list(pools)
    net.train()
    for step in range(STEPS):
        X, Yd, M, H = [], [], [], []
        for b in range(BATCH):
            rows = pools[keys[b % len(keys)]]
            _, x, d, m, h = rows[int(rng.integers(len(rows)))]
            a = int(rng.integers(0, max(1, len(x) - WIN)))
            xx = np.nan_to_num((x[a:a + WIN] - mean) / scale, nan=0.0)
            X.append(xx)
            Yd.append(d[a:a + WIN])
            M.append(m[a:a + WIN])
            H.append(h[a:a + WIN])
        L = min(len(v) for v in X)
        X = torch.as_tensor(np.stack([v[:L] for v in X]), dtype=torch.float32, device=DEVICE)
        Yd = torch.as_tensor(np.stack([v[:L] for v in Yd]), dtype=torch.float32, device=DEVICE)
        M = torch.as_tensor(np.stack([v[:L] for v in M]), dtype=torch.float32, device=DEVICE)
        H = torch.as_tensor(np.stack([v[:L] for v in H]), dtype=torch.float32, device=DEVICE)
        out = net(X)
        loss = (nn.functional.binary_cross_entropy_with_logits(out[:, :, 0], Yd, reduction="none") * M).sum() / M.sum().clamp(min=1)
        loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(out[:, :, 1:], H)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
    net.eval()
    return net, mean, scale


def score_half(net, mean, scale, xf):
    """P(dead) per frame of a half: the model at 5 Hz over the whole half, linearly interpolated to 25 fps."""
    x = pool(xf)
    xx = np.nan_to_num((x - mean) / scale, nan=0.0)
    with torch.no_grad():
        out = net(torch.as_tensor(xx[None], dtype=torch.float32, device=DEVICE))[0, :, 0].cpu().numpy()
    grid = (np.arange(len(out)) + 0.5) * STEP
    z = np.interp(np.arange(len(xf)) + 0.5, grid, out)
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def train(seed):
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    feats = {k: bench.features("base", SOURCES[k][0]) for k in SOURCES}
    res = {k: [[] for _ in data[k]] for k in SOURCES}
    for i in range(len(dfl)):
        t0 = time.time()
        hold = {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)])
        net, mean, scale = fit(data, hold, seed + i)
        for k in SOURCES:
            for j, r in enumerate(data[k]):
                if r[0] in hold:
                    res[k][j].append(score_half(net, mean, scale, feats[k][j]["XF"]))
        print("  seed %d fold %d done, %.0f s" % (seed, i, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()
    for k in SOURCES:
        out = [np.mean(np.stack(v), axis=0) for v in res[k]]
        assert all(len(v) for v in res[k])
        bench._dump(out, os.path.join(OUT, "held_%d_%s.pkl" % (seed, SOURCES[k][0].replace("@", "-"))))
    net, mean, scale = fit(data, set(), seed + 100)
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mean=mean, scale=scale, n_in=len(mean)), os.path.join(OUT, "all_%d.pt" % seed))
    print("  seed %d all-data model done" % seed, flush=True)


def load_all_model(seed):
    ck = torch.load(os.path.join(OUT, "all_%d.pt" % int(seed)), map_location="cpu", weights_only=False)
    net = Net(ck["n_in"]).to(DEVICE)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mean"], ck["scale"]


def held(src, seeds=(20261104, 20261105, 20261106)):
    """Mean held-out P(dead) per half over the seeds."""
    runs = [bench._load(os.path.join(OUT, "held_%d_%s.pkl" % (s, src.replace("@", "-")))) for s in seeds]
    return [np.mean(np.stack([r[i] for r in runs]), axis=0) for i in range(len(runs[0]))]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "train":
        train(sys.argv[2])
    else:
        raise SystemExit(__doc__)
