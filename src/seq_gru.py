"""Idea N25 (docs/RAPOR.md): a bidirectional GRU team model over per-frame feature sequences, as an
alternative / complement to the windowed LightGBM team model. Runs on a GPU (Colab); CPU works but is slow.

Input per frame = the bench features of one half (XF + both team views), standardised with training
statistics, NaN -> 0, clipped to +-5, at 5 fps (every STEP-th frame). The network reads [XF, X_own, X_opp] and
predicts P(own team in possession); training uses both side orders (side swap) and alive frames only.
Prediction for a half: p_home = mean(p(XF, XH, XA), 1 - p(XF, XA, XH)), upsampled to 25 fps in logit space.
"""
import numpy as np
import torch
import torch.nn as nn

STEP = 5           # 25 fps -> 5 fps
SEQ = 600          # training windows of 2 min at 5 fps
HOP = 300
CLIP = 5.0


class Net(nn.Module):
    def __init__(self, d_in, hidden=128, layers=2, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=drop if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        y, _ = self.gru(self.inp(x))
        return self.out(y).squeeze(-1)


def _views(f):
    """(home view, away view) per frame at full rate: [XF, XH, XA] and [XF, XA, XH]."""
    return np.hstack([f["XF"], f["XH"], f["XA"]]), np.hstack([f["XF"], f["XA"], f["XH"]])


class Scaler:
    def fit(self, feats):
        X = np.concatenate([_views(f)[0][::STEP] for f in feats]).astype(np.float64)
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 1e-6), self.sd, 1.0)
        return self

    def __call__(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP).astype(np.float32)


def _windows(Z, y):
    n = len(Z)
    starts = list(range(0, max(1, n - SEQ + 1), HOP))
    if n > SEQ and starts[-1] != n - SEQ:
        starts.append(n - SEQ)
    for s in starts:
        yield Z[s:s + SEQ], y[s:s + SEQ]


def fit(feats, labels, epochs=25, lr=2e-3, batch=32, seed=0, device=None, verbose=False):
    """Train on halves (bench feature dicts + per-frame labels 0 H / 1 A / 2 D / 3 X); returns (net, scaler)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = Scaler().fit(feats)
    seqs = []
    for f, lab in zip(feats, labels):
        vh, va = _views(f)
        lab = lab[::STEP]
        zh, za = sc(vh[::STEP]), sc(va[::STEP])
        yh = np.where(lab == 0, 1.0, np.where(lab == 1, 0.0, np.nan)).astype(np.float32)
        for z, y in ((zh, yh), (za, 1.0 - yh)):
            seqs += list(_windows(z, y))
    X = torch.tensor(np.stack([s[0] for s in seqs]))
    Y = torch.tensor(np.stack([s[1] for s in seqs]))
    net = Net(X.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(X) + batch - 1) // batch))
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        tot = cnt = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            m = torch.isfinite(yb)
            out = net(xb)
            loss = (lossf(out, torch.nan_to_num(yb)) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        if verbose:
            print("  epoch %2d loss %.4f" % (ep + 1, tot / cnt), flush=True)
    return net.eval(), sc


@torch.no_grad()
def predict(model, f, device=None):
    """P(home in possession) per frame at 25 fps for one half."""
    net, sc = model
    device = device or next(net.parameters()).device
    vh, va = _views(f)
    out = []
    for v in (vh, va):
        z = torch.tensor(sc(v[::STEP]))[None].to(device)
        out.append(net(z)[0].float().cpu().numpy())
    logit = 0.5 * (out[0] - out[1])
    n = len(f["XF"])
    t = np.arange(0, n, STEP)[:len(logit)]
    full = np.interp(np.arange(n), t, logit)
    return 1.0 / (1.0 + np.exp(-full))
