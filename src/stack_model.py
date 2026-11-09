"""The stacker of model_v21 (rounds 61, 65-68, docs/RAPOR.md), on its own: the research code is src/exp_crf.py and src/exp_stack.py.

A small sequence model on top of the team channels' outputs. Per team, at 5 Hz: the logits of v10, the shared BiGRU, J1 and of their +-1 s blend, J1's gain / lose / touch heads, each team's distance to the ball and its players within 3 m, the ball-free channel's logit; common to both: the alive evidence, J1's dead logit, the ball's speed and whether it is seen, the ball-free channel's alive evidence. A convolution and a bidirectional GRU (one view per team, symmetric by construction) give per step the emissions of a path model:

    states   H control, H free, A control, A free, dead
    moves    to the other team only into its control state; free is latent, summed over

It was trained on the whole path's likelihood on the channels' held-out outputs of the open games only, with random channel scales and dropped channels so that it does not lean on their exact calibration. It has never seen BAR-ATH. This file evaluates it; it needs no match data at import and has no training code.
"""
import warnings

import numpy as np
import torch

import model_v9
import short_specialist as ss

STEP = 5
CLIP = 6.0
N_OWN = 14
BASE_COLS = 2 * N_OWN + 4
_FIX = -1e4


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.clip(np.log(p / (1 - p)), -CLIP, CLIP)


def track_cols(b):
    """Per frame, from tracking: distance of each team to the ball, players within 3 m, ball speed, ball seen."""
    L, W = b["pitch"]
    pxy = np.asarray(b["P"], np.float32)[:, :, :2] * np.array([L, W], np.float32)
    bxy = np.asarray(b["B"], np.float32)[:, :2] * np.array([L, W], np.float32)
    team = np.asarray(b["team"])
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d0, d1 = np.where(team == 0, dist, np.nan), np.where(team == 1, dist, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dh, da = np.nanmin(d0, axis=1), np.nanmin(d1, axis=1)
    n3h, n3a = np.nansum(d0 < 3.0, axis=1), np.nansum(d1 < 3.0, axis=1)
    seen = np.isfinite(bxy[:, 0])
    sp = np.zeros(len(bxy), np.float32)
    v = np.flatnonzero(seen)
    if len(v) > 1:
        gap = np.diff(v)
        step = np.linalg.norm(np.diff(bxy[v], axis=0), axis=1) / gap * 25.0
        sp[v[1:]] = np.where(gap <= 5, step, 0.0)
    fill = lambda x: np.where(np.isfinite(x), np.log1p(np.clip(x, 0, 30)), np.log1p(30.0))  # noqa: E731
    return fill(dh), fill(da), n3h / 3.0, n3a / 3.0, np.clip(sp, 0, 40) / 10.0, seen.astype(np.float32)


def frame_inputs(b, v, g, o, a):
    """(T, N) float32: columns in pairs for the two views. Order: own columns for H, then for A, then common."""
    o = np.asarray(o, np.float32)
    n = min(len(v), len(g), len(o), len(a), len(b["P"]))
    v, g, o, a = np.asarray(v[:n], np.float64), np.asarray(g[:n], np.float64), o[:n], np.asarray(a[:n], np.float64)
    pj = np.asarray(ss.pk(o, 0), np.float64)
    blend = model_v9.smooth(0.25 * v + 0.25 * g + 0.5 * pj)
    lv, lg, lj, ls = logit(v), logit(g), logit(pj), logit(blend)
    dh, da, nh, na, sp, seen = (x[:n] for x in track_cols(b))
    gain_h, gain_a, lose_h, lose_a, touch_h, touch_a = (logit(o[:, c]) for c in (3, 4, 5, 6, 7, 8))
    own_h = [lv, lg, lj, ls, gain_h, gain_a, lose_h, lose_a, touch_h, touch_a, dh, da, nh, na]
    own_a = [-lv, -lg, -lj, -ls, gain_a, gain_h, lose_a, lose_h, touch_a, touch_h, da, dh, na, nh]
    com = [np.clip(a, -20, 20) / 5.0, logit(o[:, 2]), sp, seen]
    return np.stack(own_h + own_a + com, 1).astype(np.float32)


def pool(x, step=STEP):
    n = len(x) // step
    return x[:n * step].reshape(n, step, -1).mean(1)


def ballfree_cols(q, alive):
    """The ball-free channel's columns: its logit in both views, then its alive evidence."""
    return np.stack([logit(q), -logit(q), np.clip(alive, -20, 20) / 5.0], 1)


def inputs(b, v, g, o, a, bf_q, bf_alive):
    """The 5 Hz input of one half."""
    x = pool(frame_inputs(b, v, g, o, a)).astype(np.float32)
    z = pool(ballfree_cols(bf_q, bf_alive).astype(np.float64)).astype(np.float32)
    n = min(len(x), len(z))
    return np.hstack([x[:n], z[:n]])


def view(x, side, e=0):
    """x (B, T, N) -> one team's view: own columns, common, extra own, extra common."""
    n = N_OWN
    parts = [x[..., :n] if side == 0 else x[..., n:2 * n], x[..., 2 * n:BASE_COLS]]
    if e:
        parts.append(x[..., BASE_COLS:BASE_COLS + e] if side == 0 else x[..., BASE_COLS + e:BASE_COLS + 2 * e])
    parts.append(x[..., BASE_COLS + 2 * e:])
    return torch.cat(parts, -1)


def build_net(d_in, width=64, drop=0.1, e=0):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Conv1d(d_in, width, 5, padding=2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, 3)      # own control, own free, dead
            self.t = nn.Parameter(torch.zeros(8))    # c>f, f>c, c>C, f>C, c>d, f>d, d>c, d>f

        def one(self, x):
            h = self.front(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.gru(h)
            return self.head(h)

        def forward(self, x):
            return self.one(view(x, 0, e)), self.one(view(x, 1, e))
    return Net()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(ck["d_in"], width=int(ck["state"]["front.0.weight"].shape[0]), e=ck.get("e", 0))
    net.load_state_dict(ck["state"])
    return net.eval()


def moves(t):
    """Log potentials (5, 5): 0 H control, 1 H free, 2 A control, 3 A free, 4 dead."""
    fx = torch.full((), _FIX)
    z = 0 * t[0]
    return torch.stack([
        torch.stack([z, t[0], t[2], fx, t[4]]),
        torch.stack([t[1], z, t[3], fx, t[5]]),
        torch.stack([t[2], fx, z, t[0], t[4]]),
        torch.stack([t[3], fx, t[1], z, t[5]]),
        torch.stack([t[6], t[7], t[6], t[7], z])])


def score_half(net, x, n):
    """(q at 25 fps: P(H | live), alive log-odds at 25 fps) of a whole half, by forward-backward over the path model."""
    with torch.no_grad():
        oh, oa = net(torch.as_tensor(x[None], dtype=torch.float32))
        d = 0.5 * (oh[..., 2] + oa[..., 2])
        e = torch.stack([oh[..., 0], oh[..., 1], oa[..., 0], oa[..., 1], d], -1)[0]
        A = moves(net.t)
        T, S = e.shape
        fw = torch.zeros((T, S))
        bw = torch.zeros((T, S))
        fw[0] = e[0]
        for t in range(1, T):
            fw[t] = torch.logsumexp(fw[t - 1].unsqueeze(1) + A, dim=0) + e[t]
        for t in range(T - 2, -1, -1):
            bw[t] = torch.logsumexp(A + (e[t + 1] + bw[t + 1]).unsqueeze(0), dim=1)
        post = fw + bw
        post = post - torch.logsumexp(post, dim=1, keepdim=True)
        lp = torch.stack([torch.logsumexp(post[:, [0, 1]], dim=1), torch.logsumexp(post[:, [2, 3]], dim=1), post[:, 4]], 1).numpy()
    grid = (np.arange(len(lp)) + 0.5) * STEP
    t = np.arange(n) + 0.5
    lq = lp[:, 0] - lp[:, 1]
    q = 1 / (1 + np.exp(-np.interp(t, grid, lq)))
    alive = np.logaddexp(lp[:, 0], lp[:, 1]) - lp[:, 2]
    return q.astype(np.float32), np.interp(t, grid, alive).astype(np.float32)
