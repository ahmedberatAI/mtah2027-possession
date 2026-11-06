"""Rounds 61 / 65 (docs/RAPOR.md): the stacker on the channels' held-out outputs, without the path layer. Same input and folds as src/exp_crf.py, a plain per-step softmax, so a fold takes minutes and inputs, resolution, provider hold-outs and seeds can be measured quickly. Aggregate output only.

  python -I src/exp_stack.py train SEED_INDEX TAG            seven folds and the all-data model
  python -I src/exp_stack.py lopo SEED_INDEX TAG             each provider held out completely
  python -I src/exp_stack.py eval "TAGS" "SEEDS" [held|lopo] arms against v20 on the open sets
  python -I src/exp_stack.py stinputs SEED_INDEX             SoccerTrack's per-half inputs (unseen by every model)
  python -I src/exp_stack.py st "TAGS" "SEEDS"               arms against v20 on SoccerTrack
"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = list(sys.argv)
sys.argv = [sys.argv[0], "", "none"]
import exp_crf as ec  # noqa: E402
from exp_crf import OUT, SETS, STEP, bench, ee, eg, eg2, er, es, exp_blend, exp_smooth, ro, ss  # noqa: E402
sys.argv = _argv

RUNS = os.path.join(bench.ROOT, "tools", "runs")
WIN = 300
STEPS = 2000
BATCH = 16
LR = 1e-3
BASE = {"H": (6.0, 24.0)}
ARMS = {"q": (0.5, 6.0, 24.0, 0.0), "S.25": (0.5, 6.0, 24.0, 0.25), "S1": (0.5, 6.0, 24.0, 0.5), "S.75": (0.5, 6.0, 24.0, 0.75)}     # stacker's share in q, pen_team, c, stacker's share in the alive evidence


if os.environ.get("STACK_ARMS") == "wide":
    ARMS = dict(ARMS)
    ARMS.update({"w.35 a.25": (0.35, 6.0, 24.0, 0.25), "w.35 a.5": (0.35, 6.0, 24.0, 0.5), "w.25 a.25": (0.25, 6.0, 24.0, 0.25), "8/24 a.25": (0.5, 8.0, 24.0, 0.25), "w.35 8/24": (0.35, 8.0, 24.0, 0.25), "w.25 8/48": (0.25, 8.0, 48.0, 0.25)})
E_OWN = {"Gp": 2, "Gm": 6, "Gf": 1, "Gu": 1}      # extra columns that swap between the two views (see view)
STEP_OF = {"G2": 2}             # frames per step of a tag (default exp_crf.STEP, 5 Hz)
OPTS = {"Hw": dict(width=128), "Hl": dict(steps=4000), "Hd": dict(drop=0.3)}      # path-model variants on the base input


def is_path(tag):
    """H...: the path model of exp_crf (arm HC) on this file's inputs; "HC" alone means exp_crf's own files."""
    return tag[0] == "H" and tag != "HC"


def canon(tag):
    """The input of a tag: Gs and Hs read the same columns; Hv = Hf's input (round 66) with Hz's augmentation."""
    return "Gf" if tag[:2] == "Hv" else "G" + tag[1:2]


def augmented(tag):
    return tag[:2] in ("Hz", "Hv")


def step_of(tag):
    return STEP_OF.get(canon(tag), STEP)


def e_of(tag):
    return E_OWN.get(canon(tag), 0)


BASE_COLS = 2 * ec.N_OWN + 4


def view(x, side, e=0):
    """x (B, T, N) -> one team's view: own columns, common, extra own, extra common."""
    n = ec.N_OWN
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
            self.head = nn.Linear(2 * width, 2)      # own control, dead

        def one(self, x):
            h = self.front(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.gru(h)
            return self.head(h)

        def forward(self, x):
            oh, oa = self.one(view(x, 0, e)), self.one(view(x, 1, e))
            return torch.stack([oh[..., 0], oa[..., 0], 0.5 * (oh[..., 1] + oa[..., 1])], -1)
    return Net()


@torch.jit.script
def forward_tv(e: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """log Z over the paths with per-step moves: e (B, T, S), A (B, T, S, S)."""
    al = e[:, 0]
    for t in range(1, e.shape[1]):
        al = torch.logsumexp(al.unsqueeze(2) + A[:, t], dim=1) + e[:, t]
    return torch.logsumexp(al, dim=1)


def moves_tv(net, uh, ua):
    """(B, T, 5, 5): the fixed moves of exp_crf.transitions plus what each team's view says about this step: gain, free, back, start, stop. Moves to the other team still enter only its control state."""
    A = ec.transitions(net, "HC").view(1, 1, 5, 5).expand(uh.shape[0], uh.shape[1], 5, 5).clone()
    A[:, :, 0, 2] += ua[..., 0]
    A[:, :, 1, 2] += ua[..., 0]
    A[:, :, 2, 0] += uh[..., 0]
    A[:, :, 3, 0] += uh[..., 0]
    A[:, :, 0, 1] += uh[..., 1]
    A[:, :, 2, 3] += ua[..., 1]
    A[:, :, 1, 0] += uh[..., 2]
    A[:, :, 3, 2] += ua[..., 2]
    A[:, :, 4, 0] += uh[..., 3]
    A[:, :, 4, 1] += uh[..., 3]
    A[:, :, 4, 2] += ua[..., 3]
    A[:, :, 4, 3] += ua[..., 3]
    A[:, :, 0, 4] += uh[..., 4]
    A[:, :, 1, 4] += uh[..., 4]
    A[:, :, 2, 4] += ua[..., 4]
    A[:, :, 3, 4] += ua[..., 4]
    return A


def nll_tv(net, x, y):
    (oh, uh), (oa, ua) = net.both(x)
    e = ec.emissions(oh, oa, "HC")
    A = moves_tv(net, uh, ua)
    B = len(x)
    z = forward_tv(torch.cat([e, e + ec.allowed(y, "HC", x.device)]), torch.cat([A, A]))
    return (z[:B] - z[B:]).sum() / (B * x.shape[1])


def score_tv(net, x, n, device, step_=STEP):
    with torch.no_grad():
        (oh, uh), (oa, ua) = net.both(torch.as_tensor(x[None], dtype=torch.float32, device=device))
        e = ec.emissions(oh, oa, "HC")[0]
        A = moves_tv(net, uh, ua)[0]
        T, S = e.shape
        fw = torch.zeros((T, S), device=e.device)
        bw = torch.zeros((T, S), device=e.device)
        fw[0] = e[0]
        for t in range(1, T):
            fw[t] = torch.logsumexp(fw[t - 1].unsqueeze(1) + A[t], dim=0) + e[t]
        for t in range(T - 2, -1, -1):
            bw[t] = torch.logsumexp(A[t + 1] + (e[t + 1] + bw[t + 1]).unsqueeze(0), dim=1)
        post = fw + bw
        post = post - torch.logsumexp(post, dim=1, keepdim=True)
        lp = torch.stack([torch.logsumexp(post[:, [0, 1]], dim=1), torch.logsumexp(post[:, [2, 3]], dim=1), post[:, 4]], 1).cpu().numpy()
    grid = (np.arange(len(lp)) + 0.5) * step_
    t = np.arange(n) + 0.5
    q = 1 / (1 + np.exp(-np.interp(t, grid, lp[:, 0] - lp[:, 1])))
    alive = np.logaddexp(lp[:, 0], lp[:, 1]) - lp[:, 2]
    return q.astype(np.float32), np.interp(t, grid, alive).astype(np.float32)


def build_tv(d_in, width=64, drop=0.1, e=0):
    """Round 65: the path model whose moves also depend on the step (feature-dependent edges), so that a team change can be cheap exactly where the evidence for one is."""
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Conv1d(d_in, width, 5, padding=2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, 3)
            self.edge = nn.Linear(2 * width, 5)
            nn.init.zeros_(self.edge.weight)
            nn.init.zeros_(self.edge.bias)
            self.t = nn.Parameter(torch.tensor([-3.0, -2.0, -6.0, -4.0, -5.0, -5.0, -3.0, -6.0]))

        def one(self, x):
            h = self.front(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.gru(h)
            return self.head(h), self.edge(h)

        def both(self, x):
            return self.one(view(x, 0, e)), self.one(view(x, 1, e))
    return Net()


def build_path(d_in, width=64, drop=0.1, e=0):
    """exp_crf.build_net with this file's views."""
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Conv1d(d_in, width, 5, padding=2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, 3)      # own control, own free, dead
            self.t = nn.Parameter(torch.tensor([-3.0, -2.0, -6.0, -4.0, -5.0, -5.0, -3.0, -6.0]))    # c>f, f>c, c>C, f>C, c>d, f>d, d>c, d>f

        def one(self, x):
            h = self.front(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.gru(h)
            return self.head(h)

        def forward(self, x):
            return self.one(view(x, 0, e)), self.one(view(x, 1, e))
    return Net()


def make_net(tag, d_in, e):
    o = OPTS.get(tag[:2], {})
    if tag[:2] == "Ht":
        return build_tv(d_in, e=e)
    if is_path(tag):
        return build_path(d_in, width=o.get("width", 64), drop=o.get("drop", 0.1), e=e)
    return build_net(d_in, e=e)


CH_COLS = [0, 1, 2, 3]            # channel logits in H's view; the same + N_OWN in A's
HEAD_COLS = [4, 5, 6, 7, 8, 9]     # J1's gain / lose / touch heads


def augment(x, rng):
    """Round 65 (Hz), one training window: the channels' quality and calibration differ on a provider nobody trained them on (SkillCorner, SoccerTrack, the target), so the stacker must not lean on their exact scale. Random scale per channel, sometimes a channel missing for the whole window, scale and offset on the alive evidence."""
    x = x.copy()
    n = ec.N_OWN
    for c in CH_COLS:
        f = rng.uniform(0.5, 1.2)
        if c != 3 and rng.random() < 0.1:
            f = 0.0
        x[:, c] *= f
        x[:, c + n] *= f
    f = rng.uniform(0.7, 1.2)
    for c in HEAD_COLS:
        x[:, c] *= f
        x[:, c + n] *= f
    x[:, 2 * n] = x[:, 2 * n] * rng.uniform(0.7, 1.2) + rng.uniform(-0.3, 0.3)
    if x.shape[1] > BASE_COLS:             # the ball-free channel: both views' logit, then its alive evidence
        f = 0.0 if rng.random() < 0.1 else rng.uniform(0.5, 1.2)
        x[:, BASE_COLS] *= f
        x[:, BASE_COLS + 1] *= f
        x[:, BASE_COLS + 2] *= rng.uniform(0.7, 1.2)
    return x


def fit(data, hold, seed, device, steps=None, e=0, step_=STEP, tag="G"):
    steps = steps or OPTS.get(tag[:2], {}).get("steps", STEPS)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = {k: [r for r in rows if r[0] not in hold] for k, rows in data.items()}
    if tag[:2] == "Hq":                        # learning curve: half of each provider's other matches
        for k in pools:
            groups = sorted({r[0] for r in pools[k]}, key=str)
            keep = set(np.random.default_rng(seed + 7).permutation(len(groups))[:max(1, len(groups) // 2)].tolist())
            keep = {g for i, g in enumerate(groups) if i in keep}
            pools[k] = [r for r in pools[k] if r[0] in keep]
    keys = [k for k in pools if pools[k]]
    d_in = next(iter(data.values()))[0][1].shape[1] - ec.N_OWN - e
    net = make_net(tag, d_in, e).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    net.train()
    tail = []
    for step in range(steps):
        X, Y = [], []
        for b in range(BATCH):
            rows = pools[keys[b % len(keys)]]
            _, x, y, _ = rows[int(rng.integers(len(rows)))]
            win = WIN * STEP // step_
            a = int(rng.integers(0, max(1, len(x) - win)))
            X.append(augment(x[a:a + win], rng) if augmented(tag) else x[a:a + win])
            Y.append(y[a:a + win])
        L = min(len(v) for v in X)
        X = torch.as_tensor(np.stack([v[:L] for v in X]), dtype=torch.float32, device=device)
        Y = torch.as_tensor(np.stack([v[:L] for v in Y]), dtype=torch.long, device=device)
        if tag[:2] == "Ht":
            loss = nll_tv(net, X, Y)
        elif is_path(tag):
            loss = ec.nll(net, X, Y, "HC")
        else:
            o = net(X)
            ok = Y >= 0
            loss = torch.nn.functional.cross_entropy(o[ok], Y[ok])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        tail.append(float(loss))
    net.eval()
    return net, float(np.mean(tail[-200:]))


def score_half(net, x, n, device, step_=STEP, tag="G"):
    """(q at 25 fps: P(H | live), alive log-odds at 25 fps)"""
    if tag[:2] == "Ht":
        return score_tv(net, x, n, device, step_)
    if is_path(tag):
        return ec.score_half(net, x, n, "HC", device, step_)[:2]
    with torch.no_grad():
        o = net(torch.as_tensor(x[None], dtype=torch.float32, device=device))[0]
        lp = torch.log_softmax(o, -1).cpu().numpy()
    grid = (np.arange(len(lp)) + 0.5) * step_
    t = np.arange(n) + 0.5
    lq = lp[:, 0] - lp[:, 1]
    q = 1 / (1 + np.exp(-np.interp(t, grid, lq)))
    alive = np.logaddexp(lp[:, 0], lp[:, 1]) - lp[:, 2]
    return q.astype(np.float32), np.interp(t, grid, alive).astype(np.float32)


def _channel_stats(src, s_i):
    """Per half, each channel over its three seeds: mean logit and spread, H's view then A's."""
    res = [eg2.parts(src, j, True) for j in range(3)]
    out = []
    for i in range(len(res[0][0])):
        lv = np.stack([ec._logit(res[j][0][i]) for j in range(3)])
        lg = np.stack([ec._logit(res[j][1][i]) for j in range(3)])
        lj = np.stack([ec._logit(ss.pk(np.asarray(res[j][2][i], np.float32), 0)) for j in range(3)])
        n = min(lv.shape[1], lg.shape[1], lj.shape[1])
        m = [z[:, :n].mean(0) for z in (lv, lg, lj)]
        d = [z[:, :n].std(0) for z in (lv, lg, lj)]
        h = [m[0], d[0], m[1], d[1], m[2], d[2]]
        a = [-m[0], d[0], -m[1], d[1], -m[2], d[2]]
        out.append(np.stack(h + a, 1))
    return out


def _bf_path(src, s_i):
    """The ball-free channel's outputs of this seed index; index 0's while the others are not trained yet."""
    path = os.path.join(bench.ROOT, "cache", "ballfree", "held_%s_%d.pkl" % (src.replace("@", "-"), s_i))
    return path if os.path.isfile(path) else os.path.join(bench.ROOT, "cache", "ballfree", "held_%s_0.pkl" % src.replace("@", "-"))


def _bf_cols(q, a):
    return np.stack([ec._logit(q), -ec._logit(q), np.clip(a, -20, 20) / 5.0], 1)


def extras(tag, k, s_i):
    """Round 65's added inputs per half at 25 fps (they are pooled like the others); None for the base input."""
    fsrc, src = ee.V6[k][0], ee.V3[k][1]
    if tag.startswith("Gh"):
        runs = [bench._load(os.path.join(bench.CACHE, "tdead6n", "heads_%d_%s.pkl" % (s_, fsrc.replace("@", "-")))) for s_ in (20261104, 20261105, 20261106)]
        return [ec._logit(np.mean(np.stack([np.asarray(r[i], np.float32) for r in runs]), axis=0)) for i in range(len(runs[0]))]
    if tag.startswith("Gs"):
        with open(ro.OUT) as fh:
            conf = json.load(fh)
        tr6, vi6 = ee.rule_evidence(k, ee.V6, conf)
        m6 = [exp_blend.logit_alive(x) for x in bench._load(os.path.join(ee.OUT, "M_v6_%s.pkl" % fsrc.replace("@", "-")))]
        tn = ee._held("tdead6n", fsrc)
        return [np.clip(np.stack([t, m, x, v], 1), -20, 20) / 5.0 for t, m, x, v in zip(tr6, m6, tn, vi6)]
    if tag.startswith("Gm"):
        return _channel_stats(src, s_i)
    if tag.startswith("Gf"):
        h = bench._load(_bf_path(ee.V3[k][0 if k != "metrica" else 1], s_i))
        return [np.stack([ec._logit(q), -ec._logit(q), np.clip(a, -20, 20) / 5.0], 1) for q, a in zip(h["q"], h["alive"])]
    if tag.startswith("Gu"):                   # the dose test: the ball-free channel after a quarter of the training
        h = bench._load(os.path.join(bench.ROOT, "cache", "ballfree", "held_q_%s_%d.pkl" % (ee.V3[k][0 if k != "metrica" else 1].replace("@", "-"), s_i)))
        return [_bf_cols(q, a) for q, a in zip(h["q"], h["alive"])]
    if tag.startswith("Gp"):
        out = []
        for b in bench.blocks(fsrc):
            L, W = b["pitch"]
            team = np.asarray(b["team"])
            signs = ss._signs(b["P"], team, np.asarray(b["gk"], bool))
            pos, _, _, _, since = ss._ball(b["B"], b["pitch"])
            pm = np.asarray(b["P"], np.float32)[:, :, :2] * np.array([L, W], np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c0 = np.nanmean(pm[:, team == 0, 0], axis=1)
                c1 = np.nanmean(pm[:, team == 1, 0], axis=1)
                sp = np.zeros(len(pm), np.float32)
                sp[5:-5] = np.nanmean(np.linalg.norm(pm[10:] - pm[:-10], axis=2), axis=1) * 2.5
            bx = (pos[:, 0] - L / 2) / (L / 2)
            f = lambda z: np.nan_to_num((z - L / 2) / (L / 2))  # noqa: E731
            out.append(np.stack([signs[0] * bx, signs[0] * f(c0), signs[1] * bx, signs[1] * f(c1), np.abs(pos[:, 1] - W / 2) / (W / 2), np.nan_to_num(sp) / 5.0, since / 50.0], 1))
        return out
    return None


def half_norm(x):
    """Round 65 (Hn; the idea of round 51): each half's own scale, label-free. Channel logits and heads by the half's spread of the blend's logit; the alive evidence and J1's dead logit to the half's own median and spread."""
    x = x.copy()
    n = ec.N_OWN
    ref = max(np.quantile(np.abs(x[:, 3]), 0.9), 1e-3)
    for c in CH_COLS:
        x[:, c] *= 4.5 / ref
        x[:, c + n] *= 4.5 / ref
    for c in (2 * n, 2 * n + 1):
        q = np.quantile(x[:, c], (0.1, 0.5, 0.9))
        x[:, c] = (x[:, c] - q[1]) / max(q[2] - q[0], 1e-3) * 2.0
    return x.astype(np.float32)


def _rows_step(k, s_i, step_):
    """exp_crf.inputs at another step."""
    path = os.path.join(OUT, "inputs_step%d_%s_%d.pkl" % (step_, k, s_i))
    if os.path.isfile(path):
        return bench._load(path)
    fsrc, src = ee.V6[k][0], ee.V3[k][1]
    blocks = bench.blocks(fsrc)
    ev = bench._load(os.path.join(OUT, "evidence_%s.pkl" % k))["ev"]
    labs = [np.asarray(g) for g in bench.labels(src, ee.V3[k][2])]
    v, g, o = eg2.parts(src, s_i, True)
    rows = []
    for i, b in enumerate(blocks):
        x = ec.pool(ec.frame_inputs(b, v[i], g[i], o[i], ev[i]), step_)
        rows.append((b["group"], x.astype(np.float32), ec.step_labels(labs[i], step_)[:len(x)], len(labs[i])))
    bench._dump(rows, path)
    bench._blocks.clear()
    bench._labels.clear()
    return rows


def load_rows(s_i, tag):
    """The base rows of exp_crf.inputs, plus a tag's added columns (cached per set and seed)."""
    if canon(tag) == "G2":
        return {k: _rows_step(k, s_i, step_of(tag)) for k in SETS}
    data = {k: bench._load(ec._in_path(k, s_i)) for k in SETS}
    if tag[:2] == "Hn":
        return {k: [(r[0], half_norm(r[1]), r[2], r[3]) for r in rows] for k, rows in data.items()}
    if tag[:2] == "HS":                        # diagnosis: SoccerTrack as a further provider nobody trained the channels on
        d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s_i))
        import exp_st
        labs = [np.asarray(g) for g in bench.labels(exp_st.SRC, "evctrl")]
        data["st"] = [(r[0] if r[0] not in {x[0] for k in SETS for x in data[k]} else "st/%s" % r[0], r[1], ec.step_labels(g)[:len(r[1])], r[3]) for r, g in zip(d["rows"], labs)]
        return data
    if canon(tag) not in ("Gh", "Gs", "Gm", "Gp", "Gf", "Gu"):
        return data
    head = canon(tag)
    for k in SETS:
        path = os.path.join(OUT, "inputs_%s_%s_%d.pkl" % (head, k, s_i))
        if os.path.isfile(path):
            ex = bench._load(path)
        else:
            ex = [ec.pool(np.asarray(z, np.float64).reshape(len(z), -1)).astype(np.float32) for z in extras(head, k, s_i)]
            bench._dump(ex, path)
        rows = []
        for r, z in zip(data[k], ex):
            n = min(len(r[1]), len(z))
            rows.append((r[0], np.hstack([r[1][:n], z[:n]]), r[2][:n], r[3]))
        data[k] = rows
        bench._blocks.clear()
        bench._labels.clear()
    return data


def _save(tag, s_i, kind, k, zs):
    bench._dump(dict(q=[z[0] for z in zs], alive=[z[1] for z in zs]), os.path.join(OUT, "%s_%s_%d_%s.pkl" % (kind, tag, s_i, k)))


def _fold_job(job):
    """One stacker fit in its own process (CRF_WORKERS > 1): on the CPU one thread per fit beats shared threads."""
    s_i, tag, i, hold, seed = job
    if os.environ.get("CRF_WORKERS"):
        torch.set_num_threads(1)
    t0 = time.time()
    device = ec.pick_device()
    data = load_rows(s_i, tag)
    e = e_of(tag)
    net, tail = fit(data, hold, seed, device, e=e, step_=step_of(tag), tag=tag)
    out = {k: {j: score_half(net, r[1], r[3], device, step_of(tag), tag) for j, r in enumerate(data[k]) if r[0] in hold} for k in SETS}
    print("  %s seed %d fold %s: loss %.4f, %.0f s" % (tag, s_i, i, tail, time.time() - t0), flush=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()} if not hold else None
    d_in = next(iter(data.values()))[0][1].shape[1] - ec.N_OWN - e
    return out, state, tail, d_in, e


def _pool_map(fn, jobs):
    workers = int(os.environ.get("CRF_WORKERS", "1"))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, jobs))
    return [fn(j) for j in jobs]


def train(s_i, tag):
    s_i = int(s_i)
    data = load_rows(s_i, tag)
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    res = {k: [[] for _ in data[k]] for k in SETS}
    base = 20261107 + 100 * s_i + 1000 * (ord(tag[-1]) - ord("a") + 1 if tag[-1] in "abc" else 0)     # Gb, Gc: the same input, other initial weights
    jobs = [(s_i, tag, i, {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)]), base + i) for i in range(len(dfl))] + [(s_i, tag, "all", set(), base + 50)]
    done = _pool_map(_fold_job, jobs)
    for d in done[:-1]:
        for k in SETS:
            for j, z in d[0][k].items():
                res[k][j].append(z)
    for k in SETS:
        assert all(len(v) for v in res[k])
        out = []
        for v in res[k]:
            lq = np.mean([ec._logit(z[0]) for z in v], axis=0)
            out.append(((1 / (1 + np.exp(-lq))).astype(np.float32), np.mean([z[1] for z in v], axis=0).astype(np.float32)))
        _save(tag, s_i, "sheld", k, out)
    torch.save(dict(state=done[-1][1], d_in=done[-1][3], e=done[-1][4]), os.path.join(OUT, "sall_%s_%d.pt" % (tag, s_i)))
    print("  %s seed %d all-data model, loss %.4f" % (tag, s_i, done[-1][2]), flush=True)


def lopo_evidence(k):
    """v20's dead-ball evidence with every model trained without provider k (round 55's M6+TD6n A25)."""
    path = os.path.join(OUT, "evidence_lopo_%s.pkl" % k)
    if os.path.isfile(path):
        return bench._load(path)["ev"]
    import exp_deadmix as dm
    import exp_scdead as sd
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    fsrc, src = ee.V6[k][0], ee.V3[k][1]
    b6, g6 = ee.lopo_rules(k, ee.V6, conf)
    m6 = [exp_blend.logit_alive(x) for x in ee.lopo_m("v6", ee.V6, k)]
    tn = ee._lopo_td("tdead6n", fsrc)
    blocks = bench.blocks(fsrc)
    truth = [np.asarray(g) for g in bench.labels(src, dm.CONVS[src][0])]
    cues = sd.view_cues() if k == "sc" else [(None, None)] * len(blocks)
    ev = []
    for i, (b, g) in enumerate(zip(blocks, truth)):
        T = ro.terms(exp_blend.rule_f(b), t, c)
        v_, l_ = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i])) if cues[i][0] is None else (cues[i][0], cues[i][0] & (cues[i][1] >= 4))
        ev.append(0.25 * (T @ b6) + 0.375 * m6[i] + 0.375 * tn[i] + g6[0] * (~v_) + g6[1] * l_)
    bench._dump(dict(ev=[np.asarray(a, np.float32) for a in ev]), path)
    return ev


def _lopo_job(job):
    s_i, tag, j, k = job
    if os.environ.get("CRF_WORKERS"):
        torch.set_num_threads(1)
    t0 = time.time()
    device = ec.pick_device()
    data = load_rows(s_i, tag)
    hold = {r[0] for r in data[k]}
    net, tail = fit(data, hold, 20261207 + 100 * s_i + j, device, e=e_of(tag), step_=step_of(tag), tag=tag)
    ev = lopo_evidence(k)
    zs = []
    raw_rows = bench._load(ec._in_path(k, s_i)) if tag[:2] == "Hn" else None
    for r, a in zip(data[k], ev):
        x = (raw_rows[len(zs)] if raw_rows else r)[1].copy()
        col = ec.pool(np.clip(np.asarray(a, np.float64), -20, 20)[:, None] / 5.0).astype(np.float32)
        n = min(len(x), len(col))
        x = x[:n]
        x[:, 2 * ec.N_OWN] = col[:n, 0]
        if tag[:2] == "Hn":
            x = half_norm(x)
        zs.append(score_half(net, x, r[3], device, step_of(tag), tag))
    _save(tag, s_i, "slopo", k, zs)
    print("  %s seed %d without %s: loss %.4f, %.0f s" % (tag, s_i, k, tail, time.time() - t0), flush=True)


def lopo(s_i, tag, sets=SETS):
    """Each provider out of the stacker and out of the dead-ball models behind its input."""
    s_i = int(s_i)
    for k in sets:
        lopo_evidence(k)           # once, before the workers read it
        bench._blocks.clear()
        bench._labels.clear()
    _pool_map(_lopo_job, [(s_i, tag, SETS.index(k), k) for k in sets])


def _mix(hq, s, w):
    return [1 / (1 + np.exp(-(w * ec._logit(q) + (1 - w) * ec._logit(x)))) for q, x in zip(hq, s)]


def _arms(tags):
    return ["v20"] + ["%s %s" % (t, a) for t in tags for a in ARMS]


def _decode_arms(tag, ev, s50, hq, alive):
    res = {}
    for a, (wq, pen, c_, wa) in ARMS.items():
        evs = [(1 - wa) * np.asarray(e)[:len(x)] + wa * x for e, x in zip(ev, alive)] if wa else ev
        res["%s %s" % (tag, a)] = ec.decode_all(evs, _mix(hq, s50, wq), pen, c_)
    return res


def _held(tag, s_i, k, kind):
    if tag == "HC":
        assert kind == "sheld"
        h = bench._load(os.path.join(OUT, "held_HC_%d_%s.pkl" % (s_i, k)))
        return h["q"], h["alive"]
    h = bench._load(os.path.join(OUT, "%s_%s_%d_%s.pkl" % (kind, tag, s_i, k)))
    return h["q"], h["alive"]


def evaluate(tags, seed_is, kind="sheld", sets=SETS):
    import exp_blend_weight as bw
    import exp_deadmix as dm
    names = _arms(tags)
    cells = {}
    for k in sets:
        fsrc, src = ee.V6[k][0], ee.V3[k][1]
        blocks = bench.blocks(fsrc)
        convs = dm.CONVS[src]
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        ev = lopo_evidence(k) if kind == "slopo" else bench._load(os.path.join(OUT, "evidence_%s.pkl" % k))["ev"]
        for s_i in seed_is:
            v, g, o = eg2.parts(src, s_i, True)
            s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
            arms = {"v20": eg.decode_par(ev, s50)}
            for tag in tags:
                hq, alive = _held(tag, s_i, k, kind)
                arms.update(_decode_arms(tag, ev, s50, hq, alive))
            for name, preds in arms.items():
                preds = [np.asarray(p)[:len(labs[convs[0]][i])] for i, p in enumerate(preds)]
                for c in convs:
                    cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
                if s_i == seed_is[0]:
                    print("  %-8s %-10s dead share %.3f (true %.3f), dead F1 %.3f" % (k, name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(labs[convs[0]]) == 2), dm._dead_f1(preds, labs[convs[0]])), flush=True)
            print("  %s seed %d done" % (k, s_i), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    with open(os.path.join(RUNS, "stack_cells_%s_%s_%s.pkl" % (kind, "-".join(tags), "".join(str(s) for s in seed_is))), "wb") as fh:
        pickle.dump(cells, fh)
    mean, _ = er.summarize(cells, names, ref="v20")
    if len(sets) == len(SETS):
        for cand in names[1:]:
            bw.decide(mean, "v20", cand)


def stinputs(s_i):
    """SoccerTrack, unseen by every model: the channels as in exp_crf.st, kept so that later stackers are quick to score."""
    import exp_deadmix as dm
    import exp_st
    import exp_tdead as etd
    s_i = int(s_i)
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t_, c_ = conf["thresholds"], conf["bounds"]
    b6, g6 = ee.weights_on(ee.V6, conf)
    src = "st@v6"
    blocks = bench.blocks(src)
    truth = [np.asarray(g) for g in bench.labels(exp_st.SRC, "evctrl")]
    path = os.path.join(OUT, "evidence_st.pkl")
    if os.path.isfile(path):
        ev = bench._load(path)["ev"]
    else:
        m6 = bench.load_booster(os.path.join(ee.OUT, "M_v6_all.txt"))
        nets = []
        for seed in (20261104, 20261105, 20261106):
            ck = torch.load(os.path.join(bench.CACHE, "tdead6n", "all_%d.pt" % seed), map_location="cpu", weights_only=False)
            net = etd.Net(ck["n_in"]).to(etd.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            nets.append((net, ck["mean"], ck["scale"]))
        ev = []
        os.environ["TD_NORM"] = "1"
        for i, (b, f, g) in enumerate(zip(blocks, bench.features("base", src), truth)):
            T = ro.terms(exp_blend.rule_f(b), t_, c_)
            vv, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
            a6 = exp_blend.logit_alive(m6.predict(f["XF"], num_threads=bench.THREADS))
            xn = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, f["XF"]) for n_, m_, s_ in nets], axis=0))
            ev.append(0.25 * (T @ b6) + 0.375 * a6 + 0.375 * xn + g6[0] * (~vv) + g6[1] * l)
        os.environ.pop("TD_NORM")
        bench._dump(dict(ev=[np.asarray(a, np.float32) for a in ev]), path)
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))
    raw = exp_st.v10_raw(s_i)
    oo = outs[ss.SEEDS[s_i]]
    rows = []
    s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(raw, gru, oo)], 25)
    for b, r, gg, o, a, g in zip(blocks, raw, gru, oo, ev, truth):
        rows.append((b["group"], ec.pool(ec.frame_inputs(b, r, gg, o, a)).astype(np.float32), None, len(g)))
    bench._dump(dict(rows=rows, s50=[np.asarray(x, np.float32) for x in s50]), os.path.join(OUT, "inputs_st_%d.pkl" % s_i))
    print("  SoccerTrack inputs, seed %d: %d halves" % (s_i, len(rows)), flush=True)


def st(tags, seed_is):
    import exp_st
    names = _arms(tags)
    device = ec.pick_device()
    ev = bench._load(os.path.join(OUT, "evidence_st.pkl"))["ev"]
    blocks = bench.blocks("st@v6")
    labs = {cv: [np.asarray(g) for g in bench.labels(exp_st.SRC, cv)] for cv in exp_st.CONVS}
    cells = {}
    for s_i in seed_is:
        d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s_i))
        s50 = d["s50"]
        arms = {"v20": eg.decode_par(ev, s50)}
        for tag in tags:
            if tag == "HC":
                ck = torch.load(os.path.join(OUT, "all_HC_%d.pt" % s_i), map_location="cpu", weights_only=False)
                net = ec.build_net(ck["d_in"]).to(device)
                net.load_state_dict(ck["state"])
                net.eval()
                zs = [ec.score_half(net, r[1], r[3], "HC", device)[:2] for r in d["rows"]]
            else:
                ck = torch.load(os.path.join(OUT, "sall_%s_%d.pt" % (tag, s_i)), map_location="cpu", weights_only=False)
                net = make_net(tag, ck["d_in"], ck.get("e", 0)).to(device)
                net.load_state_dict(ck["state"])
                net.eval()
                rows_ = d["rows"]
                if canon(tag) == "Gf":
                    h = bench._load(_bf_path("st@v3", s_i))
                    rows_ = []
                    for r, q, a in zip(d["rows"], h["q"], h["alive"]):
                        z = ec.pool(_bf_cols(q, a).astype(np.float64)).astype(np.float32)
                        n = min(len(r[1]), len(z))
                        rows_.append((r[0], np.hstack([r[1][:n], z[:n]]), None, r[3]))
                zs = [score_half(net, half_norm(r[1]) if tag[:2] == "Hn" else r[1], r[3], device, step_of(tag), tag) for r in rows_]
            arms.update(_decode_arms(tag, ev, s50, [z[0] for z in zs], [z[1] for z in zs]))
        for name, preds in arms.items():
            preds = [np.asarray(p)[:len(labs["evctrl"][i])] for i, p in enumerate(preds)]
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "v6", name), []).append(er.by_match(blocks, labs[cv], preds))
    er.summarize(cells, names, ref="v20")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "lopo":
        lopo(sys.argv[2], sys.argv[3], sys.argv[4].split() if len(sys.argv) > 4 else SETS)
    elif cmd == "eval":
        evaluate(sys.argv[2].split(), [int(s) for s in sys.argv[3].split()], ("s" + sys.argv[4]) if len(sys.argv) > 4 else "sheld", sys.argv[5].split() if len(sys.argv) > 5 else SETS)
    elif cmd == "stinputs":
        stinputs(sys.argv[2])
    elif cmd == "st":
        st(sys.argv[2].split(), [int(s) for s in sys.argv[3].split()])
    else:
        raise SystemExit(__doc__)
