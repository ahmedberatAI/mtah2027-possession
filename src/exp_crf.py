"""Round 61 (docs/RAPOR.md): a path model on top of the team channels. The official definition keeps possession with the last team while the ball is free or contested and changes it only when the other team gains control; the channels are trained frame by frame and must name a team on free frames too. Here the held-out outputs of the channels go into a small sequence model trained on the whole path's likelihood:

  states     H control, H free, A control, A free, dead (arm HC); H, A, dead (arm P)
  moves      to the other team only into its control state; free is latent, summed over in the likelihood
  input      5 Hz, one view per team, symmetric by construction
  folds      round 46's seven; provider-balanced batches; target-convention labels

Runs on the local GPU. Aggregate output only.

  python -I src/exp_crf.py inputs                the per-half inputs for each seed
  python -I src/exp_crf.py train SEED_INDEX ARM  the seven folds and the all-data model (ARM P or HC)
  python -I src/exp_crf.py eval SEED_INDEX ...   arms against v20 on the open sets
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
import bench  # noqa: E402
import exp_blend  # noqa: E402
import exp_deglue as eg  # noqa: E402
import exp_edges as ee  # noqa: E402
import exp_gate as eg2  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_rules_open as ro  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import short_specialist as ss  # noqa: E402
sys.argv = _argv

STEP = 5
WIN = 300               # steps per training window (60 s)
STEPS = 2000
BATCH = 16
LR = 1e-3
OUT = os.path.join(bench.CACHE, "crf")
SETS = ("dfl", "metrica", "sc")
CLIP = 6.0
N_OWN = 14              # columns that swap with the other team's view (see view())
_FIX = -1e4


def pick_device():
    """CRF_CPU=1 keeps a run off the GPU (hiding the devices breaks torch's cudnn check on CPU tensors)."""
    return "cpu" if os.environ.get("CRF_CPU") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")


def _logit(p):
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
    fill = lambda x: np.where(np.isfinite(x), np.log1p(np.clip(x, 0, 30)), np.log1p(30.0))
    return fill(dh), fill(da), n3h / 3.0, n3a / 3.0, np.clip(sp, 0, 40) / 10.0, seen.astype(np.float32)


def frame_inputs(b, v, g, o, a):
    """(T, N) float32: columns in pairs for the two views. Order: own columns for H, then for A, then common."""
    o = np.asarray(o, np.float32)
    n = min(len(v), len(g), len(o), len(a), len(b["P"]))
    v, g, o, a = np.asarray(v[:n], np.float64), np.asarray(g[:n], np.float64), o[:n], np.asarray(a[:n], np.float64)
    pj = np.asarray(ss.pk(o, 0), np.float64)
    blend = exp_smooth.smooth([0.25 * v + 0.25 * g + 0.5 * pj], 25)[0]
    lv, lg, lj, ls = _logit(v), _logit(g), _logit(pj), _logit(blend)
    dh, da, nh, na, sp, seen = (x[:n] for x in track_cols(b))
    gain_h, gain_a, lose_h, lose_a, touch_h, touch_a = (_logit(o[:, c]) for c in (3, 4, 5, 6, 7, 8))
    own_h = [lv, lg, lj, ls, gain_h, gain_a, lose_h, lose_a, touch_h, touch_a, dh, da, nh, na]
    own_a = [-lv, -lg, -lj, -ls, gain_a, gain_h, lose_a, lose_h, touch_a, touch_h, da, dh, na, nh]
    com = [np.clip(a, -20, 20) / 5.0, _logit(o[:, 2]), sp, seen]
    return np.stack(own_h + own_a + com, 1).astype(np.float32)


def pool(x, step=STEP):
    n = len(x) // step
    return x[:n * step].reshape(n, step, -1).mean(1)


def step_labels(lab, step=STEP):
    lab = np.asarray(lab)
    n = len(lab) // step
    y = lab[:n * step].reshape(n, step)
    out = np.full(n, -1, np.int64)
    for k in (0, 1, 2):
        out[(y == k).sum(1) >= 3] = k
    return out


def view(x, side):
    """x (B, T, N) -> one team's view: its own columns then the common ones."""
    own = x[..., :N_OWN] if side == 0 else x[..., N_OWN:2 * N_OWN]
    return torch.cat([own, x[..., 2 * N_OWN:]], -1)


def build_net(d_in, width=64, drop=0.1):
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
            return self.one(view(x, 0)), self.one(view(x, 1))
    return Net()


def transitions(net, arm):
    """Log potentials (S, S). HC: 0 Hc, 1 Hf, 2 Ac, 3 Af, 4 D. P: 0 H, 1 A, 2 D."""
    t = net.t
    if arm == "P":
        A = torch.zeros((3, 3), device=t.device)
        A = A + torch.stack([torch.stack([0 * t[0], t[2], t[4]]), torch.stack([t[2], 0 * t[0], t[4]]), torch.stack([t[6], t[6], 0 * t[0]])])
        return A
    fx = torch.full((), _FIX, device=t.device)
    z = 0 * t[0]
    return torch.stack([
        torch.stack([z, t[0], t[2], fx, t[4]]),
        torch.stack([t[1], z, t[3], fx, t[5]]),
        torch.stack([t[2], fx, z, t[0], t[4]]),
        torch.stack([t[3], fx, t[1], z, t[5]]),
        torch.stack([t[6], t[7], t[6], t[7], z])])


def emissions(oh, oa, arm):
    d = 0.5 * (oh[..., 2] + oa[..., 2])
    if arm == "P":
        return torch.stack([oh[..., 0], oa[..., 0], d], -1)
    return torch.stack([oh[..., 0], oh[..., 1], oa[..., 0], oa[..., 1], d], -1)


def allowed(y, arm, device):
    """(B, T, S) additive mask for the labelled path: 0 where the state agrees with the label."""
    groups = {"P": ((0,), (1,), (2,)), "HC": ((0, 1), (2, 3), (4,))}[arm]
    S = sum(len(g) for g in groups)
    m = torch.full(y.shape + (S,), _FIX, device=device)
    for k, g in enumerate(groups):
        for s in g:
            m[..., s][y == k] = 0.0
    m[y < 0] = 0.0
    return m


@torch.jit.script
def forward(e: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """log Z over the paths: e (B, T, S), A (S, S)."""
    al = e[:, 0]
    for t in range(1, e.shape[1]):
        al = torch.logsumexp(al.unsqueeze(2) + A.unsqueeze(0), dim=1) + e[:, t]
    return torch.logsumexp(al, dim=1)


def nll(net, x, y, arm):
    oh, oa = net(x)
    e = emissions(oh, oa, arm)
    A = transitions(net, arm)
    B = len(x)
    z = forward(torch.cat([e, e + allowed(y, arm, x.device)]), A)
    return (z[:B] - z[B:]).sum() / (B * x.shape[1])


def posterior(net, x, arm):
    """Per step of a whole half (CPU-friendly loops): log P(H), log P(A), log P(D), and the Viterbi label path."""
    with torch.no_grad():
        oh, oa = net(x)
        e = emissions(oh, oa, arm)[0]
        A = transitions(net, arm)
        T, S = e.shape
        fw = torch.zeros((T, S), device=e.device)
        bw = torch.zeros((T, S), device=e.device)
        vit = torch.zeros((T, S), device=e.device)
        back = torch.zeros((T, S), dtype=torch.long, device=e.device)
        fw[0] = e[0]
        vit[0] = e[0]
        for t in range(1, T):
            fw[t] = torch.logsumexp(fw[t - 1].unsqueeze(1) + A, dim=0) + e[t]
            m, idx = (vit[t - 1].unsqueeze(1) + A).max(dim=0)
            vit[t] = m + e[t]
            back[t] = idx
        for t in range(T - 2, -1, -1):
            bw[t] = torch.logsumexp(A + (e[t + 1] + bw[t + 1]).unsqueeze(0), dim=1)
        post = fw + bw
        post = post - torch.logsumexp(post, dim=1, keepdim=True)
        path = torch.zeros(T, dtype=torch.long, device=e.device)
        path[-1] = vit[-1].argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
    groups = {"P": ((0,), (1,), (2,)), "HC": ((0, 1), (2, 3), (4,))}[arm]
    lp = torch.stack([torch.logsumexp(post[:, list(g)], dim=1) for g in groups], 1).cpu().numpy()
    lab = np.zeros(S, np.int8)
    for k, g in enumerate(groups):
        lab[list(g)] = k
    return lp, lab[path.cpu().numpy()]


def _in_path(k, s_i):
    return os.path.join(OUT, "inputs_%s_%d.pkl" % (k, s_i))


def evidence(k):
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    fsrc = ee.V6[k][0]
    tr6, vi6 = ee.rule_evidence(k, ee.V6, conf)
    m6 = [exp_blend.logit_alive(x) for x in bench._load(os.path.join(ee.OUT, "M_v6_%s.pkl" % fsrc.replace("@", "-")))]
    tn = ee._held("tdead6n", fsrc)
    return [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, tn, vi6)]


def inputs(seed_is=None):
    os.makedirs(OUT, exist_ok=True)
    seed_is = range(len(es.SEEDS)) if seed_is is None else seed_is
    for k in SETS:
        fsrc, src = ee.V6[k][0], ee.V3[k][1]
        blocks = bench.blocks(fsrc)
        ev = evidence(k)
        labs = [np.asarray(g) for g in bench.labels(src, ee.V3[k][2])]
        groups = [b["group"] for b in blocks]
        bench._dump(dict(ev=[np.asarray(a, np.float32) for a in ev]), os.path.join(OUT, "evidence_%s.pkl" % k))
        for s_i in seed_is:
            v, g, o = eg2.parts(src, s_i, True)
            rows = []
            for i, b in enumerate(blocks):
                x = pool(frame_inputs(b, v[i], g[i], o[i], ev[i]))
                rows.append((groups[i], x.astype(np.float32), step_labels(labs[i])[:len(x)], len(labs[i])))
            bench._dump(rows, _in_path(k, s_i))
            print("  %s seed %d: %d halves, %d columns" % (k, s_i, len(rows), rows[0][1].shape[1]), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


def load_rows(s_i):
    return {k: bench._load(_in_path(k, s_i)) for k in SETS}


def fit(data, hold, seed, arm, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = {k: [r for r in rows if r[0] not in hold] for k, rows in data.items()}
    keys = [k for k in pools if pools[k]]
    d_in = next(iter(data.values()))[0][1].shape[1] - N_OWN
    net = build_net(d_in).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    net.train()
    tail = []
    for step in range(STEPS):
        X, Y = [], []
        for b in range(BATCH):
            rows = pools[keys[b % len(keys)]]
            _, x, y, _ = rows[int(rng.integers(len(rows)))]
            a = int(rng.integers(0, max(1, len(x) - WIN)))
            X.append(x[a:a + WIN])
            Y.append(y[a:a + WIN])
        L = min(len(v) for v in X)
        X = torch.as_tensor(np.stack([v[:L] for v in X]), dtype=torch.float32, device=device)
        Y = torch.as_tensor(np.stack([v[:L] for v in Y]), dtype=torch.long, device=device)
        loss = nll(net, X, Y, arm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        tail.append(float(loss))
    net.eval()
    return net, float(np.mean(tail[-200:]))


def score_half(net, x, n, arm, device, step=STEP):
    """(q at 25 fps: P(H | live) as a probability, alive log-odds at 25 fps, Viterbi labels at 25 fps)"""
    lp, path = posterior(net, torch.as_tensor(x[None], dtype=torch.float32, device=device), arm)
    grid = (np.arange(len(lp)) + 0.5) * step
    t = np.arange(n) + 0.5
    lq = lp[:, 0] - lp[:, 1]
    q = 1 / (1 + np.exp(-np.interp(t, grid, lq)))
    alive = np.logaddexp(lp[:, 0], lp[:, 1]) - lp[:, 2]
    lab = np.repeat(path, step)
    lab = np.r_[lab, np.full(max(0, n - len(lab)), lab[-1])][:n]
    return q.astype(np.float32), np.interp(t, grid, alive).astype(np.float32), lab.astype(np.int8)


def _fold_job(job):
    """One stacker fit in its own process (CRF_WORKERS > 1): on the CPU one thread per fit beats shared threads (a step takes 0.45 s with one, 0.9 s with six)."""
    s_i, arm, i, hold, seed = job
    if os.environ.get("CRF_WORKERS"):
        torch.set_num_threads(1)
    t0 = time.time()
    device = pick_device()
    data = load_rows(s_i)
    net, tail = fit(data, hold, seed, arm, device)
    out = {k: {j: score_half(net, r[1], r[3], arm, device) for j, r in enumerate(data[k]) if r[0] in hold} for k in SETS}
    print("  seed %d arm %s fold %s: nll %.4f, %.0f s, moves %s" % (s_i, arm, i, tail, time.time() - t0, np.round(net.t.detach().cpu().numpy(), 2).tolist()), flush=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()} if not hold else None
    return out, state, tail


def train(s_i, arm):
    s_i = int(s_i)
    data = load_rows(s_i)
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    res = {k: [[] for _ in data[k]] for k in SETS}
    base = 20261106 + 100 * s_i
    jobs = [(s_i, arm, i, {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)]), base + i) for i in range(len(dfl))] + [(s_i, arm, "all", set(), base + 50)]
    workers = int(os.environ.get("CRF_WORKERS", "1"))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_fold_job, jobs))
    else:
        done = [_fold_job(j) for j in jobs]
    for out, _, _ in done[:-1]:
        for k in SETS:
            for j, z in out[k].items():
                res[k][j].append(z)
    for k in SETS:
        assert all(len(v) for v in res[k])
        out = []
        for v in res[k]:
            lq = np.mean([np.log(np.clip(z[0], 1e-4, 1 - 1e-4)) - np.log(1 - np.clip(z[0], 1e-4, 1 - 1e-4)) for z in v], axis=0)
            out.append((1 / (1 + np.exp(-lq))).astype(np.float32))
        bench._dump(dict(q=out, alive=[np.mean([z[1] for z in v], axis=0).astype(np.float32) for v in res[k]], path=[v[0][2] for v in res[k]]), os.path.join(OUT, "held_%s_%d_%s.pkl" % (arm, s_i, k)))
    torch.save(dict(state=done[-1][1], d_in=next(iter(data.values()))[0][1].shape[1] - N_OWN, arm=arm), os.path.join(OUT, "all_%s_%d.pt" % (arm, s_i)))
    print("  seed %d arm %s all-data model done, nll %.4f" % (s_i, arm, done[-1][2]), flush=True)


def decode_all(a_list, q_list, pen_team=12.0, c_=48.0, pen_dead=30.0):
    from concurrent.futures import ThreadPoolExecutor
    import decode2
    import exp_v7
    sp = decode2.duration_spec(pen_team, pen_dead, 8, 50, exp_v7.ramp(175, c_))

    def one(i):
        q = np.clip(q_list[i], 1e-4, 1 - 1e-4)
        return sp.decode(np.stack([a_list[i] / 2 + np.log(q), a_list[i] / 2 + np.log(1 - q), -a_list[i] / 2], 1)).astype(np.int8)
    with ThreadPoolExecutor(int(os.environ.get("DECODE_WORKERS", "6"))) as pool_:
        return list(pool_.map(one, range(len(a_list))))


LOW = ("qH.5 6/24", "qH.5 8/24", "qH.5 4/24")      # round 61's added pre-registration: 6/24 decides, the neighbours show flatness
PEN = {"qH.5 6/24": (6.0, 24.0), "qH.5 8/24": (8.0, 24.0), "qH.5 4/24": (4.0, 24.0)}


def _arms(seed_is):
    """qP failed round 61's screen on seed 0, so seeds 1-2 train the HC arm only."""
    have_p = all(os.path.exists(os.path.join(OUT, "all_P_%d.pt" % s)) for s in seed_is)
    return ("v20", "qP", "qH", "qH.5", "qHl", "VH") + LOW if have_p else ("v20", "qH", "qH.5", "qHl", "VH") + LOW


def evaluate(seed_is):
    import exp_blend_weight as bw
    names = _arms(seed_is)
    cells = {}
    os.makedirs(os.path.join(bench.ROOT, "tools", "runs"), exist_ok=True)
    for k in SETS:
        fsrc, src = ee.V6[k][0], ee.V3[k][1]
        blocks = bench.blocks(fsrc)
        import exp_deadmix as dm
        convs = dm.CONVS[src]
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        ev = bench._load(os.path.join(OUT, "evidence_%s.pkl" % k))["ev"]
        for s_i in seed_is:
            v, g, o = eg2.parts(src, s_i, True)
            s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
            hh = bench._load(os.path.join(OUT, "held_HC_%d_%s.pkl" % (s_i, k)))
            half = [1 / (1 + np.exp(-0.5 * (_logit(q) + _logit(s)))) for q, s in zip(hh["q"], s50)]
            arms = {"v20": eg.decode_par(ev, s50), "qH": decode_all(ev, hh["q"]), "qH.5": decode_all(ev, half), "qHl": decode_all(ev, hh["q"], 6.0, 24.0), "VH": hh["path"]}
            for name in LOW:
                arms[name] = decode_all(ev, half, *PEN[name])
            if "qP" in names:
                arms["qP"] = decode_all(ev, bench._load(os.path.join(OUT, "held_P_%d_%s.pkl" % (s_i, k)))["q"])
            for name, preds in arms.items():
                preds = [np.asarray(p)[:len(labs[convs[0]][i])] for i, p in enumerate(preds)]
                for c in convs:
                    cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
                if s_i == seed_is[0]:
                    n = sum(np.sum(np.r_[True, np.asarray(p)[1:] != np.asarray(p)[:-1]] & (np.asarray(p) < 2)) for p in preds)
                    live = sum(np.sum(np.asarray(p) < 2) for p in preds) / 1500.0
                    print("  %-8s %-5s spells per live minute %.2f" % (k, name, n / max(live, 1e-9)), flush=True)
            print("  %s seed %d done" % (k, s_i), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    with open(os.path.join(bench.ROOT, "tools", "runs", "r61_cells_%s.pkl" % "_".join(str(s) for s in seed_is)), "wb") as fh:
        pickle.dump(cells, fh)
    mean, _ = er.summarize(cells, list(names), ref="v20")
    for cand in names[1:]:
        bw.decide(mean, "v20", cand)


def st(seed_is):
    """SoccerTrack, unseen by every model: all-data stacker, channels as in exp_edges.final_st."""
    import exp_deadmix as dm
    import exp_st
    import exp_tdead as etd
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t_, c_ = conf["thresholds"], conf["bounds"]
    b6, g6 = ee.weights_on(ee.V6, conf)
    m6 = bench.load_booster(os.path.join(ee.OUT, "M_v6_all.txt"))
    nets = []
    for seed in (20261104, 20261105, 20261106):
        ck = torch.load(os.path.join(bench.CACHE, "tdead6n", "all_%d.pt" % seed), map_location="cpu", weights_only=False)
        net = etd.Net(ck["n_in"]).to(etd.DEVICE)
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append((net, ck["mean"], ck["scale"]))
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))
    labs = {cv: [np.asarray(g) for g in bench.labels(exp_st.SRC, cv)] for cv in exp_st.CONVS}
    truth = labs["evctrl"]
    src = "st@v6"
    blocks = bench.blocks(src)
    ev = []
    os.environ["TD_NORM"] = "1"
    for i, (b, f, g) in enumerate(zip(blocks, bench.features("base", src), truth)):
        T = ro.terms(exp_blend.rule_f(b), t_, c_)
        vv, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        a6 = exp_blend.logit_alive(m6.predict(f["XF"], num_threads=bench.THREADS))
        xn = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, f["XF"]) for n_, m_, s_ in nets], axis=0))
        ev.append(0.25 * (T @ b6) + 0.375 * a6 + 0.375 * xn + g6[0] * (~vv) + g6[1] * l)
    os.environ.pop("TD_NORM")
    device = pick_device()
    names = _arms(seed_is)
    cells = {}
    for s_i in seed_is:
        raw = exp_st.v10_raw(s_i)
        oo = outs[ss.SEEDS[s_i]]
        s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(raw, gru, oo)], 25)
        models = {}
        for arm in ("P", "HC") if "qP" in names else ("HC",):
            ck = torch.load(os.path.join(OUT, "all_%s_%d.pt" % (arm, s_i)), map_location="cpu", weights_only=False)
            net = build_net(ck["d_in"]).to(device)
            net.load_state_dict(ck["state"])
            net.eval()
            models[arm] = net
        res = {arm: [] for arm in models}
        for b, r, gg, o, a, g in zip(blocks, raw, gru, oo, ev, truth):
            x = pool(frame_inputs(b, r, gg, o, a)).astype(np.float32)
            for arm in res:
                res[arm].append(score_half(models[arm], x, len(g), arm, device))
        half = [1 / (1 + np.exp(-0.5 * (_logit(z[0]) + _logit(q)))) for z, q in zip(res["HC"], s50)]
        arms = {"v20": eg.decode_par(ev, s50), "qH": decode_all(ev, [z[0] for z in res["HC"]]), "qH.5": decode_all(ev, half), "qHl": decode_all(ev, [z[0] for z in res["HC"]], 6.0, 24.0), "VH": [z[2] for z in res["HC"]]}
        for name in LOW:
            arms[name] = decode_all(ev, half, *PEN[name])
        if "qP" in names:
            arms["qP"] = decode_all(ev, [z[0] for z in res["P"]])
        for name, preds in arms.items():
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "v6", name), []).append(er.by_match(blocks, labs[cv], preds))
    er.summarize(cells, list(names), ref="v20")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "inputs":
        inputs([int(s) for s in sys.argv[2:]] or None)
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "eval":
        evaluate([int(s) for s in sys.argv[2:]])
    elif cmd == "st":
        st([int(s) for s in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)
