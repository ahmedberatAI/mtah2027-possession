"""Round 66 (docs/RAPOR.md): a channel that never sees the ball. The three team channels share their errors (round 48: false 4-8 s spells are believed by all of them), all three read the ball. This one reads only the players: every player of both teams as a set per step, 5 Hz, a bidirectional GRU over a minute, and learns who controls the ball from PFF's controlled reading; it has never seen DFL, Metrica, SkillCorner, SoccerTrack or BAR-ATH, so it scores all of them out of sample. An auxiliary head guesses where the ball is. Aggregate output only.

  python -I src/exp_ballfree.py prep [WORKERS]        PFF halves (sim.V3) -> per-team player tokens at 5 Hz
  python -I src/exp_ballfree.py train SEED_INDEX      one model (BF_CPU=1 keeps it off the GPU)
  python -I src/exp_ballfree.py infer SRC SEED_INDEX  its outputs on a bench source
  python -I src/exp_ballfree.py audit SEED_INDEX      frame accuracy on the open sets next to the other channels
  python -I src/exp_ballfree.py eval "SEEDS"          the channel in v20's blend at weights 0.1, 0.2, 0.3
"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP = 5
WIN = int(os.environ.get("BF_WIN", "300"))
CORE = WIN // 2
M = 14                 # players kept per team and step
FPS = 25
SEEDS = (20261106, 20261107, 20261108, 20261109, 20261110, 20261111)
STEPS = int(os.environ.get("BF_STEPS", "12000"))
BATCH = int(os.environ.get("BF_BATCH", "32"))
LR = 1e-3
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "cache", "ballfree")
PREP = os.path.join(OUT, "pff")
C1_PREP = os.path.join(ROOT, "cache", "kpu", "prep_c1")
TAG = os.environ.get("BF_TAG", "")      # dose test: a run with another amount of training keeps its own files


def tokens(P, B, team, gk, pitch):
    """(T, 2, M, 6) float32 at 5 Hz: for each team, up to M present players: x, y from the centre (m), velocity (m/s), goalkeeper, present. Also the ball's position (T, 2, NaN when unseen) and the side signs."""
    import short_specialist as ss
    L, W = pitch
    team = np.asarray(team)
    gk = np.asarray(gk, bool)
    Pm = np.asarray(P[:, :, :2], np.float64) * np.array(pitch)
    n = len(Pm)
    V = np.zeros_like(Pm)
    V[2:-2] = (Pm[4:] - Pm[:-4]) * FPS / 4.0
    V = np.clip(np.nan_to_num(V), -12.0, 12.0)
    idx = np.arange(STEP // 2, n - STEP // 2, STEP)
    Pc = Pm[idx] - np.array([L / 2, W / 2])
    Vc = V[idx]
    T = len(idx)
    out = np.zeros((T, 2, M, 6), np.float32)
    for t_ in (0, 1):
        sl = np.flatnonzero(team == t_)
        ok = np.isfinite(Pc[:, sl, 0])
        order = np.argsort(~ok, axis=1, kind="stable")[:, :M]
        rows = np.arange(T)[:, None]
        pick = sl[order]
        have = ok[rows, order]
        k = pick.shape[1]
        out[:, t_, :k, 0] = np.nan_to_num(Pc[rows, pick, 0])
        out[:, t_, :k, 1] = np.nan_to_num(Pc[rows, pick, 1])
        out[:, t_, :k, 2] = Vc[rows, pick, 0]
        out[:, t_, :k, 3] = Vc[rows, pick, 1]
        out[:, t_, :k, 4] = gk[pick]
        out[:, t_, :k, 5] = have
        out[:, t_, :k, :5] *= have[:, :, None]
    ball = np.asarray(B[idx, :2], np.float64) * np.array(pitch) - np.array([L / 2, W / 2])
    return out, ball.astype(np.float32), np.array(ss._signs(P, team, gk), np.float32)


def step_labels(lab):
    lab = np.asarray(lab)
    idx = np.arange(STEP // 2, len(lab) - STEP // 2, STEP)
    y = np.stack([lab[idx + d] for d in range(-(STEP // 2), STEP // 2 + 1)], 1)
    out = np.full(len(idx), -1, np.int64)
    for k in (0, 1, 2):
        out[(y == k).sum(1) >= STEP // 2 + 1] = k
    return out


def _prep_one(job):
    mi, match = job
    import gradient
    import sim
    warnings.filterwarnings("ignore")
    t0 = time.time()
    keys = []
    path, info = gradient.parse(match)
    with np.load(path) as d:
        pitch = tuple(float(v) for v in d["pitch"])
        for per in d["periods"]:
            tag = "h%d" % per
            cp = os.path.join(C1_PREP, "%s_%s_v3_ctrl.npy" % (match, tag))
            if not os.path.isfile(cp):
                continue
            key = "%s_%s" % (match, tag)
            dest = os.path.join(PREP, key + ".npz")
            keys.append(key)
            if os.path.isfile(dest):
                continue
            ctrl = np.load(cp)
            lab = np.where(ctrl < 0, 3, ctrl)
            Pn, Bn = sim.degrade(d[tag + "_players"], d[tag + "_ball"], lab, pitch, seed=30000 + 10 * mi + int(per), params=sim.V3)
            tok, ball, signs = tokens(Pn[:, :, :2].astype(np.float32), Bn[:, :2].astype(np.float32), d["team"], d["gk"], pitch)
            np.savez(dest, tok=tok.astype(np.float16), ball=ball.astype(np.float16), signs=signs, y=step_labels(lab).astype(np.int8))
    print("prepared %s: %d halves, %.0f s" % (match, len(keys), time.time() - t0), flush=True)
    return keys


def prep(workers=8):
    from concurrent.futures import ProcessPoolExecutor
    import gradient
    os.makedirs(PREP, exist_ok=True)
    keys = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for part in pool.map(_prep_one, list(enumerate(gradient.matches()))):
            keys += part
    with open(os.path.join(C1_PREP, "index.json")) as fh:
        val = json.load(fh)["val_matches"]
    with open(os.path.join(PREP, "index.json"), "w") as fh:
        json.dump({"keys": sorted(keys), "val_matches": val}, fh)
    print("index: %d halves; validation matches %d" % (len(keys), len(val)), flush=True)


# -- model -------------------------------------------------------------------------------------------------------------

def build_net(emb=64, hid=128, heads=4):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(10, emb), nn.GELU(), nn.Linear(emb, emb), nn.GELU())
            self.att = nn.MultiheadAttention(emb, heads, batch_first=True)
            self.norm = nn.LayerNorm(emb)
            self.gru = nn.GRU(4 * emb, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hid, 2)        # own control, dead
            self.ball = nn.Linear(2 * hid, 2)        # auxiliary: the ball in this side's frame

        def side(self, x, g, s):
            """x (B, T, 2, M, 6), g (B, 2): side s's view -> (B, T, 4 emb)."""
            B, T = x.shape[:2]
            own, opp = x[:, :, s], x[:, :, 1 - s]
            gs = g[:, s].view(-1, 1, 1, 1)
            scale = torch.tensor([52.5, 34.0, 8.0, 8.0], device=x.device)
            po, pp = own[..., :2] * gs, opp[..., :2] * gs
            vo, vp = own[..., 2:4] * gs, opp[..., 2:4] * gs
            mo, mp = own[..., 5] > 0.5, opp[..., 5] > 0.5
            d = torch.linalg.norm(po.unsqueeze(3) - pp.unsqueeze(2), dim=-1)
            d = d.masked_fill(~(mo.unsqueeze(3) & mp.unsqueeze(2)), 1e3)

            def tok(p, v, m, other, dist, flag, gk):
                near, idx = dist.min(-1)
                to = torch.gather(other, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2)) - p
                unit = to / near.clamp(min=0.1).unsqueeze(-1)
                closing = (v * unit).sum(-1) / 5.0
                return torch.cat([torch.cat([p, v], -1) / scale, (torch.linalg.norm(v, dim=-1, keepdim=True) / 8.0), gk.unsqueeze(-1), (near.clamp(max=30.0) / 10.0).unsqueeze(-1), closing.unsqueeze(-1), torch.full_like(near, flag).unsqueeze(-1), m.float().unsqueeze(-1)], -1)
            to = tok(po, vo, mo, pp, d, 1.0, own[..., 4])
            tp = tok(pp, vp, mp, po, d.transpose(2, 3), 0.0, opp[..., 4])
            t = self.phi(torch.cat([to, tp], 2)).reshape(B * T, 2 * M, -1)
            mask = torch.cat([mo, mp], 2).reshape(B * T, 2 * M)
            empty = ~mask.any(1)
            if empty.any():
                mask = mask.clone()
                mask[empty, 0] = True
            a, _ = self.att(t, t, t, key_padding_mask=~mask, need_weights=False)
            t = self.norm(t + a)
            w = mask.unsqueeze(-1).float()
            own_t, opp_t = t[:, :M], t[:, M:]
            own_w, opp_w = w[:, :M], w[:, M:]
            mean_o = (own_t * own_w).sum(1) / own_w.sum(1).clamp(min=1)
            mean_p = (opp_t * opp_w).sum(1) / opp_w.sum(1).clamp(min=1)
            max_o = (own_t - 1e3 * (1 - own_w)).max(1).values
            max_p = (opp_t - 1e3 * (1 - opp_w)).max(1).values
            return torch.cat([mean_o, max_o, mean_p, max_p], -1).reshape(B, T, -1)

        def forward(self, x, g):
            h0, _ = self.gru(self.side(x, g, 0))
            h1, _ = self.gru(self.side(x, g, 1))
            o0, o1 = self.head(h0), self.head(h1)
            ctrl = torch.stack([o0[..., 0], o1[..., 0], 0.5 * (o0[..., 1] + o1[..., 1])], -1)
            return ctrl, self.ball(h0), self.ball(h1)
    return Net()


class Data:
    def __init__(self, keys):
        self.d = []
        for k in keys:
            with np.load(os.path.join(PREP, k + ".npz")) as z:
                self.d.append((z["tok"].astype(np.float32), z["ball"].astype(np.float32), z["signs"].astype(np.float32), z["y"].astype(np.int64)))
        self.w = np.array([len(x[3]) for x in self.d], np.float64)
        self.w /= self.w.sum()

    def batch(self, rng, size):
        X, Bb, G, Y = [], [], [], []
        for _ in range(size):
            i = int(rng.choice(len(self.d), p=self.w))
            tok, ball, signs, y = self.d[i]
            a = int(rng.integers(0, max(1, len(y) - WIN)))
            x = tok[a:a + WIN].copy()
            bb = ball[a:a + WIN].copy()
            if rng.integers(2):                  # mirror the pitch in y
                x[..., 1] *= -1
                x[..., 3] *= -1
                bb[..., 1] *= -1
            X.append(x)
            Bb.append(bb)
            G.append(signs)
            Y.append(y[a:a + WIN])
        L = min(len(v) for v in Y)
        return (np.stack([v[:L] for v in X]), np.stack([v[:L] for v in Bb]), np.stack(G), np.stack([v[:L] for v in Y]))


def device():
    import torch
    return "cpu" if os.environ.get("BF_CPU") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(net, x, bb, g, y):
    import torch
    import torch.nn.functional as F
    ctrl, b0, b1 = net(x, g)
    ok = y >= 0
    edge = WIN // 12
    ok[:, :edge] = False
    ok[:, -edge:] = False
    lc = F.cross_entropy(ctrl[ok], y[ok])
    seen = torch.isfinite(bb[..., 0])
    bb = torch.nan_to_num(bb)
    lb = 0.5 * (F.smooth_l1_loss(b0[seen] * 10.0, bb[seen] * g[:, 0].view(-1, 1, 1).expand_as(bb)[seen]) + F.smooth_l1_loss(b1[seen] * 10.0, bb[seen] * g[:, 1].view(-1, 1, 1).expand_as(bb)[seen]))
    acc_ok = ok & (y < 2)
    pred = ctrl[..., :2].argmax(-1)
    acc = (pred[acc_ok] == y[acc_ok]).float().mean()
    return lc + 0.02 * lb, lc, lb, acc


def train(s_i):
    import torch
    s_i = int(s_i)
    seed = SEEDS[s_i]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with open(os.path.join(PREP, "index.json")) as fh:
        idx = json.load(fh)
    val = set(idx["val_matches"])
    dt = Data([k for k in idx["keys"] if k.split("_")[0] not in val])
    dv = Data([k for k in idx["keys"] if k.split("_")[0] in val])
    dev = device()
    net = build_net().to(dev)
    print("  %d training halves, %d validation halves, %d parameters, device %s" % (len(dt.d), len(dv.d), sum(p.numel() for p in net.parameters()), dev), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    vrng = np.random.default_rng(7)
    vb = [dv.batch(vrng, 16) for _ in range(8)]
    best, t0 = np.inf, time.time()
    path = os.path.join(OUT, "bf%s_s%d.pt" % (TAG, seed))
    for step in range(1, STEPS + 1):
        net.train()
        x, bb, g, y = dt.batch(rng, BATCH)
        loss, _, _, _ = loss_fn(net, torch.from_numpy(x).to(dev), torch.from_numpy(bb).to(dev), torch.from_numpy(g).to(dev), torch.from_numpy(y).to(dev))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 500 == 0 or step == STEPS:
            net.eval()
            res = []
            with torch.no_grad():
                for x, bb, g, y in vb:
                    res.append([float(v) for v in loss_fn(net, torch.from_numpy(x).to(dev), torch.from_numpy(bb).to(dev), torch.from_numpy(g).to(dev), torch.from_numpy(y).to(dev))])
            r = np.mean(res, axis=0)
            print("  seed %d step %d: validation control %.4f, ball %.3f, team accuracy on live steps %.4f, %.0f s" % (seed, step, r[1], r[2], r[3], time.time() - t0), flush=True)
            if r[1] < best:
                best = r[1]
                torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, seed=seed, step=step), path)


def predict(net, tok, signs, n, dev):
    """(q at 25 fps: P(H | live), alive log-odds at 25 fps) from overlapping windows."""
    import torch
    T = len(tok)
    lp = np.zeros((T, 3), np.float32)
    starts = list(range(-(WIN - CORE) // 2, T, CORE))
    with torch.no_grad():
        for b0 in range(0, len(starts), 16):
            part = starts[b0:b0 + 16]
            xs = np.zeros((len(part), WIN) + tok.shape[1:], np.float32)
            for j, a in enumerate(part):
                lo, hi = max(0, a), min(T, a + WIN)
                xs[j, lo - a:hi - a] = tok[lo:hi]
            g = np.repeat(signs[None], len(part), 0)
            ctrl, _, _ = net(torch.from_numpy(xs).to(dev), torch.from_numpy(g).to(dev))
            o = torch.log_softmax(ctrl, -1).cpu().numpy()
            for j, a in enumerate(part):
                lo = max(0, a + (WIN - CORE) // 2)
                hi = min(T, a + (WIN - CORE) // 2 + CORE)
                if hi > lo:
                    lp[lo:hi] = o[j, lo - a:hi - a]
    grid = STEP // 2 + np.arange(T) * STEP
    t = np.arange(n)
    q = 1 / (1 + np.exp(-np.interp(t, grid, lp[:, 0] - lp[:, 1])))
    alive = np.interp(t, grid, np.logaddexp(lp[:, 0], lp[:, 1]) - lp[:, 2])
    return q.astype(np.float32), alive.astype(np.float32)


def load_model(s_i, dev):
    import torch
    ck = torch.load(os.path.join(OUT, "bf%s_s%d.pt" % (TAG, SEEDS[int(s_i)])), map_location="cpu", weights_only=False)
    net = build_net().to(dev)
    net.load_state_dict(ck["state"])
    return net.eval()


def infer(src, s_i):
    import bench
    dev = device()
    net = load_model(s_i, dev)
    res = []
    for b in bench.blocks(src):
        tok, _, signs = tokens(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        res.append(predict(net, tok, signs, len(b["P"]), dev))
    bench._dump(dict(q=[z[0] for z in res], alive=[z[1] for z in res]), os.path.join(OUT, "held%s_%s_%d.pkl" % (TAG, src.replace("@", "-"), int(s_i))))
    print("  %s seed index %s: %d halves" % (src, s_i, len(res)), flush=True)


def _source(k):
    return {"dfl": "dfl@v3", "metrica": "metrica@v3", "sc": "sc@v3"}[k]


def audit(s_i):
    """Live-frame accuracy of each channel and of the new one; how often it alone is right when it disagrees with all three."""
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_crf as ec
    from exp_crf import bench, ee, eg2, exp_smooth, ss
    sys.argv = _argv
    s_i = int(s_i)
    for k in ec.SETS:
        src = ee.V3[k][1]
        labs = [np.asarray(g) for g in bench.labels(src, ee.V3[k][2])]
        v, g, o = eg2.parts(src, s_i, True)
        h = bench._load(os.path.join(OUT, "held%s_%s_%d.pkl" % (TAG, _source(k).replace("@", "-"), s_i)))
        tot = {}
        dis = np.zeros(2)
        for i, y in enumerate(labs):
            n = min(len(y), len(v[i]), len(h["q"][i]))
            live = y[:n] < 2
            yy = (y[:n] == 0)[live]
            ch = {"v10": np.asarray(v[i])[:n], "BiGRU": np.asarray(g[i])[:n], "J1": ss.pk(np.asarray(o[i], np.float32), 0)[:n], "BF": h["q"][i][:n]}
            ch["blend"] = exp_smooth.smooth([0.25 * ch["v10"] + 0.25 * ch["BiGRU"] + 0.5 * ch["J1"]], 25)[0]
            ch["BF s50"] = exp_smooth.smooth([np.asarray(ch["BF"], np.float64)], 25)[0]
            for name, q in ch.items():
                t = tot.setdefault(name, np.zeros(2))
                t += (np.sum((q[live] > 0.5) == yy), len(yy))
            agree = (ch["blend"][live] > 0.5) != (ch["BF s50"][live] > 0.5)
            dis += (np.sum(((ch["BF s50"][live] > 0.5) == yy) & agree), np.sum(agree))
        print("  %-8s %s | BF against the blend: disagree on %.3f of live frames, right in %.3f of them" % (k, " ".join("%s %.4f" % (n_, t[0] / t[1]) for n_, t in tot.items()), dis[1] / tot["BF"][1], dis[0] / max(dis[1], 1)), flush=True)
        bench._labels.clear()


def evaluate(seed_is):
    """The channel in v20's blend, raw probabilities, before the +-1 s smoothing; the rest as v20."""
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_crf as ec
    from exp_crf import bench, ee, eg, eg2, er, exp_smooth, ss
    sys.argv = _argv
    import exp_blend_weight as bw
    import exp_deadmix as dm
    W = (0.1, 0.2, 0.3)
    names = ["v20"] + ["BF %.1f" % w for w in W]
    cells = {}
    for k in ec.SETS:
        fsrc, src = ee.V6[k][0], ee.V3[k][1]
        blocks = bench.blocks(fsrc)
        convs = dm.CONVS[src]
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        ev = bench._load(os.path.join(ec.OUT, "evidence_%s.pkl" % k))["ev"]
        for s_i in seed_is:
            v, g, o = eg2.parts(src, s_i, True)
            h = bench._load(os.path.join(OUT, "held_%s_%d.pkl" % (_source(k).replace("@", "-"), s_i)))
            raw = [0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)]
            arms = {"v20": eg.decode_par(ev, exp_smooth.smooth(raw, 25))}
            for w in W:
                mix = [(1 - w) * r[:min(len(r), len(q))] + w * np.asarray(q, np.float64)[:min(len(r), len(q))] for r, q in zip(raw, h["q"])]
                arms["BF %.1f" % w] = eg.decode_par(ev, exp_smooth.smooth(mix, 25))
            for name, preds in arms.items():
                preds = [np.asarray(p)[:len(labs[convs[0]][i])] for i, p in enumerate(preds)]
                for c in convs:
                    cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (k, s_i), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    with open(os.path.join(ROOT, "tools", "runs", "r66_cells.pkl"), "wb") as fh:
        pickle.dump(cells, fh)
    mean, _ = er.summarize(cells, names, ref="v20")
    for cand in names[1:]:
        bw.decide(mean, "v20", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(*sys.argv[2:])
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "infer":
        infer(sys.argv[2], sys.argv[3])
    elif cmd == "audit":
        audit(sys.argv[2])
    elif cmd == "eval":
        evaluate([int(s) for s in sys.argv[2].split()])
    else:
        raise SystemExit(__doc__)
