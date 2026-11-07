"""Round 71 (docs/RAPOR.md): the ball-free channel's player set with the ball added, BS. Round 70: half of v21's lost credit is undisputed model error, above all short possessions every reading agrees on, and a different way of mixing today's channels does not reach it (false spells are believed by three or four of the four channels, the mix is calibrated). The channel that never sees the ball is already the best single one on DFL and SkillCorner, so a long view of all players is worth a lot; but no channel reads all players, the ball and a long context together (J1: five players nearest to the ball, +-2-3 s; v10 and BiGRU: hand-made features). Trained on PFF only (controlled reading, sim.V6, so without V3's leak at the restarts); it has never seen DFL, Metrica, SkillCorner, SoccerTrack or BAR-ATH. Aggregate output only.

  python -I src/exp_ballset.py prep [WORKERS]         PFF halves (sim.V6) -> player tokens and ball columns at 5 Hz
  python -I src/exp_ballset.py train SEED_INDEX       one model (BS_CPU=1 keeps it off the GPU)
  python -I src/exp_ballset.py infer SEED_INDEX       its outputs on the open sets
  python -I src/exp_ballset.py audit SEED_INDEX       accuracy and spells alone, next to the other channels
  BS_TAG=J python -I src/exp_ballset.py finetune SEED_INDEX   J1's recipe: fine-tuned on the ten target-convention games, each one's output from a model that never saw it
"""
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP = int(os.environ.get("BS_STEP", "5"))      # frames per step; round 76's dose: 2 (12.5 Hz)
WIN = 60 * 25 // STEP
CORE = WIN // 2
M = 14
FPS = 25
SEEDS = (20261116, 20261117, 20261118)
STEPS = int(os.environ.get("BS_STEPS", "12000"))
BATCH = int(os.environ.get("BS_BATCH", "32"))
LR = 1e-3
GAP_P = 0.25            # share of training windows with a long hole in the ball
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "cache", "ballset")
PREP = os.path.join(OUT, "pff" if STEP == 5 else "pff_s%d" % STEP)
C1_PREP = os.path.join(ROOT, "cache", "kpu", "prep_c1")
TAG = os.environ.get("BS_TAG", "")
READ = os.environ.get("BS_READ", "ctrl")
SOURCES = ("dfl@v6", "metrica@v6g", "sc@v6", "st@v6")


def _fb():
    """exp_ballfree's helpers at this file's rate."""
    import exp_ballfree as bf
    bf.STEP = STEP
    return bf


def ball_cols(B, pitch):
    """(T, 7) float32 at 5 Hz: the ball's last seen position from the centre (m), velocity (m/s), hardest change of velocity within the step, share of the step's frames seen, frames since it was seen."""
    import short_specialist as ss
    pos, vel, acc, seen, since = ss._ball(np.asarray(B, np.float64), pitch)
    n = len(pos)
    L, W = pitch
    idx = np.arange(STEP // 2, n - STEP // 2, STEP)
    around = [idx + d for d in range(-(STEP // 2), STEP // 2 + 1)]
    out = np.zeros((len(idx), 7), np.float32)
    out[:, 0] = pos[idx, 0] - L / 2
    out[:, 1] = pos[idx, 1] - W / 2
    out[:, 2:4] = vel[idx]
    out[:, 4] = np.clip(np.max([acc[i] for i in around], axis=0), 0.0, 400.0)
    out[:, 5] = np.mean([seen[i] for i in around], axis=0)
    out[:, 6] = since[idx]
    return out


def _prep_one(job):
    mi, match = job
    bf = _fb()
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
            Pn, Bn = sim.degrade(d[tag + "_players"], d[tag + "_ball"], lab, pitch, seed=30000 + 10 * mi + int(per), params=sim.V6)
            tok, _, signs = bf.tokens(Pn[:, :, :2].astype(np.float32), Bn[:, :2].astype(np.float32), d["team"], d["gk"], pitch)
            np.savez(dest, tok=tok.astype(np.float16), ball=ball_cols(Bn[:, :2], pitch).astype(np.float16), signs=signs, y=bf.step_labels(lab).astype(np.int8))
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


def prep_touch():
    """Round 78: PFF's touch-level reading (its own labels before the control rule) per step, on the same grid."""
    bf = _fb()
    import gradient
    dest = os.path.join(OUT, "pff_touch")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for match in gradient.matches():
        path, _ = gradient.parse(match)
        with np.load(path) as d:
            for per in d["periods"]:
                tag = "h%d" % per
                key = "%s_%s" % (match, tag)
                if os.path.isfile(os.path.join(PREP, key + ".npz")):
                    np.save(os.path.join(dest, key + ".npy"), bf.step_labels(np.asarray(d[tag + "_labels"])).astype(np.int8))
                    n += 1
    print("touch-level labels: %d halves" % n, flush=True)


def build_net(emb=64, hid=128, heads=4):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(16, emb), nn.GELU(), nn.Linear(emb, emb), nn.GELU())
            self.phi_b = nn.Sequential(nn.Linear(8, emb), nn.GELU(), nn.Linear(emb, emb), nn.GELU())
            self.att = nn.MultiheadAttention(emb, heads, batch_first=True)
            self.norm = nn.LayerNorm(emb)
            self.gru = nn.GRU(7 * emb, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hid, 2)        # own control, dead

        def side(self, x, b, g, s):
            """x (B, T, 2, M, 6), b (B, T, 7), g (B, 2): side s's view -> (B, T, 7 emb)."""
            B, T = x.shape[:2]
            own, opp = x[:, :, s], x[:, :, 1 - s]
            gs = g[:, s].view(-1, 1, 1, 1)
            gb = g[:, s].view(-1, 1, 1)
            scale = torch.tensor([52.5, 34.0, 8.0, 8.0], device=x.device)
            po, pp = own[..., :2] * gs, opp[..., :2] * gs
            vo, vp = own[..., 2:4] * gs, opp[..., 2:4] * gs
            mo, mp = own[..., 5] > 0.5, opp[..., 5] > 0.5
            bp, bv = b[..., :2] * gb, b[..., 2:4] * gb
            seen, since = b[..., 5:6], b[..., 6:7] / 50.0
            d = torch.linalg.norm(po.unsqueeze(3) - pp.unsqueeze(2), dim=-1)
            d = d.masked_fill(~(mo.unsqueeze(3) & mp.unsqueeze(2)), 1e3)

            def tok(p, v, m, other, dist, flag, gk):
                near, idx = dist.min(-1)
                to = torch.gather(other, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2)) - p
                unit = to / near.clamp(min=0.1).unsqueeze(-1)
                closing = (v * unit).sum(-1) / 5.0
                tb = bp.unsqueeze(2) - p
                db = torch.linalg.norm(tb, dim=-1).clamp(min=0.1)
                closing_b = (v * (tb / db.unsqueeze(-1))).sum(-1) / 5.0
                k = p.shape[2]
                f = torch.cat([torch.cat([p, v], -1) / scale, torch.linalg.norm(v, dim=-1, keepdim=True) / 8.0, gk.unsqueeze(-1), (near.clamp(max=30.0) / 10.0).unsqueeze(-1), closing.unsqueeze(-1), torch.full_like(near, flag).unsqueeze(-1), m.float().unsqueeze(-1), (tb / 20.0).clamp(-3.0, 3.0), (db.clamp(max=30.0) / 10.0).unsqueeze(-1), closing_b.unsqueeze(-1), seen.unsqueeze(2).expand(-1, -1, k, -1), since.unsqueeze(2).expand(-1, -1, k, -1)], -1)
                return f, db
            to, db_o = tok(po, vo, mo, pp, d, 1.0, own[..., 4])
            tp, db_p = tok(pp, vp, mp, po, d.transpose(2, 3), 0.0, opp[..., 4])
            sp = torch.linalg.norm(bv, dim=-1, keepdim=True)
            tb = self.phi_b(torch.cat([bp / torch.tensor([52.5, 34.0], device=x.device), bv / 15.0, sp / 15.0, b[..., 4:5] / 100.0, seen, since], -1)).reshape(B * T, 1, -1)
            t = self.phi(torch.cat([to, tp], 2)).reshape(B * T, 2 * M, -1)
            mask = torch.cat([mo, mp], 2).reshape(B * T, 2 * M)
            all_t = torch.cat([t, tb], 1)
            all_m = torch.cat([mask, torch.ones(B * T, 1, dtype=torch.bool, device=x.device)], 1)
            a, _ = self.att(all_t, all_t, all_t, key_padding_mask=~all_m, need_weights=False)
            all_t = self.norm(all_t + a)
            t, tb = all_t[:, :2 * M], all_t[:, 2 * M]
            w = mask.unsqueeze(-1).float()
            own_t, opp_t = t[:, :M], t[:, M:]
            own_w, opp_w = w[:, :M], w[:, M:]
            mean_o = (own_t * own_w).sum(1) / own_w.sum(1).clamp(min=1)
            mean_p = (opp_t * opp_w).sum(1) / opp_w.sum(1).clamp(min=1)
            max_o = (own_t - 1e3 * (1 - own_w)).max(1).values
            max_p = (opp_t - 1e3 * (1 - opp_w)).max(1).values

            def near_ball(t_, db_, w_):
                sc = (-db_.reshape(B * T, M) / 2.0).masked_fill(w_.squeeze(-1) < 0.5, -1e4)
                return (t_ * (torch.softmax(sc, -1).unsqueeze(-1) * w_)).sum(1)
            near_o, near_p = near_ball(own_t, db_o, own_w), near_ball(opp_t, db_p, opp_w)
            return torch.cat([mean_o, max_o, near_o, mean_p, max_p, near_p, tb], -1).reshape(B, T, -1)

        def forward(self, x, b, g):
            h0, _ = self.gru(self.side(x, b, g, 0))
            h1, _ = self.gru(self.side(x, b, g, 1))
            o0, o1 = self.head(h0), self.head(h1)
            return torch.stack([o0[..., 0], o1[..., 0], 0.5 * (o0[..., 1] + o1[..., 1])], -1)
    return Net()


def hole(bb, rng):
    """A long hole in the ball within one training window: the position stays where it was last seen."""
    L = len(bb)
    n = int(rng.integers(5 * 25 // STEP, 30 * 25 // STEP))
    a = int(rng.integers(0, max(1, L - n)))
    e = min(L, a + n)
    bb[a:e, :2] = bb[max(a - 1, 0), :2]
    bb[a:e, 2:6] = 0.0
    bb[a:e, 6] = 50.0
    return bb


class Data:
    def __init__(self, keys, arrays=None):
        self.d = list(arrays) if arrays is not None else []
        for k in keys:
            with np.load(os.path.join(PREP, k + ".npz")) as z:
                y = z["y"].astype(np.int64)
                if READ == "touch":          # round 78: the touch-level reading in place of the controlled one
                    y = np.load(os.path.join(OUT, "pff_touch", k + ".npy")).astype(np.int64)[:len(y)]
                self.d.append((z["tok"].astype(np.float32), z["ball"].astype(np.float32), z["signs"].astype(np.float32), y))
        self.w = np.array([len(x[3]) for x in self.d], np.float64)
        self.w /= self.w.sum()

    def batch(self, rng, size, gaps=True):
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
                bb[..., 3] *= -1
            if gaps and rng.random() < GAP_P:
                bb = hole(bb, rng)
            X.append(x)
            Bb.append(bb)
            G.append(signs)
            Y.append(y[a:a + WIN])
        L = min(len(v) for v in Y)
        return (np.stack([v[:L] for v in X]), np.stack([v[:L] for v in Bb]), np.stack(G), np.stack([v[:L] for v in Y]))


def device():
    import torch
    return "cpu" if os.environ.get("BS_CPU") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(net, x, bb, g, y):
    import torch
    import torch.nn.functional as F
    ctrl = net(x, bb, g)
    ok = y >= 0
    edge = WIN // 12
    ok[:, :edge] = False
    ok[:, -edge:] = False
    lc = F.cross_entropy(ctrl[ok], y[ok])
    acc_ok = ok & (y < 2)
    acc = (ctrl[..., :2].argmax(-1)[acc_ok] == y[acc_ok]).float().mean()
    return lc, acc


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
    vb = [dv.batch(vrng, 16, gaps=False) for _ in range(8)]
    best, t0 = np.inf, time.time()
    path = os.path.join(OUT, "bs%s_s%d.pt" % (TAG, seed))
    as_t = lambda *a: [torch.from_numpy(v).to(dev) for v in a]  # noqa: E731
    for step in range(1, STEPS + 1):
        net.train()
        x, bb, g, y = dt.batch(rng, BATCH)
        loss, _ = loss_fn(net, *as_t(x, bb, g, y))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 500 == 0 or step == STEPS:
            net.eval()
            with torch.no_grad():
                r = np.mean([[float(v) for v in loss_fn(net, *as_t(*b))] for b in vb], axis=0)
            print("  seed %d step %d: validation control %.4f, team accuracy on live steps %.4f, %.0f s" % (seed, step, r[0], r[1], time.time() - t0), flush=True)
            if r[0] < best:
                best = r[0]
                torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, seed=seed, step=step), path)


def predict(net, tok, ball, signs, n, dev):
    """(q at 25 fps: P(H | live), alive log-odds at 25 fps) from overlapping windows."""
    import torch
    T = len(tok)
    lp = np.zeros((T, 3), np.float32)
    starts = list(range(-(WIN - CORE) // 2, T, CORE))
    with torch.no_grad():
        for b0 in range(0, len(starts), 16):
            part = starts[b0:b0 + 16]
            xs = np.zeros((len(part), WIN) + tok.shape[1:], np.float32)
            bs = np.zeros((len(part), WIN, ball.shape[1]), np.float32)
            bs[..., 6] = 50.0
            for j, a in enumerate(part):
                lo, hi = max(0, a), min(T, a + WIN)
                xs[j, lo - a:hi - a] = tok[lo:hi]
                bs[j, lo - a:hi - a] = ball[lo:hi]
            g = np.repeat(signs[None], len(part), 0)
            ctrl = net(torch.from_numpy(xs).to(dev), torch.from_numpy(bs).to(dev), torch.from_numpy(g).to(dev))
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
    ck = torch.load(os.path.join(OUT, "bs%s_s%d.pt" % (TAG, SEEDS[int(s_i)])), map_location="cpu", weights_only=False)
    net = build_net().to(dev)
    net.load_state_dict(ck["state"])
    print("  checkpoint of step %d" % ck["step"], flush=True)
    return net.eval()


FT_STEPS = 2000
FT_LR = 3e-4
TARGET = (("dfl@v6", "dfl@v3", "evctrl"), ("metrica@v6g", "metrica@v3", "ctrl"))


def finetune(s_i):
    """J1's recipe on this channel: half of every batch target-convention windows, half its own PFF windows; each of the ten games is predicted by the fold that never saw it; SkillCorner and SoccerTrack by the model of all ten. The base checkpoint is BS's (no tag)."""
    import torch
    import bench
    bf = _fb()
    assert TAG, "set BS_TAG"
    s_i = int(s_i)
    seed = SEEDS[s_i]
    dev = device()
    base = torch.load(os.path.join(OUT, "bs_s%d.pt" % seed), map_location="cpu", weights_only=False)["state"]
    with open(os.path.join(PREP, "index.json")) as fh:
        idx = json.load(fh)
    val = set(idx["val_matches"])
    dp = Data([k for k in idx["keys"] if k.split("_")[0] not in val])
    halves = []
    for fsrc, src, conv in TARGET:
        for i, (b, lab) in enumerate(zip(bench.blocks(fsrc), bench.labels(src, conv))):
            tok, _, signs = bf.tokens(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
            ball = ball_cols(b["B"], b["pitch"])
            y = bf.step_labels(np.asarray(lab))
            T = min(len(tok), len(ball), len(y))
            halves.append(dict(src=fsrc, i=i, match="%s/%s" % (fsrc, b["group"]), tok=tok[:T].astype(np.float32), ball=ball[:T], signs=np.asarray(signs, np.float32), y=y[:T].astype(np.int64), n=len(b["P"])))
        bench._blocks.clear()
    folds = sorted({h["match"] for h in halves})
    outs = {fsrc: [None] * sum(h["src"] == fsrc for h in halves) for fsrc, _, _ in TARGET}
    as_t = lambda *a: [torch.from_numpy(v).to(dev) for v in a]  # noqa: E731
    for j, fold in enumerate(folds + [None]):
        t0 = time.time()
        dt = Data([], arrays=[(h["tok"], h["ball"], h["signs"], h["y"]) for h in halves if h["match"] != fold])
        dt.w = np.array([len(x[3]) for x in dt.d], np.float64)
        dt.w /= dt.w.sum()
        torch.manual_seed(seed + j)
        rng = np.random.default_rng(seed + j)
        net = build_net().to(dev)
        net.load_state_dict(base)
        opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, FT_LR, total_steps=FT_STEPS, pct_start=0.1)
        net.train()
        for _ in range(FT_STEPS):
            x1, b1, g1, y1 = dt.batch(rng, BATCH // 2)
            x2, b2, g2, y2 = dp.batch(rng, BATCH // 2)
            L = min(x1.shape[1], x2.shape[1])
            loss, _ = loss_fn(net, *as_t(np.concatenate([x1[:, :L], x2[:, :L]]), np.concatenate([b1[:, :L], b2[:, :L]]), np.concatenate([g1, g2]), np.concatenate([y1[:, :L], y2[:, :L]])))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
        net.eval()
        if fold is None:
            torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, seed=seed, step=FT_STEPS), os.path.join(OUT, "bs%s_s%d.pt" % (TAG, seed)))
        else:
            for h in halves:
                if h["match"] == fold:
                    outs[h["src"]][h["i"]] = predict(net, h["tok"], h["ball"], h["signs"], h["n"], dev)
        print("  fold %s: %.0f s" % (fold or "all ten", time.time() - t0), flush=True)
    for fsrc, lst in outs.items():
        bench._dump(dict(q=[z[0] for z in lst], alive=[z[1] for z in lst]), held_path(fsrc, s_i))
    infer(s_i, ("sc@v6", "st@v6"))


def held_path(src, s_i):
    return os.path.join(OUT, "held%s_%s_%d.pkl" % (TAG, src.replace("@", "-"), int(s_i)))


def infer(s_i, sources=SOURCES):
    import bench
    bf = _fb()
    dev = device()
    net = load_model(s_i, dev)
    for src in sources:
        res = []
        for b in bench.blocks(src):
            tok, _, signs = bf.tokens(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
            res.append(predict(net, tok, ball_cols(b["B"], b["pitch"]), signs, len(b["P"]), dev))
        bench._dump(dict(q=[z[0] for z in res], alive=[z[1] for z in res]), held_path(src, s_i))
        print("  %s seed index %s: %d halves" % (src, s_i, len(res)), flush=True)
        bench._blocks.clear()


def audit(s_i):
    """Alone, on the frames and on spells with the true dead ball: what this channel knows next to the others."""
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_crf as ec
    from exp_crf import bench, ee, eg2, er, es, exp_smooth, ss
    sys.argv = _argv
    import exp_stack as xs
    s_i = int(s_i)
    for k in ("dfl", "metrica", "sc", "st"):
        if k == "st":
            import exp_st
            fsrc, src, conv = "st@v6", exp_st.SRC, "evctrl"
            v, g = exp_st.v10_raw(s_i), exp_st.gru_mean()
            o = [ss.pk(np.asarray(x, np.float32), 0) for x in bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))[ss.SEEDS[s_i]]]
            bfsrc = "st@v3"
        else:
            fsrc, src, conv = ee.V6[k][0], ee.V3[k][1], ee.V3[k][2]
            v, g, o = eg2.parts(src, s_i, True)
            o = [ss.pk(np.asarray(z, np.float32), 0) for z in o]
            bfsrc = src
        blocks = bench.blocks(fsrc)
        labs = [np.asarray(y) for y in bench.labels(src, conv)]
        one = bench._load(os.path.join(bench.ROOT, "cache", "ballfree", "held_f_%s_%d.pkl" % (bfsrc.replace("@", "-"), s_i)))["q"]
        mean3 = [z[0] for z in xs.bf_mean(bfsrc, (0, 1, 2))]
        h = bench._load(held_path(fsrc, s_i))
        ch = {"v10": v, "BiGRU": g, "J1": o, "BF": one, "BF mean": mean3, "BS": h["q"]}
        ns = [min([len(y)] + [len(c[i]) for c in ch.values()]) for i, y in enumerate(labs)]
        tot, parts = {}, []
        for name, c in ch.items():
            sm = exp_smooth.smooth([np.asarray(c[i], np.float64)[:ns[i]] for i in range(len(ns))], 25)
            right, n_live = 0, 0
            for s_, y, m in zip(sm, labs, ns):
                live = y[:m] < 2
                right += np.sum((s_[live] > 0.5) == (y[:m][live] == 0))
                n_live += live.sum()
            true_d = [np.where(y[:m] == 2, -12.0, 12.0) for y, m in zip(labs, ns)]
            preds = ec.decode_all(true_d, sm, 6.0, 24.0)
            t = es.Tally()
            for y, p, m in zip(labs, preds, ns):
                t.add(y[:m], np.asarray(p))
            sf = np.mean(list(er.by_match(blocks, [y[:m] for y, m in zip(labs, ns)], preds).values()))
            short = t.g_hit[:es.N_SHORT].sum() / max(t.g_n[:es.N_SHORT].sum(), 1)
            prec = t.p_hit[:es.N_SHORT].sum() / max(t.p_n[:es.N_SHORT].sum(), 1)
            tot[name] = right / n_live
            parts.append("%s acc %.4f spell %.4f <4s %.3f/%.3f" % (name, right / n_live, sf, short, prec))
        print("%-8s %s" % (k, " | ".join(parts)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(*sys.argv[2:])
    elif cmd == "prep-touch":
        prep_touch()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "infer":
        infer(sys.argv[2])
    elif cmd == "audit":
        audit(sys.argv[2])
    elif cmd == "finetune":
        finetune(sys.argv[2])
    else:
        raise SystemExit(__doc__)
