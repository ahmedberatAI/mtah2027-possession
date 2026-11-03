"""Round 23: the short-possession specialist (KPU, "kısa possession uzmanı"), docs/RAPOR.md Tur 23 pre-registration.

An event-level model of who controls the ball and when control changes, trained only on PFF's human-labelled touches
(src/gradient.py controlled labels) and never on DFL, Metrica, SkillCorner or BAR-ATH, so it scores all of them out of
sample. It is meant to act only where short possessions are decided: as new evidence for the short-possession verifier
and as a source of new "insert" candidates.

Representation, per frame (25 fps) and per side s (home / away), rotated so that side s attacks +x, relative to the
ball: ball state (position, velocity, acceleration, visibility, time since seen), the K = 5 players of each team nearest
to the ball (relative position, distance, velocity, closing speed, presence) and context (counts within 3 / 6 m, team
centroids, goalkeeper distances, players present). Variant S2 adds v10's raw and +-25-smoothed logit seen from side s.
Model: a shared frame encoder and one bidirectional GRU run on both sides' views (symmetric by construction), +-2 s of
context around 2 s cores; per side a control, a "gains control" and a "loses control" logit, and a dead-ball logit.
Outputs per frame (7): P(H), P(A), P(D), gain H, gain A, lose H, lose A.
Variant C1 (round 24) learns the target convention instead: PFF's labels under the challenge's controlled-possession
reading (gradient.controlled_reading, the rule behind Metrica ctrl / DFL evctrl), plus a separate per-team touch head
from PFF's on-the-ball events, so that touching the ball and gaining control are learnt apart (9 outputs).

  python -I src/short_specialist.py prep-pff OUT [WORKERS] [CONV]    degraded PFF halves -> features + targets
                                                                     (CONV pff for S1 / S2, ctrl for C1,
                                                                     ctrlmask for C2: round 32)
  python -I src/short_specialist.py train DATA VARIANT SEED OUT      GPU: one model (VARIANT S1 / S2)
  python -I src/short_specialist.py infer SRC VARIANT                local: outputs on bench halves, 3 seeds
"""
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

FPS = 25
K = 5
CORE, HALO = 50, 50
WIN = CORE + 2 * HALO
SIGMA = 3.0
MASK_PAD = 6                    # round 32 (C2): frames around a masked spell without gain / lose target
SEEDS = (20261102, 20261103, 20261104)
MIX = (("change", 0.4), ("contest", 0.3), ("edge", 0.1), ("live", 0.2))      # training window centres
N_BASE = 8 + 2 * 7 * K + 12
Y_COLS = [1, 3] + [8 + 7 * j + c for j in range(2 * K) for c in (1, 4)] + [8 + 14 * K + c for c in (5, 7)]
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "cache", "kpu")


# -- representation -------------------------------------------------------------------------------------------------

def _signs(P, team, gk):
    """+1 when side s attacks +x (its goalkeepers, else its players, sit at x < 0.5 on average)."""
    signs = []
    for s in (0, 1):
        x = np.nan
        for sel in ((team == s) & gk, team == s):
            if sel.any() and np.isfinite(P[:, sel, 0]).any():
                x = float(np.nanmean(P[:, sel, 0]))
                break
        signs.append(1.0 if not np.isfinite(x) or x < 0.5 else -1.0)
    if signs[0] == signs[1]:
        signs[1] = -signs[0]
    return signs


def _ball(B, pitch):
    n = len(B)
    xy = np.asarray(B[:, :2], np.float64) * np.array(pitch)
    seen = np.isfinite(xy[:, 0])
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    first = int(np.argmax(seen)) if seen.any() else 0
    last = np.where(last < 0, first, last)
    pos = xy[last] if seen.any() else np.zeros((n, 2)) + np.array(pitch) / 2
    since = np.clip(np.arange(n) - np.where(idx >= 0, idx, last), 0, 50)
    vel = np.zeros((n, 2))
    vel[2:-2] = np.clip((pos[4:] - pos[:-4]) * FPS / 4.0, -40.0, 40.0)            # m/s; track jumps are not shots
    acc = np.zeros(n)
    acc[2:-2] = np.linalg.norm(vel[4:] - vel[:-4], axis=1) * FPS / 4.0
    return pos, vel, acc, seen, since


def features(P, B, team, gk, pitch, prior=None):
    """(T, 2, N_BASE [+ 2]) float32: side s's view of every frame. prior: v10's P(home) per frame (S2)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    L, W = pitch
    n = len(P)
    team = np.asarray(team)
    gk = np.asarray(gk, bool)
    pos, vel, acc, seen, since = _ball(B, pitch)
    Pm = np.asarray(P[:, :, :2], np.float64) * np.array(pitch)
    present = np.isfinite(Pm[:, :, 0])
    Pv = np.zeros_like(Pm)
    Pv[2:-2] = (Pm[4:] - Pm[:-4]) * FPS / 4.0
    Pv[~np.isfinite(Pv)] = 0.0
    Pv = np.clip(Pv, -12.0, 12.0)                                                   # m/s; slot swaps are not sprints
    rel = Pm - pos[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    unit = -rel / np.maximum(dist, 1e-3)[:, :, None]
    closing = np.sum(Pv * unit, axis=2)
    dist_inf = np.where(present, dist, np.inf)
    signs = _signs(P, team, gk)
    out = np.zeros((n, 2, N_BASE + (2 if prior is not None else 0)), np.float32)
    if prior is not None:
        import exp_smooth
        lg = np.log(np.clip(prior, 1e-4, 1 - 1e-4) / np.clip(1 - prior, 1e-4, 1))
        sm = np.asarray(exp_smooth.smooth([np.asarray(prior, np.float64)], 25)[0])
        lgs = np.log(np.clip(sm, 1e-4, 1 - 1e-4) / np.clip(1 - sm, 1e-4, 1))
    for s in (0, 1):
        g = signs[s]
        f = out[:, s]
        f[:, 0] = g * (pos[:, 0] - L / 2) / (L / 2)
        f[:, 1] = g * (pos[:, 1] - W / 2) / (W / 2)
        f[:, 2] = g * vel[:, 0] / 10.0
        f[:, 3] = g * vel[:, 1] / 10.0
        f[:, 4] = np.linalg.norm(vel, axis=1) / 10.0
        f[:, 5] = np.minimum(acc / 50.0, 3.0)
        f[:, 6] = seen
        f[:, 7] = since / 50.0
        c = 8
        for t in (s, 1 - s):
            m = np.flatnonzero(team == t)
            order = np.argsort(dist_inf[:, m], axis=1)[:, :K]
            sl = m[order]
            rows = np.arange(n)[:, None]
            ok = np.isfinite(dist_inf[rows, sl])
            blk = np.zeros((n, sl.shape[1], 7))
            blk[:, :, 0] = g * rel[rows, sl, 0] / 10.0
            blk[:, :, 1] = g * rel[rows, sl, 1] / 10.0
            blk[:, :, 2] = dist[rows, sl] / 10.0
            blk[:, :, 3] = g * Pv[rows, sl, 0] / 5.0
            blk[:, :, 4] = g * Pv[rows, sl, 1] / 5.0
            blk[:, :, 5] = closing[rows, sl] / 5.0
            blk[:, :, 6] = 1.0
            blk[~ok] = 0.0
            f[:, c:c + 7 * sl.shape[1]] = blk.reshape(n, -1)
            c += 7 * K
        for j, t in enumerate((s, 1 - s)):
            m = team == t
            d = np.where(present[:, m], dist[:, m], np.inf)
            f[:, c + j] = np.sum(d <= 3.0, axis=1) / 3.0
            f[:, c + 2 + j] = np.sum(d <= 6.0, axis=1) / 5.0
            cen = np.nanmean(np.where(present[:, m][:, :, None], Pm[:, m], np.nan), axis=1) - pos
            f[:, c + 4 + 2 * j] = np.nan_to_num(g * cen[:, 0] / 30.0)
            f[:, c + 5 + 2 * j] = np.nan_to_num(g * cen[:, 1] / 30.0)
            kg = m & gk
            dg = np.where(present[:, kg], dist[:, kg], np.inf).min(axis=1) if kg.any() else np.full(n, np.inf)
            f[:, c + 8 + j] = np.where(np.isfinite(dg), dg, 100.0) / 50.0
            f[:, c + 10 + j] = present[:, m].sum(axis=1) / 11.0
        if prior is not None:
            f[:, N_BASE] = (lg if s == 0 else -lg) / 5.0
            f[:, N_BASE + 1] = (lgs if s == 0 else -lgs) / 5.0
    return np.clip(np.nan_to_num(out), -5.0, 5.0)


def targets(lab):
    """ctrl (T,) int8 in {0, 1, 2} or -1; events (T, 4) float32: gain H, gain A, lose H, lose A (Gaussian, sigma 3)."""
    lab = np.asarray(lab)
    n = len(lab)
    ctrl = np.where(lab <= 2, lab, -1).astype(np.int8)
    ev = np.zeros((n, 4), np.float32)
    prev, cur = lab[:-1], lab[1:]
    known = (prev <= 2) & (cur <= 2) & (prev != cur)
    t = np.arange(n)
    for k, sel in enumerate(((cur == 0), (cur == 1), (prev == 0), (prev == 1))):
        for e in np.flatnonzero(known & sel) + 1:
            lo, hi = max(0, e - 12), min(n, e + 13)
            ev[lo:hi, k] = np.maximum(ev[lo:hi, k], np.exp(-0.5 * ((t[lo:hi] - e) / SIGMA) ** 2))
    return ctrl, ev


def touch_targets(evs, n, sigma=2.0):
    """(T, 2) float32: Gaussian bumps (sigma 2 frames) at every on-the-ball event of the home / away team
    (gradient.team_events) -- who touches the ball, kept apart from who controls it (round 24, variant C1)."""
    out = np.zeros((n, 2), np.float32)
    t = np.arange(n)
    for f, team, kind, oc in evs:
        lo, hi = max(0, f - 8), min(n, f + 9)
        out[lo:hi, team] = np.maximum(out[lo:hi, team], np.exp(-0.5 * ((t[lo:hi] - f) / sigma) ** 2))
    return out


def centres(lab, x):
    """Training window centres per kind (MIX): team changes, contests (both teams within 3 m), live/dead edges, live."""
    lab = np.asarray(lab)
    live = lab < 2
    change = np.flatnonzero((lab[1:] < 2) & (lab[:-1] < 2) & (lab[1:] != lab[:-1])) + 1
    edge = np.flatnonzero((lab[1:] <= 2) & (lab[:-1] <= 2) & ((lab[1:] == 2) != (lab[:-1] == 2))) + 1
    c = 8 + 14 * K
    contest = np.flatnonzero(live & (x[:, 0, c] > 0) & (x[:, 0, c + 1] > 0))[::5]
    return {"change": change, "contest": contest, "edge": edge, "live": np.flatnonzero(live)[::25]}


# -- PFF preparation (Colab) ------------------------------------------------------------------------------------------

def _prep_one(job):
    """One PFF match -> per half and degradation copy: features, targets, window centres. conv "pff": PFF's controlled
    labels (variants S1 / S2); "ctrl": their controlled-possession reading (gradient.controlled_reading) plus touch
    targets (variant C1)."""
    mi, match, out, with_prior, conv = job
    import gradient
    import sim
    warnings.filterwarnings("ignore")
    t0 = time.time()
    done = []
    path, info = gradient.parse(match)                    # cached npz (src/gradient.py), parsed here when missing
    shifts = {p["period"]: p["applied_shift"] for p in info["periods"]}
    rows = gradient.read_json(gradient.DATA / "events" / (match + ".json")) if conv in ("ctrl", "ctrlmask") else None
    with np.load(path) as d:
        pitch = tuple(float(v) for v in d["pitch"])
        for per in d["periods"]:
            tag = "h%d" % per
            lab = d[tag + "_labels"]
            if not ((lab == 0) | (lab == 1)).any():
                continue
            touch = None
            unknown = None
            if conv in ("ctrl", "ctrlmask"):
                evs = gradient.team_events(rows, int(per), d[tag + "_time"], shifts[int(per)])
                touch = touch_targets(evs, len(lab))
                lab_r = gradient.controlled_reading(lab, evs)
                if conv == "ctrlmask":                 # round 32: PFF possessions the reading hands back are unknown
                    unknown = (lab_r != lab) & (lab < 2)
                lab = lab_r
            for var, params, base in (("v3", sim.V3, 30000), ("v4", sim.V4, 40000)):
                key = "%s_%s_%s" % (match, tag, var)
                if os.path.isfile(os.path.join(out, key + "_x.npy")):
                    done.append(key)
                    continue
                Pn, Bn = sim.degrade(d[tag + "_players"], d[tag + "_ball"], lab, pitch, seed=base + 10 * mi + int(per),
                                     params=params)
                P, B = Pn[:, :, :2].astype(np.float32), Bn[:, :2].astype(np.float32)
                prior = _v10_prior(P, B, d["team"], d["gk"], pitch, os.path.join(out, key + "_v7.pkl")) \
                    if with_prior else None
                x = features(P, B, d["team"], d["gk"], pitch, prior)
                ctrl, ev = targets(lab)
                if unknown is not None and unknown.any():
                    ctrl[unknown] = -1
                    near = np.convolve(unknown.astype(np.int8), np.ones(2 * MASK_PAD + 1, np.int8), "same") > 0
                    ev[near] = np.nan                  # no gain / lose target around them (touch targets stay)
                if touch is not None:
                    ev = np.concatenate([ev, touch], 1)
                np.save(os.path.join(out, key + "_x.npy"), x.astype(np.float16))
                np.save(os.path.join(out, key + "_ctrl.npy"), ctrl)
                np.save(os.path.join(out, key + "_ev.npy"), ev.astype(np.float16))
                if prior is not None:
                    live = lab < 2
                    wrong = live & (np.where(lab == 0, prior, 1 - prior) < 0.5)
                    np.save(os.path.join(out, key + "_hard.npy"), wrong)
                with open(os.path.join(out, key + "_centres.json"), "w") as fh:
                    json.dump({k: v.tolist() for k, v in centres(lab, x).items()}, fh)
                done.append(key)
    print("prepared %s: %d halves x copies, %.0f s" % (match, len(done), time.time() - t0), flush=True)
    return done


def _v10_prior(P, B, team, gk, pitch, tmp):
    """v10's P(home) on a degraded block (v7 features + ball-free features, the shipped team model)."""
    import bench
    import bfree
    import model_v7
    import model_v10
    b = dict(P=P, B=B, team=np.asarray(team), gk=np.asarray(gk, bool), pitch=pitch)
    bench._block_feats((model_v7.VERSION, b, tmp))
    f = bench._load(tmp)
    os.remove(tmp)
    h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
    return bench.predict_team(model_v10.team_model(), dict(XF=f["XF"], XH=np.hstack([f["XH"], h]),
                                                           XA=np.hstack([f["XA"], a])))


def prep_pff(out, workers, conv="pff"):
    from concurrent.futures import ProcessPoolExecutor
    import gradient
    os.makedirs(out, exist_ok=True)
    jobs = [(mi, m, out, conv == "pff", conv) for mi, m in enumerate(gradient.matches())]
    keys = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_prep_one, jobs):
            keys += part
    rng = np.random.default_rng(20261102)
    matches = gradient.matches()
    val = sorted(rng.permutation(matches)[:8].tolist())
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump({"keys": sorted(keys), "val_matches": val}, fh)
    print("index: %d halves x copies; validation matches %d" % (len(keys), len(val)), flush=True)


# -- model ------------------------------------------------------------------------------------------------------------

def net(d_in, emb=128, hid=96, heads=4):
    """heads 4: own control, own gains, own loses, dead; 5 (variant C1) adds own touch before dead."""
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_in, emb), nn.GELU(), nn.Linear(emb, emb), nn.GELU())
            self.gru = nn.GRU(2 * emb, hid, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * hid, heads)

        def forward(self, x):                                    # x: (B, T, 2, d)
            e = self.enc(x)
            h0, _ = self.gru(torch.cat([e[:, :, 0], e[:, :, 1]], -1))
            h1, _ = self.gru(torch.cat([e[:, :, 1], e[:, :, 0]], -1))
            o0, o1 = self.head(h0), self.head(h1)
            ctrl = torch.stack([o0[..., 0], o1[..., 0], 0.5 * (o0[..., -1] + o1[..., -1])], -1)
            ev = torch.stack([o[..., j] for j in range(1, heads - 1) for o in (o0, o1)], -1)
            return ctrl, ev
    return Net()


class Data:
    def __init__(self, folder, keys, variant):
        self.x, self.ctrl, self.ev, self.hard, self.cen = [], [], [], [], []
        for k in keys:
            x = np.load(os.path.join(folder, k + "_x.npy"), mmap_mode="r")
            self.x.append(x if variant == "S2" else x[:, :, :N_BASE])
            self.ctrl.append(np.load(os.path.join(folder, k + "_ctrl.npy")))
            self.ev.append(np.load(os.path.join(folder, k + "_ev.npy")))
            hp = os.path.join(folder, k + "_hard.npy")
            self.hard.append(np.load(hp) if variant == "S2" and os.path.isfile(hp) else None)
            with open(os.path.join(folder, k + "_centres.json")) as fh:
                self.cen.append({kk: np.asarray(v, np.int64) for kk, v in json.load(fh).items()})
        self.d = self.x[0].shape[-1]

    def window(self, i, c, flip):
        n = len(self.ctrl[i])
        a = c - WIN // 2
        x = np.zeros((WIN, 2, self.d), np.float32)
        y = np.full(WIN, -1, np.int64)
        e = np.zeros((WIN, self.ev[i].shape[1]), np.float32)
        w = np.ones(WIN, np.float32)
        lo, hi = max(0, a), min(n, a + WIN)
        x[lo - a:hi - a] = self.x[i][lo:hi]
        y[lo - a:hi - a] = self.ctrl[i][lo:hi]
        e[lo - a:hi - a] = self.ev[i][lo:hi]
        if self.hard[i] is not None:
            w[lo - a:hi - a] += self.hard[i][lo:hi]
        if flip:
            x[:, :, Y_COLS] *= -1
        y[:HALO] = -1
        y[HALO + CORE:] = -1
        return x, y, e, w

    def batch(self, rng, size):
        kinds = [k for k, _ in MIX]
        p = np.array([q for _, q in MIX])
        out = []
        while len(out) < size:
            i = int(rng.integers(len(self.x)))
            kind = kinds[int(rng.choice(len(kinds), p=p))]
            cs = self.cen[i][kind]
            if not len(cs):
                continue
            c = int(cs[rng.integers(len(cs))]) + int(rng.integers(-CORE // 2, CORE // 2 + 1))
            out.append(self.window(i, c, bool(rng.integers(2))))
        return [np.stack(z) for z in zip(*out)]


def loss_fn(ctrl, ev, y, e, w):
    import torch
    import torch.nn.functional as F
    core = slice(HALO, HALO + CORE)
    ctrl, ev, y, e, w = ctrl[:, core], ev[:, core], y[:, core], e[:, core], w[:, core]
    ok = y >= 0
    ce = F.cross_entropy(ctrl[ok], y[ok], reduction="none")
    lc = (ce * w[ok]).sum() / w[ok].sum().clamp(min=1)
    known = torch.isfinite(e)                          # round 32 (C2): NaN = no event target there
    e = torch.nan_to_num(e)
    le = ((F.binary_cross_entropy_with_logits(ev, e, reduction="none") * (1 + 9 * e))[known]).sum() / known.sum().clamp(min=1)
    return lc + le, lc, le


def train(folder, variant, seed, out, steps=12000, batch=64, lr=1e-3):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with open(os.path.join(folder, "index.json")) as fh:
        idx = json.load(fh)
    val_m = set(idx["val_matches"])
    tr = [k for k in idx["keys"] if k.split("_")[0] not in val_m]
    va = [k for k in idx["keys"] if k.split("_")[0] in val_m and k.endswith("_v3")]
    dt, dv = Data(folder, tr, variant), Data(folder, va, variant)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    heads = 2 + dt.ev[0].shape[1] // 2
    model = net(dt.d, heads=heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    vrng = np.random.default_rng(7)
    val_batches = [dv.batch(vrng, 128) for _ in range(20)]
    best, hist, t0 = np.inf, [], time.time()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "kpu_%s_s%d.pt" % (variant, seed))
    for step in range(1, steps + 1):
        model.train()
        x, y, e, w = (torch.from_numpy(a).to(dev) for a in dt.batch(rng, batch))
        ctrl, ev = model(x)
        loss, _, _ = loss_fn(ctrl, ev, y, e, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            model.eval()
            vl = []
            with torch.no_grad():
                for vb in val_batches:
                    x, y, e, w = (torch.from_numpy(a).to(dev) for a in vb)
                    ctrl, ev = model(x)
                    vl.append([float(v) for v in loss_fn(ctrl, ev, y, e, torch.ones_like(w))])
            vl = np.mean(vl, axis=0)
            hist.append(dict(step=step, val=vl.tolist(), seconds=time.time() - t0))
            print("%s seed %d step %d: validation loss %.4f (control %.4f, events %.4f), %.0f s" % (
                variant, seed, step, vl[0], vl[1], vl[2], time.time() - t0), flush=True)
            if vl[0] < best:
                best = vl[0]
                torch.save(dict(state=model.state_dict(), d=dt.d, heads=heads, variant=variant, seed=seed, step=step),
                           path)
    with open(os.path.join(out, "kpu_%s_s%d.json" % (variant, seed)), "w") as fh:
        json.dump(dict(history=hist, best=best, train_halves=len(tr), val_halves=len(va)), fh)


# -- inference (local) ------------------------------------------------------------------------------------------------

def load(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = net(ck["d"], heads=ck.get("heads", 4))
    model.load_state_dict(ck["state"])
    return model.eval()


def predict(model, x, batch=64):
    """(T, 3 + events) float32: P(H), P(A), P(D), gain H, gain A, lose H, lose A (+ touch H, touch A for C1), from
    +-2 s context around 2 s cores."""
    import torch
    n = len(x)
    out = None
    starts = list(range(-HALO, n, CORE))
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            xs = np.zeros((len(starts[b0:b0 + batch]), WIN) + x.shape[1:], np.float32)
            for j, a in enumerate(starts[b0:b0 + batch]):
                lo, hi = max(0, a), min(n, a + WIN)
                xs[j, lo - a:hi - a] = x[lo:hi]
            ctrl, ev = model(torch.from_numpy(xs))
            pc = torch.softmax(ctrl, -1).numpy()
            pe = torch.sigmoid(ev).numpy()
            if out is None:
                out = np.zeros((n, 3 + pe.shape[-1]), np.float32)
            for j, a in enumerate(starts[b0:b0 + batch]):
                lo, hi = max(0, a + HALO), min(n, a + HALO + CORE)
                if hi > lo:
                    out[lo:hi, :3] = pc[j, lo - a:hi - a]
                    out[lo:hi, 3:] = pe[j, lo - a:hi - a]
    return out


# -- use at decision time (rounds 23-24c, src/exp_kpu.py; src/model_v15.py) -------------------------------------------

def pk(o, k):
    """Live-normalised probability that team k controls the ball, per frame."""
    o = np.asarray(o, np.float32)
    return o[:, k] / np.maximum(o[:, 0] + o[:, 1], 1e-6)


def runs(mask, min_len):
    st = np.flatnonzero(np.r_[mask[0], mask[1:] & ~mask[:-1]])
    en = np.flatnonzero(np.r_[mask[:-1] & ~mask[1:], mask[-1]]) + 1
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def contest_mask(b):
    """Frames where both teams have a player within 3 m of the ball (the specialist's own representation)."""
    x = features(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
    c = 8 + 14 * K
    return (x[:, 0, c] > 0) & (x[:, 0, c + 1] > 0)


def kpu_cands(o, pr, contest, theta, existing):
    """Insert candidates of the specialist: runs >= 6 frames inside a decoded possession of team k where its live-
    normalised control probability of the other team exceeds theta, touching a contested frame, not overlapping one of
    the blend's own insert candidates of that team (IoU >= 0.3)."""
    out = []
    for kk in (0, 1):
        for s, t in runs((pr == kk) & (pk(o, 1 - kk) > theta), 6):
            if not contest[s:t].any():
                continue
            if any(c == 1 - kk and min(t, e) - max(s, a) >= 0.3 * (max(t, e) - min(s, a)) for a, e, c in existing):
                continue
            out.append((s, t, 1 - kk))
    return out


def add_features(rows, outs):
    """The specialist's five candidate features (round 23): mean / max control of the candidate team inside, its gain
    peak at the start, its loss peak at the end, and its control in the +-2 s around."""
    for r in rows:
        o = np.asarray(outs[r["i"]], np.float32)
        n, s, t, k = len(o), r["s"], r["t"], r["k"]
        p = pk(o, k)
        r["kpu_c"] = float(p[s:t].mean())
        r["kpu_cmax"] = float(p[s:t].max())
        r["kpu_gin"] = float(o[max(0, s - 12):min(n, s + 13), 3 + k].max())
        r["kpu_gout"] = float(o[max(0, t - 12):min(n, t + 13), 5 + k].max())
        ctx = np.r_[p[max(0, s - 50):s], p[t:min(n, t + 50)]]
        r["kpu_ctx"] = float(ctx.mean()) if len(ctx) else 0.5
    return rows


def touch_features(rows, outs):
    """C1's touch head (round 24b): the candidate team's peak touch at the start, its mean touch inside, the other
    team's peak touch at the end."""
    for r in rows:
        o = np.asarray(outs[r["i"]], np.float32)
        n, s, t, kk = len(o), r["s"], r["t"], r["k"]
        r["kpu_tin"] = float(o[max(0, s - 6):min(n, s + 7), 7 + kk].max())
        r["kpu_tmean"] = float(o[s:t, 7 + kk].mean())
        r["kpu_tout"] = float(o[max(0, t - 6):min(n, t + 7), 8 - kk].max())
    return rows


def infer(src, variant):
    """Outputs of the three seeds' models on bench halves -> cache/kpu/<variant>_<src>.pkl {seed: [ (T, 7) per half ]}.
    S2's prior is v10's held-out probability of the paired v10 seed (es.v10_raw), BAR-ATH: v10's shipped model."""
    import bench
    import exp_short as es
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "%s_%s.pkl" % (variant, src.replace("@", "-")))
    blocks = bench.blocks(src)
    res = {}
    for k, seed in enumerate(SEEDS):
        model = load(os.path.join(OUT, "models", "kpu_%s_s%d.pt" % (variant, seed)))
        priors = es.v10_raw(k, src) if variant == "S2" else [None] * len(blocks)
        res[seed] = [predict(model, features(b["P"], b["B"], b["team"], b["gk"], b["pitch"], pr)).astype(np.float16)
                     for b, pr in zip(blocks, priors)]
        print("  %s %s seed %d: %d halves" % (variant, src, seed, len(blocks)), flush=True)
    bench._dump(res, dest)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stage = sys.argv[1]
    if stage == "prep-pff":
        prep_pff(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6, sys.argv[4] if len(sys.argv) > 4 else "pff")
    elif stage == "train":
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif stage == "infer":
        infer(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
