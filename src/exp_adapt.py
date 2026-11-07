"""Round 72 (docs/RAPOR.md): the ball-free channel adapted to the match in front of it, from the other view. Two teams' shapes with and without the ball are specific to a match, and the ball-free channel (BF) has only seen other teams. The channels that read the ball are an independent view of the same frames: where they are sure, they can teach BF these two teams' shapes (co-training; the rules allow pseudo-labels an algorithm produces, no person decides anything, no ground truth, no public window). To keep the two views independent at the frames that matter, every block of the match is predicted by a model that has not seen any pseudo-label of that block. Earlier self-training (round 9, round 62) stayed inside one view and failed. Aggregate output only.

  python -I src/exp_adapt.py run SEED_INDEX "SETS"      adapted outputs on the open sets (the labels are read only by audit)
  python -I src/exp_adapt.py audit SEED_INDEX "SETS"    frame accuracy before and after, and how often BF is right against the ball channels
"""
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["BF_WIN"] = "300"
os.environ["BF_TAG"] = "_f"
import exp_ballfree as bf  # noqa: E402

TAU = float(os.environ.get("AD_TAU", "2.0"))        # |logit| of the ball channels' blend that counts as sure
TAU_A = float(os.environ.get("AD_TAU_A", "2.0"))     # the same on the alive evidence
STEPS = int(os.environ.get("AD_STEPS", "200"))
LR = float(os.environ.get("AD_LR", "1e-4"))
BLOCKS = int(os.environ.get("AD_BLOCKS", "2"))       # per half
BATCH = 16
PAD = 25                                             # steps (5 s) without a pseudo-label around the block being predicted
TAG = os.environ.get("AD_TAG", "")
OUT = os.path.join(bf.ROOT, "cache", "adapt")


def _load():
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_crf as ec
    sys.argv = _argv
    return ec


def channels(k, s_i):
    """(tokens' source, per half: the ball channels' blend at 25 fps, alive evidence)"""
    ec = _load()
    from exp_crf import bench, ee, eg2, exp_smooth, ss
    if k == "st":
        import exp_st
        src = exp_st.SRC
        d = bench._load(os.path.join(ec.OUT, "inputs_st_%d.pkl" % s_i))
        s50 = d["s50"]
        ev = bench._load(os.path.join(ec.OUT, "evidence_st.pkl"))["ev"]
    else:
        src = ee.V3[k][1]
        v, g, o = eg2.parts(src, s_i, True)
        s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
        ev = bench._load(os.path.join(ec.OUT, "evidence_%s.pkl" % k))["ev"]
    return src, s50, ev


def pseudo(q, a):
    """Per 5 Hz step: 0 H, 1 A, 2 dead where the other view is sure, else -1."""
    n = min(len(q), len(a))
    idx = np.arange(bf.STEP // 2, n - bf.STEP // 2, bf.STEP)
    p = np.clip(np.asarray(q, np.float64)[idx], 1e-4, 1 - 1e-4)
    lq = np.log(p / (1 - p))
    al = np.asarray(a, np.float64)[idx]
    y = np.full(len(idx), -1, np.int64)
    y[(al > TAU_A) & (lq > TAU)] = 0
    y[(al > TAU_A) & (lq < -TAU)] = 1
    y[al < -TAU_A] = 2
    return y


def adapt_match(base_state, toks, signs, ys, ns, seed, dev):
    """One match (its halves). Returns per half (q, alive) with every block predicted by a model blind to that block's pseudo-labels."""
    import torch
    import torch.nn.functional as F
    out = [(np.zeros(n, np.float32), np.zeros(n, np.float32)) for n in ns]
    jobs = [(h, b) for h in range(len(toks)) for b in range(BLOCKS)]
    rng = np.random.default_rng(seed)
    for j, (h, b) in enumerate(jobs):
        T = len(ys[h])
        lo, hi = T * b // BLOCKS, T * (b + 1) // BLOCKS
        yy = [y.copy() for y in ys]
        yy[h][max(0, lo - PAD):hi + PAD] = -1
        torch.manual_seed(seed + j)
        net = bf.build_net().to(dev)
        net.load_state_dict(base_state)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
        weights = np.array([np.sum(y >= 0) for y in yy], np.float64)
        weights /= weights.sum()
        net.train()
        for _ in range(STEPS):
            X, G, Y = [], [], []
            for _ in range(BATCH):
                i = int(rng.choice(len(yy), p=weights))
                a = int(rng.integers(0, max(1, len(yy[i]) - bf.WIN)))
                x = toks[i][a:a + bf.WIN].copy()
                if rng.integers(2):
                    x[..., 1] *= -1
                    x[..., 3] *= -1
                X.append(x)
                G.append(signs[i])
                Y.append(yy[i][a:a + bf.WIN])
            L = min(len(v) for v in Y)
            xt = torch.from_numpy(np.stack([v[:L] for v in X])).to(dev)
            gt = torch.from_numpy(np.stack(G)).to(dev)
            yt = torch.from_numpy(np.stack([v[:L] for v in Y])).to(dev)
            ctrl, _, _ = net(xt, gt)
            ok = yt >= 0
            if not ok.any():
                continue
            loss = F.cross_entropy(ctrl[ok], yt[ok])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
        net.eval()
        q, al = bf.predict(net, toks[h], signs[h], ns[h], dev)
        f0, f1 = lo * bf.STEP, (hi * bf.STEP if b + 1 < BLOCKS else ns[h])
        out[h][0][f0:f1] = q[f0:f1]
        out[h][1][f0:f1] = al[f0:f1]
    return out


def run(s_i, sets):
    import torch
    ec = _load()
    from exp_crf import bench
    s_i = int(s_i)
    dev = bf.device()
    state = torch.load(os.path.join(bf.OUT, "bf_f_s%d.pt" % bf.SEEDS[s_i]), map_location="cpu", weights_only=False)["state"]
    os.makedirs(OUT, exist_ok=True)
    for k in sets:
        t0 = time.time()
        src, s50, ev = channels(k, s_i)
        blocks = bench.blocks(src)
        res = []
        for i in range(0, len(blocks), 2):
            pair = blocks[i:i + 2]
            toks, signs, ys, ns = [], [], [], []
            for j, b in enumerate(pair):
                tok, _, sg = bf.tokens(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
                y = pseudo(s50[i + j], ev[i + j])
                T = min(len(tok), len(y))
                toks.append(tok[:T].astype(np.float32))
                signs.append(np.asarray(sg, np.float32))
                ys.append(y[:T])
                ns.append(len(b["P"]))
            res += adapt_match(state, toks, signs, ys, ns, 20261120 + 1000 * s_i + i, dev)
            print("  %s match %d / %d, sure steps %.3f, %.0f s" % (k, i // 2 + 1, len(blocks) // 2, np.mean(np.concatenate(ys) >= 0), time.time() - t0), flush=True)
        bench._dump(dict(q=[z[0] for z in res], alive=[z[1] for z in res]), os.path.join(OUT, "held%s_%s_%d.pkl" % (TAG, src.replace("@", "-"), s_i)))
        bench._blocks.clear()


def audit(s_i, sets):
    ec = _load()
    from exp_crf import bench, ee, exp_smooth
    s_i = int(s_i)
    for k in sets:
        src, s50, _ = channels(k, s_i)
        conv = "evctrl" if k == "st" else ee.V3[k][2]
        labs = [np.asarray(y) for y in bench.labels(src, conv)]
        one = bench._load(os.path.join(bf.OUT, "held_f_%s_%d.pkl" % (src.replace("@", "-"), s_i)))
        ad = bench._load(os.path.join(OUT, "held%s_%s_%d.pkl" % (TAG, src.replace("@", "-"), s_i)))
        tot = {}
        for i, y in enumerate(labs):
            n = min(len(y), len(one["q"][i]), len(ad["q"][i]), len(s50[i]))
            live = y[:n] < 2
            yy = y[:n][live] == 0
            bl = np.asarray(s50[i])[:n][live] > 0.5
            for name, h in (("BF", one), ("adapted", ad)):
                sm = exp_smooth.smooth([np.asarray(h["q"][i], np.float64)[:n]], 25)[0][live] > 0.5
                t = tot.setdefault(name, np.zeros(4))
                dis = sm != bl
                t += (np.sum(sm == yy), len(yy), np.sum(dis & (sm == yy)), np.sum(dis))
            t = tot.setdefault("ball channels", np.zeros(4))
            t += (np.sum(bl == yy), len(yy), 0, 0)
        print("%-8s %s" % (k, " | ".join("%s acc %.4f" % (nm, t[0] / t[1]) + (", against the ball channels: disagree %.3f, right %.3f" % (t[3] / t[1], t[2] / max(t[3], 1)) if t[3] else "") for nm, t in tot.items())), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    sets = sys.argv[3].split() if len(sys.argv) > 3 else ("dfl", "metrica", "sc", "st")
    if cmd == "run":
        run(sys.argv[2], sets)
    elif cmd == "audit":
        audit(sys.argv[2], sets)
    else:
        raise SystemExit(__doc__)
