"""Round 20: a dense short-spell detector (local GPU) as new evidence for v14's verifier.

The verifier decides on a few thousand candidates from hand-made interval summaries; with perfect decisions on the same
candidates v14 would gain +0.04..+0.06 (round 18 oracle). Here a sequence model learns short possessions from every
frame instead: per team view (the shared BiGRU's v7 inputs) plus three held-out probabilities (v10 mean of three seeds,
the BiGRU's equal mean -- leave-one-match-out on DFL --, the learned dead ball), the target is "this frame lies in a
true possession of this team shorter than 4 s" (primary labels; dead frames masked). Same architecture as the shared
BiGRU (src/seq_pff.py: 64 units, dropout .15, 12.5 Hz, 512-frame windows), source-balanced windows with half of them
centred on short-spell frames, positive weight 4. Every frame gets an out-of-fold probability from a model that never
saw its match (5 folds over the 30 open matches, grouped by match). Its means over a candidate join v14's verifier
features; the pooled leave-one-match-out verifier is retrained and judged against v14 on the ten contexts.
AGGREGATE output only.

  python -I src/exp_ssd.py build                  inputs and targets per half (cache/bench/ssd/)
  python -I src/exp_ssd.py train FOLD SEED        one fold's model (GPU)
  python -I src/exp_ssd.py predict SEED           out-of-fold probabilities for every half
  python -I src/exp_ssd.py rows SEED              detector features on v14's candidates
  python -I src/exp_ssd.py trainpool SEED VSEED   pooled leave-one-match-out verifier with them
  python -I src/exp_ssd.py eval VSEED[,VSEED]     v14 against v14 + detector
"""
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_verify as ev  # noqa: E402

OUT = os.path.join(bench.CACHE, "ssd")
FOLDS, EPOCHS, STEPS, BATCH, POS_W = 5, 10, 192, 30, 4.0
SHORT = 100
FEATS = ["det_k", "det_o", "det_k_max", "det_o_max"]


def index():
    return json.load(open(os.path.join(OUT, "index.json"), encoding="utf-8"))


def fold_of(source, group):
    keys = sorted({(s, g) for s, *_ in ev.SOURCES for g in {b["group"] for b in bench.blocks(s)}})
    order = np.random.default_rng(20261101).permutation(len(keys))
    return {k: int(order[n] % FOLDS) for n, k in enumerate(keys)}[(source, group)]


def build():
    import exp_ctrl2 as e2
    import seq_pff as seq
    os.makedirs(OUT, exist_ok=True)
    rows = []
    folds = {}
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = bench.labels(src, prim)
        gbm = [np.mean([es.v10_raw(s, src)[i] for s in es.SEEDS], 0) for i in range(len(blocks))]
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
        gru = [np.mean([m[i] for m in members], 0) for i in range(len(blocks))]
        dead = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        for i, b in enumerate(blocks):
            v = seq.views(bench._load(e2._feat_path(b)))                                    # (m, 2, 172)
            sl = slice(0, None, seq.STEP)
            ph, pg, pd = (np.asarray(a, np.float32)[sl] for a in (gbm[i], gru[i], dead[i]))
            extra = np.stack([np.stack([ph, pg, pd], 1), np.stack([1 - ph, 1 - pg, pd], 1)], 1)  # (m, 2, 3)
            x = np.concatenate([v, extra[:len(v)]], 2).astype(np.float16)
            g = np.asarray(labs[i])
            st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            en = np.r_[st[1:], len(g)]
            short = np.zeros(len(g), bool)
            for a, e in zip(st, en):
                if g[a] < 2 and e - a < SHORT:
                    short[a:e] = True
            y = np.stack([np.where(g == 2, 2, (g == k) & short) for k in (0, 1)], 1).astype(np.int8)[sl]
            key = b["key"].replace("/", "_").replace("@", "-")
            np.save(os.path.join(OUT, key + "_x.npy"), x)
            np.save(os.path.join(OUT, key + "_y.npy"), y)
            if (src, b["group"]) not in folds:
                folds[(src, b["group"])] = fold_of(src, b["group"])
            rows.append(dict(source=src, group=str(b["group"]), i=i, key=key, full_length=len(g),
                             fold=folds[(src, b["group"])]))
        bench._blocks.clear()
        bench._labels.clear()
        print("  %s inputs ready (%d halves)" % (src, len(blocks)), flush=True)
    json.dump(rows, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), indent=1)
    print("folds:", {f: sum(1 for k, v in folds.items() if v == f) for f in range(FOLDS)}, "matches", flush=True)


class Data:
    def __init__(self, rows):
        self.rows = rows
        self.x = [np.load(os.path.join(OUT, r["key"] + "_x.npy"), mmap_mode="r") for r in rows]
        self.y = [np.load(os.path.join(OUT, r["key"] + "_y.npy")) for r in rows]
        self.by_src = {}
        for n, r in enumerate(rows):
            self.by_src.setdefault(r["source"], []).append(n)
        self.pos = [np.argwhere(y == 1) for y in self.y]                                       # (frame, view)
        self.live = [np.flatnonzero(y[:, 0] < 2) for y in self.y]

    def scaler(self, seed):
        import seq_pff as seq
        rng = np.random.default_rng(seed)
        samples = []
        for src, ns in sorted(self.by_src.items()):
            take = [np.asarray(self.x[n][rng.choice(self.live[n], size=min(400, len(self.live[n])))], np.float32)
                    .reshape(-1, self.x[n].shape[-1]) for n in ns if len(self.live[n])]
            samples.append(np.concatenate(take))
        return seq.Scaler.fit(samples)

    def batch(self, rng, scaler):
        import seq_pff as seq
        import torch
        xx, yy, mm = [], [], []
        srcs = sorted(self.by_src)
        for k in range(BATCH):
            ns = self.by_src[srcs[k % len(srcs)]]
            n = ns[int(rng.integers(len(ns)))]
            if rng.random() < 0.5 and len(self.pos[n]):
                c, side = self.pos[n][int(rng.integers(len(self.pos[n])))]
            else:
                c, side = int(rng.choice(self.live[n])), int(rng.integers(2))
            start = int(c) - seq.CORE // 2
            xx.append(seq.window(self.x[n], start, int(side), scaler))
            lab = np.full(seq.CORE, 2, np.int8)
            a, b = max(0, start), min(len(self.y[n]), start + seq.CORE)
            if b > a:
                lab[a - start:b - start] = self.y[n][a:b, int(side)]
            yy.append((lab == 1).astype(np.float32))
            mm.append((lab < 2).astype(np.float32))
        return tuple(torch.from_numpy(np.stack(a)) for a in (xx, yy, mm))


def model_path(fold, seed):
    return os.path.join(OUT, "ssd_f%d_s%d.pt" % (fold, seed))


def train(fold, seed):
    import torch
    import seq_pff as seq
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda")
    data = Data([r for r in index() if r["fold"] != fold])
    scaler = data.scaler(seed)
    net = seq.Net(2 * data.x[0].shape[-1] + 1, "bigru", 64, 0.15).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    pw = torch.tensor(POS_W, device=dev)
    t0 = time.time()
    for epoch in range(EPOCHS):
        net.train()
        losses = []
        for _ in range(STEPS):
            x, y, m = (a.to(dev) for a in data.batch(rng, scaler))
            pred = net(x)[:, seq.HALO:seq.HALO + seq.CORE]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y, pos_weight=pw, reduction="none")
            loss = ((loss * m).sum(1) / m.sum(1).clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        print("    fold %d seed %d epoch %d loss %.4f (%.0f s)" % (fold, seed, epoch + 1, np.mean(losses),
                                                                   time.time() - t0), flush=True)
    torch.save(dict(state=net.state_dict(), mean=scaler.mean, scale=scaler.scale, d_in=2 * data.x[0].shape[-1] + 1),
               model_path(fold, seed))


def predict(seed):
    """Out-of-fold P(short spell of home), P(short spell of away) per half at 25 fps: {source: [ (n, 2) ]}."""
    import torch
    import seq_pff as seq
    dev = torch.device("cuda")
    rows = index()
    out = {}
    for fold in range(FOLDS):
        ck = torch.load(model_path(fold, seed), map_location=dev, weights_only=False)
        net = seq.Net(ck["d_in"], "bigru", 64, 0.15).to(dev)
        net.load_state_dict(ck["state"])
        net.eval()
        scaler = seq.Scaler(ck["mean"], ck["scale"])
        for r in rows:
            if r["fold"] != fold:
                continue
            x = np.load(os.path.join(OUT, r["key"] + "_x.npy"), mmap_mode="r")
            logits = np.empty((len(x), 2), np.float32)
            with torch.no_grad():
                for side in (0, 1):
                    for s0 in range(0, len(x), seq.CORE * 16):
                        ss = list(range(s0, min(len(x), s0 + seq.CORE * 16), seq.CORE))
                        xx = np.stack([seq.window(x, s, side, scaler) for s in ss])
                        yy = net(torch.from_numpy(xx).to(dev))[:, seq.HALO:seq.HALO + seq.CORE].cpu().numpy()
                        for k, s in enumerate(ss):
                            n = min(seq.CORE, len(x) - s)
                            logits[s:s + n, side] = yy[k, :n]
            full = np.stack([np.interp(np.arange(r["full_length"]), np.arange(len(x)) * seq.STEP, logits[:, k])
                             for k in (0, 1)], 1)
            out.setdefault(r["source"], {})[r["i"]] = (1 / (1 + np.exp(-np.clip(full, -20, 20)))).astype(np.float16)
    res = {src: [d[i] for i in range(len(d))] for src, d in out.items()}
    bench._dump(res, os.path.join(OUT, "oof_s%d.pkl" % seed))
    for src, convs, dec, prim in ev.SOURCES:
        labs = bench.labels(src, prim)
        pos, neg = [], []
        for p, g in zip(res[src], labs):
            st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            en = np.r_[st[1:], len(g)]
            for a, e in zip(st, en):
                if g[a] < 2:
                    (pos if e - a < SHORT else neg).append(float(np.mean(p[a:e, g[a]])))
        print("  %s: mean detector output inside true short / long spells of the right team %.3f / %.3f, spell AUC "
              "%.3f" % (src.split("@")[0], np.mean(pos), np.mean(neg), _auc(np.array(pos), np.array(neg))), flush=True)
        bench._labels.clear()


def _auc(pos, neg):
    import exp_short_angles as ea
    return ea.auc(pos, neg)


def _kind(src):
    return "rows4w50" if src.startswith("dfl") else "rows3w50"


def rows(seed):
    """v14's candidates (blend .5, this v10 seed) with the detector's out-of-fold features."""
    det = bench._load(os.path.join(OUT, "oof_s0.pkl"))           # one detector (seed 0) for every v10 seed
    for src, *_ in ev.SOURCES:
        for dname in ev.DEADS:
            d = bench._load(ev._path(_kind(src), src, seed, dname))
            for r in d["rows"]:
                p = np.asarray(det[src][r["i"]][r["s"]:r["t"]], np.float32)
                r["det_k"], r["det_o"] = float(p[:, r["k"]].mean()), float(p[:, 1 - r["k"]].mean())
                r["det_k_max"], r["det_o_max"] = float(p[:, r["k"]].max()), float(p[:, 1 - r["k"]].max())
            bench._dump(d, ev._path(_kind(src) + "ssd", src, seed, dname))
    print("  seed %d: detector features added to v14's candidates" % seed, flush=True)


def trainpool(seed, vseed):
    parts = {dname: [r for src, *_ in ev.SOURCES for r in bench._load(ev._path(_kind(src) + "ssd", src, seed,
                                                                               dname))["rows"]]
             for dname in ev.DEADS}
    pooled = parts[ev.DEADS[0]] + parts[ev.DEADS[1]]
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    pp = ev.predict_lomo(pooled, ev.FEATSETS["G1"] + FEATS, params)
    n = len(parts[ev.DEADS[0]])
    for k, dname in enumerate(ev.DEADS):
        bench._dump({"G1": pp[:n] if k == 0 else pp[n:]}, ev._path("probs4w50ssd", seed, dname, "q%d" % vseed))
    yy = np.array([r["y"] for r in pooled])
    cells = []
    for typ in ("ins", "del"):
        for src, *_ in ev.SOURCES:
            sel = np.array([(r["typ"] == typ and r["src"] == src and r["y"] >= 0) for r in pooled])
            cells.append("%s %s %.3f" % (typ, src.split("@")[0], _auc(pp[sel & (yy == 1)], pp[sel & (yy == 0)])))
    print("    seed %d q%d verifier + detector AUC %s" % (seed, vseed, " | ".join(cells)), flush=True)


def evaluate(vseeds):
    import exp_blend_weight as bw
    bw.use_weight(0.5)
    cells = {}
    names = ["v14", "v14 + detector"]
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                for vseed in vseeds:
                    outs = {"v14": bw._verified(src, seed, dname, vseed, lambda s2: ev._rows4(s2, seed, dname),
                                                "probs4")[1],
                            "v14 + detector": bw._verified(
                                src, seed, dname, vseed,
                                lambda s2: bench._load(ev._path(_kind(s2) + "ssd", s2, seed, dname)), "probs4w50ssd")[1]}
                    for c in convs:
                        for name, preds in outs.items():
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    mean, regret = er.summarize(cells, names, ref="v14")
    bw.regret_boot(mean, names, "v14", names[1])
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage = sys.argv[1]
    if stage == "build":
        build()
    elif stage == "train":
        train(int(sys.argv[2]), int(sys.argv[3]))
    elif stage == "predict":
        predict(int(sys.argv[2]))
    elif stage == "rows":
        rows(int(sys.argv[2]))
    elif stage == "trainpool":
        trainpool(int(sys.argv[2]), int(sys.argv[3]))
    elif stage == "eval":
        evaluate(tuple(int(x) for x in sys.argv[2].split(",")))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
