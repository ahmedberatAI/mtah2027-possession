"""Round 8 (research a08 / R-3): players-only dead-ball model (src/podead.py) with bounded, gated fusion.

Model: LightGBM, dead vs alive, on DFL (sim V3) native D; DFL leave-one-match-out, the all-DFL model on Metrica and
SkillCorner. Fusion into the learned dead-ball probability used by v7's decoder (exp_oracle.probs):
  l_new = l_base + g * lam * clip(l_po - l_base, -c, c)
gates g: all frames | base uncertain (0.1 < p_base < 0.9) | no ball seen within +-1 s | either; lam = 0 is v7.
v7 team probabilities and decoder unchanged. Frame dead-ball AUC / F1 of each model, then paired spell F1 against v7
on DFL-trk, Metrica-ctrl, SkillCorner phase / player-possession labels and the worst-case regret over them.
AGGREGATE output only.

  python -I src/exp_podead.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import podead  # noqa: E402
from mlfeat import win_mean  # noqa: E402

OUT = os.path.join(bench.CACHE, "podead")
EPS = 1e-4
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("sc@v3", "pp", (30.0, 8, 50)))
GATES = ("all", "uncertain", "no ball", "either")
GRID = [(g, lam, c) for g in GATES for lam in (0.25, 0.5, 1.0) for c in (1.0, 2.0, 4.0)]


def _job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    return i, podead.po_feats(b["P"], b["team"], b["gk"], b["pitch"])[0]


def feats(src):
    path = os.path.join(OUT, "feat_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_job, [(src, i) for i in range(len(bench.blocks(src)))]), key=lambda r: r[0])
    out = [r[1] for r in res]
    bench._dump(out, path)
    return out


def _fit(X, labs):
    Xs = np.concatenate([x[::bench.STRIDE] for x in X])
    ys = np.concatenate([l[::bench.STRIDE] for l in labs])
    keep = ys < 3
    return lgb.train(bench.LGB_DEAD, lgb.Dataset(Xs[keep], (ys[keep] == 2).astype(int)), bench.ROUNDS)


def po_probs():
    """{src: [p_dead per block]} -- DFL out of fold, the all-DFL model elsewhere."""
    path = os.path.join(OUT, "probs.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    fd = feats("dfl@v3")
    labs = bench.labels("dfl", "native")
    bl = bench.blocks("dfl@v3")
    oof = [None] * len(bl)
    for m in sorted({b["group"] for b in bl}):
        tr = [i for i, b in enumerate(bl) if b["group"] != m]
        mdl = _fit([fd[i] for i in tr], [labs[i] for i in tr])
        for i, b in enumerate(bl):
            if b["group"] == m:
                oof[i] = mdl.predict(fd[i], num_threads=bench.THREADS)
        print("  LOMO %s done" % m, flush=True)
    mdl = _fit(fd, labs)
    mdl.save_model(os.path.join(OUT, "podead_all_dfl.txt"))
    res = {"dfl@v3": oof}
    for src in ("metrica@v3", "sc@v3"):
        res[src] = [mdl.predict(x, num_threads=bench.THREADS) for x in feats(src)]
    bench._dump(res, path)
    return res


def gate(name, pb, seen):
    unc = (pb > 0.1) & (pb < 0.9)
    noball = win_mean(seen.astype(np.float64), -25, 25) == 0
    return {"all": np.ones(len(pb), bool), "uncertain": unc, "no ball": noball, "either": unc | noball}[name]


def fuse(pb, po, g, lam, c):
    lb = e2.logit(pb)
    return e2.sig(lb + g * lam * np.clip(e2.logit(po) - lb, -c, c))


def dead_metrics(pds, labs):
    from sklearn.metrics import f1_score, roc_auc_score
    p = np.concatenate(pds)
    y = np.concatenate(labs)
    k = y < 3
    return roc_auc_score(y[k] == 2, p[k]), f1_score(y[k] == 2, p[k] > 0.5)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    po = po_probs()
    rows = {}
    cache = {}
    for src, conv, dec in SETS:
        if src not in cache:
            phl, pdl = exp_oracle.probs(src)
            seen = [np.isfinite(b["B"][:, 0]) for b in bench.blocks(src)]
            cache[src] = (phl, pdl, seen)
        phl, pdl, seen = cache[src]
        labs = bench.labels(src, conv)
        auc_b, f1_b = dead_metrics(pdl, labs)
        auc_p, f1_p = dead_metrics(po[src], labs)
        print("%-10s %-6s frame dead-ball AUC / F1: learned (v7) %.4f / %.4f | players-only %.4f / %.4f" % (
            src, conv, auc_b, f1_b, auc_p, f1_p), flush=True)
        ref = e2.score(src, conv, e2.decode_all(dec, pdl, phl))
        key = "%s %s" % (src.split("@")[0], conv)
        rows[key] = {("v7", 0, 0): ref.mean()}
        cells = []
        for gname, lam, c in GRID:
            pds = [fuse(pb, pp, gate(gname, pb, s), lam, c) for pb, pp, s in zip(pdl, po[src], seen)]
            v = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            rows[key][(gname, lam, c)] = v.mean()
            cells.append("%s l%g c%g %+.4f (%d/%d)" % (gname, lam, c, v.mean() - ref.mean(), int((v > ref).sum()),
                                                       len(v)))
        print("%-10s %-6s v7 %.4f | %s" % (src, conv, ref.mean(), " | ".join(cells)), flush=True)
    configs = list(rows[next(iter(rows))])
    regret = {cfg: max(max(r.values()) - r[cfg] for r in rows.values()) for cfg in configs}
    best = sorted(regret.items(), key=lambda kv: kv[1])[:8]
    print("worst-case regret (DFL-trk, Metrica-ctrl, SC phase, SC pp): " + " | ".join(
        "%s %.4f" % ("v7" if cfg[0] == "v7" else "%s l%g c%g" % cfg, r) for cfg, r in best), flush=True)
    print("  v7 regret %.4f" % regret[("v7", 0, 0)], flush=True)


if __name__ == "__main__":
    main()
