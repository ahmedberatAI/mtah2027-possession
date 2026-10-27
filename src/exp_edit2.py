"""Round 9 (research a17), step 2: an edit-utility rescorer for DELETING spurious spells of v7's decoded output.

The step-1 oracle (src/exp_edit.py) puts the deletion headroom at +0.040..+0.061 (insertion only +0.017..+0.029:
raw-probability candidates cover few true short spells). Idea 3 deleted short spells with a "is it real?"
classifier and gained nothing; here the target is the edit's exact effect on the match score: delete one decoded
spell (merge it into the neighbouring possession), recount the official greedy IoU matches of its half and take the
change of the match's mean spell F1 (2M/(P+G) per threshold, halves pooled as the scorer does).

Candidates: decoded H/A spells shorter than 15 s with at least one H/A neighbour (learned D, v7 decoder).
Features: duration, the spell's own-team evidence (logit statistics, integrated evidence, shares), its neighbours
(team pattern, durations, evidence), dead-ball probability inside and at the edges, ball visibility, time since the
last stoppage. LightGBM regression on the utility. Protocols: within each dataset by match, and leave-one-dataset-
out (train on the other two). Apply: delete candidates in descending predicted gain (> 0, top share q), skipping
neighbours of a deleted spell; paired spell F1 against v7 per set. AGGREGATE output only.

  python -I src/exp_edit2.py
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

pm = exp_oracle.pm
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))
THR = (0.3, 0.4, 0.5, 0.6, 0.7)
MAX_DUR = 375
EPS = 1e-4
OUT = os.path.join(bench.CACHE, "edit")
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
              force_row_wise=True, num_threads=8)
ROUNDS = 300
SHARES = (0.05, 0.1, 0.2, 0.35, 0.5)
NO_D = os.environ.get("EDIT_NO_D") == "1"         # drop the dead-ball probability features (BAR-ATH's D is rule-based)
TRUE_D = os.environ.get("EDIT_TRUE_D") == "1"     # candidates from the decode with the true dead-ball mask
D_COLS = (10, 11, 12, 13)


def counts(gs, ps):
    """Greedy one-to-one matches per IoU threshold (the official scorer's procedure)."""
    ov = pm.spell_overlaps(gs, ps)
    out = np.zeros(len(THR))
    for t, thr in enumerate(THR):
        mg, mp = set(), set()
        for v, i, j in ov:
            if v < thr:
                break
            if i not in mg and j not in mp:
                mg.add(i), mp.add(j)
        out[t] = len(mg)
    return out


def f1(M, P, G):
    return float(np.mean(2 * M / max(P + G, 1)))


def segments(lab):
    e = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], e, [len(lab)]))
    return [(int(a), int(b), int(lab[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def delete(y, i, segs):
    """y with segment i merged into its H/A neighbour (the longer one if both)."""
    a, b, k = segs[i]
    prev = segs[i - 1] if i > 0 and segs[i - 1][2] < 2 else None
    nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][2] < 2 else None
    if prev is None and nxt is None:
        return None
    if prev is not None and nxt is not None:
        tgt = prev[2] if (prev[1] - prev[0]) >= (nxt[1] - nxt[0]) else nxt[2]
    else:
        tgt = (prev or nxt)[2]
    out = y.copy()
    out[a:b] = tgt
    return out


def cand_rows(p, pd_, y, seen, g):
    """Feature rows, utility parts (dM per threshold, dP) and segment ids for one half."""
    segs = segments(y)
    z = e2.logit(p)
    gs = exp_oracle.spells(g)
    ps = exp_oracle.spells(y)
    M0 = counts(gs, ps)
    rows, dM, dP, ids = [], [], [], []
    last_dead = -10 ** 9
    for i, (a, b, k) in enumerate(segs):
        if k == 2:
            last_dead = b
            continue
        if k > 2 or b - a >= MAX_DUR:
            continue
        yd = delete(y, i, segs)
        if yd is None:
            continue
        ps2 = exp_oracle.spells(yd)
        dM.append(counts(gs, ps2) - M0)
        dP.append(len(ps2) - len(ps))
        zk = z[a:b] if k == 0 else -z[a:b]
        L = b - a
        q = max(1, L // 5)
        prev = segs[i - 1] if i > 0 else (a, a, 3)
        nxt = segs[i + 1] if i + 1 < len(segs) else (b, b, 3)

        def side(s, own):
            c, d, kk = s
            if kk >= 2:
                return [float(kk == 2), 0.0, 0.0, np.nan]
            zz = z[c:d] if kk == 0 else -z[c:d]
            return [0.0, float(kk == own), float(d - c), float(np.mean(zz)) if d > c else np.nan]

        feats = [L, np.log1p(L), np.mean(zk), np.min(zk), np.max(zk), np.std(zk), np.sum(zk), np.mean(zk > 0),
                 np.mean(zk[:q]), np.mean(zk[-q:]), np.mean(pd_[a:b]), np.max(pd_[a:b]),
                 pd_[max(0, a - 1)], pd_[min(len(pd_) - 1, b)], np.mean(seen[a:b]), float(min(a - last_dead, 5000))]
        feats += side(prev, k) + side(nxt, k)
        rows.append(feats)
        ids.append(i)
    return np.array(rows, np.float32), np.array(dM), np.array(dP), ids, M0, len(ps), len(gs)


def build(src, conv, dec):
    path = os.path.join(OUT, "cand_%s_%s%s.pkl" % (src.replace("@", "-"), conv, "_trueD" if TRUE_D else ""))
    if os.path.isfile(path):
        return bench._load(path)
    phl, pdl = exp_oracle.probs(src)
    labs = bench.labels(src, conv)
    bl = bench.blocks(src)
    if TRUE_D:
        pdl = [np.where(g == 2, 1 - EPS, EPS) for g in labs]
    y0s = e2.decode_all(dec, pdl, phl)
    halves = []
    for b, p, pd_, y, g in zip(bl, phl, pdl, y0s, labs):
        X, dM, dP, ids, M0, P0, G0 = cand_rows(p, pd_, y, np.isfinite(b["B"][:, 0]), g)
        halves.append(dict(X=X, dM=dM, dP=dP, ids=ids, M0=M0, P0=P0, G0=G0, group=(b["group"], b.get("seed", 0)),
                           y0=y))
    # match-level utility: totals over the halves of the same group
    tot = {}
    for h in halves:
        t = tot.setdefault(h["group"], [np.zeros(len(THR)), 0, 0])
        t[0] = t[0] + h["M0"]
        t[1] += h["P0"]
        t[2] += h["G0"]
    for h in halves:
        M, P, G = tot[h["group"]]
        base = f1(M, P, G)
        h["util"] = np.array([f1(M + dm, P + dp, G) - base for dm, dp in zip(h["dM"], h["dP"])])
    os.makedirs(OUT, exist_ok=True)
    bench._dump(halves, path)
    return halves


def apply(h, pred, share):
    """Delete the top `share` of positive-gain candidates of a half, skipping neighbours of deleted spells."""
    y = h["y0"].copy()
    segs = segments(y)
    order = np.argsort(-pred)
    k_max = int(np.ceil(share * len(pred)))
    done = set()
    for r in order[:k_max]:
        if pred[r] <= 0:
            break
        i = h["ids"][r]
        if i in done or (i - 1) in done or (i + 1) in done:
            continue
        yd = delete(y, i, segs)
        if yd is not None:
            a, b, _ = segs[i]
            y[a:b] = yd[a:b]
            done.add(i)
    return y


def cols(X):
    return np.delete(X, D_COLS, axis=1) if NO_D else X


def fit(halves):
    X = cols(np.concatenate([h["X"] for h in halves if len(h["X"])]))
    u = np.concatenate([h["util"] for h in halves if len(h["X"])]) * 1000.0
    return lgb.train(PARAMS, lgb.Dataset(X, u), ROUNDS)


def evaluate(src, conv, halves, preds):
    ref = e2.score(src, conv, [h["y0"] for h in halves])
    cells = []
    for share in SHARES:
        ys = [apply(h, pr, share) if len(pr) else h["y0"] for h, pr in zip(halves, preds)]
        v = e2.score(src, conv, ys)
        cells.append("q%.2f %+.4f (%d/%d)" % (share, v.mean() - ref.mean(), int((v > ref).sum()), len(v)))
    return ref.mean(), cells


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    print("options: no D features %s, true-D candidates %s" % (NO_D, TRUE_D), flush=True)
    data = {src: build(src, conv, dec) for src, conv, dec in SETS}
    for src, conv, _ in SETS:
        hs = data[src]
        u = np.concatenate([h["util"] for h in hs if len(h["X"])])
        print("%-10s candidates %d, utility > 0: %.3f, oracle (delete all positive) %+.4f" % (
            src, len(u), np.mean(u > 0), np.sum(u[u > 0])), flush=True)
    # within-dataset: grouped by match
    for src, conv, _ in SETS:
        hs = data[src]
        groups = sorted({h["group"][0] for h in hs})
        folds = [groups[i::5] for i in range(min(5, len(groups)))]
        preds = [None] * len(hs)
        for fold in folds:
            tr = [h for h in hs if h["group"][0] not in fold]
            mdl = fit(tr)
            for i, h in enumerate(hs):
                if h["group"][0] in fold:
                    preds[i] = mdl.predict(cols(h["X"])) if len(h["X"]) else np.zeros(0)
        ref, cells = evaluate(src, conv, hs, preds)
        print("within %-10s %-6s v7 %.4f | %s" % (src, conv, ref, " | ".join(cells)), flush=True)
    # leave one dataset out
    for src, conv, _ in SETS:
        mdl = fit([h for s, hs in data.items() if s != src for h in hs])
        hs = data[src]
        preds = [mdl.predict(cols(h["X"])) if len(h["X"]) else np.zeros(0) for h in hs]
        ref, cells = evaluate(src, conv, hs, preds)
        print("LODO   %-10s %-6s v7 %.4f | %s" % (src, conv, ref, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
