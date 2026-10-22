"""Idea N20 (docs/FIKIRLER.md): temporal stacking. A second LightGBM sees the first-stage (v7 team model)
probabilities around each frame -- logit p_home at offsets up to +-5 s, window means / mins / maxes, the
dead-ball probability -- and predicts possession again, i.e. a learned, context-dependent smoother in front
of the B2 decoder. Level 1 = out-of-fold DFL probabilities (bench oof of v7's tag); level 2 is trained
leave-one-match-out on DFL (native labels) and, trained on all DFL, applied to Metrica / SkillCorner whose
level-1 probabilities come from the all-DFL model (also out of sample). AGGREGATE output only.

  python -I src/exp_stack.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402
from mlfeat import win_mean  # noqa: E402

TAG = "%s_native_v3" % bench._safe(model_v7.VERSION)
OFFS = (-125, -75, -50, -25, -12, 12, 25, 50, 75, 125)
WINS = ((-25, 25), (-75, 75), (-150, 150), (-75, 0), (0, 75))
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=bench.THREADS,
              deterministic=True, force_row_wise=True)
ROUNDS = 300


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def shift(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    else:
        out[-k:] = x[:k]
    return out


def level2_features(ph, pd_):
    """Home-perspective features; the away view is the same with logits negated (side swap)."""
    z = lg(ph)
    cols = {"z0": z, "dead": lg(pd_)}
    for k in OFFS:
        cols["z%+d" % k] = shift(z, k)
    for a, b in WINS:
        cols["zmean_%d_%d" % (a, b)] = win_mean(z, a, b)
    # extremes over +-75 / +-150 via strided windows
    for h in (75, 150):
        pad = np.concatenate([np.full(h, np.nan), z, np.full(h, np.nan)])
        w = np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)
        with np.errstate(all="ignore"):
            cols["zmin_%d" % h] = np.nanmin(w, 1)
            cols["zmax_%d" % h] = np.nanmax(w, 1)
    names = sorted(cols)
    return np.stack([cols[k] for k in names], 1).astype(np.float32), names


def swap(X, names):
    Xs = X.copy()
    for j, k in enumerate(names):
        if k != "dead":
            Xs[:, j] = -X[:, j]
    return Xs


def fit(Xs, ys, names, stride=2):
    X = np.concatenate([x[::stride] for x in Xs])
    y = np.concatenate([v[::stride] for v in ys])
    al = y < 2
    Xa = np.concatenate([X[al], swap(X[al], names)])
    ya = np.concatenate([y[al] == 0, y[al] == 1]).astype(int)
    return lgb.train(PARAMS, lgb.Dataset(Xa, ya), ROUNDS)


def predict(model, X, names):
    return 0.5 * (model.predict(X) + 1 - model.predict(swap(X, names)))


def score(src, conv, pdl, phl, dec):
    bl, gts = bench.blocks(src), bench.labels(src, conv)
    best = None
    for pt in (4, 8, 12, 20):
        sp = decode2.duration_spec(pt, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
        r = bench.spell_by_group(bl, gts, [sp.decode(td.emissions(pd_, ph, 1.0, 1.0)).astype(np.int8)
                                           for pd_, ph in zip(pdl, phl)])
        v = np.array([r[k][0] for k in sorted(r)])
        if best is None or v.mean() > best[1].mean():
            best = (pt, v)
    return best


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    bl = bench.blocks("dfl@v3")
    ph1 = bench._load(os.path.join(bench.CACHE, "oof", TAG + ".pkl"))
    pd1 = bench.dead_probs("dfl@v3", "base")
    ylab = bench.labels("dfl@v3", "native")
    feats = [level2_features(p, d) for p, d in zip(ph1, pd1)]
    names = feats[0][1]
    X = [f[0] for f in feats]
    ph2 = [None] * len(bl)
    for m in sorted({b["group"] for b in bl}):
        tr = [i for i, b in enumerate(bl) if b["group"] != m]
        mdl = fit([X[i] for i in tr], [ylab[i] for i in tr], names)
        for i, b in enumerate(bl):
            if b["group"] == m:
                ph2[i] = predict(mdl, X[i], names)
    full = fit(X, ylab, names)
    l1 = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", TAG + ".txt"))
    rows = []
    for src, conv, dec, ph_a, pdl in [("dfl@v3", "trk", (32, 5, 25), ph1, pd1)] + [
            (s, c, (30, 8, 50), None, None) for s, c in (("metrica@v3", "ctrl"), ("sc@v3", "phase"))]:
        if ph_a is None:
            ph_a = [bench.predict_team(l1, f) for f in bench.features(model_v7.VERSION, src)]
            pdl = bench.dead_ext("base", "v3", src) if src.startswith("sc") else bench.dead_probs(src, "base")
            ph_b = [predict(full, level2_features(p, d)[0], names) for p, d in zip(ph_a, pdl)]
        else:
            ph_b = ph2
        variants = [("learned D", pdl)]
        if src.startswith("sc"):
            variants.append(("true D", [np.where(g == 2, 1 - 1e-4, 1e-4) for g in bench.labels(src, conv)]))
        for dn, pds in variants:
            a = score(src, conv, pds, ph_a, dec)
            b = score(src, conv, pds, ph_b, dec)
            d = b[1] - a[1]
            rows.append("%-11s vs %-6s %-9s | v7 %.4f@%d | stacked %.4f@%d | %+.4f (wins %d/%d)" % (
                src, conv, dn, a[1].mean(), a[0], b[1].mean(), b[0], d.mean(), int((d > 0).sum()), len(d)))
            print(rows[-1], flush=True)
    gain = full.feature_importance("gain")
    print("level-2 gain: " + ", ".join("%s %.0f%%" % (names[j], 100 * gain[j] / gain.sum())
                                       for j in np.argsort(-gain)[:8]))


if __name__ == "__main__":
    main()
