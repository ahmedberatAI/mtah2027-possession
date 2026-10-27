"""Round 9 diagnostic: how far is BAR-ATH from v10's training inputs, feature by feature? (adversarial validation)
  1. per column of v10's team-model input (v7 features + ball-free features; home view), the standardised median
     shift of BAR-ATH against DFL@V3 + Metrica@V3 (median difference / pooled IQR) -- the largest shifts, and every
     ball-free column, which were added tonight and never checked on BAR-ATH;
  2. a domain classifier (LightGBM) training frames vs BAR-ATH, and vs SkillCorner@V3 for scale: out-of-fold AUC
     (2 folds by half) and the most discriminative columns.
Frames every 1 s. AGGREGATE output only (medians, IQR ratios, AUC, gain shares).

  python -I src/exp_domain.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import bfree  # noqa: E402
import exp_combo  # noqa: E402
import feat_team2  # noqa: E402
import mlfeat  # noqa: E402
import model_v7  # noqa: E402

STEP = 25


def names():
    b = bench.blocks("dfl@v3")[0]
    P, B = b["P"][:3000], b["B"][:3000]
    F, Th, _ = mlfeat.compute(P, B, b["team"], b["gk"], b["pitch"])
    for fn in bench._feat_fns(model_v7.VERSION):
        f2, th2, _ = fn(P, B, b["team"], b["gk"], b["pitch"])
        F.update(f2)
        Th.update(th2)
    S = feat_team2._prep(P, np.full((len(P), 2), np.nan, np.float32), b["team"], b["gk"], b["pitch"])
    v = bfree.velocity(S["pxy"])
    bf = []
    for k in sorted(bfree._side(S, 0, v)):
        bf += ["bf:%s own" % k, "bf:%s opp" % k, "bf:%s diff" % k]
        if k in ("close", "support", "att_rev"):
            bf += ["bf:%s diff %s" % (k, w) for w in ("+-12", "-25..0", "0..25")]
    return ["F:" + k for k in sorted(F)] + ["T:" + k for k in sorted(Th)] + bf


def home_rows(feats, blocks, keep=None):
    X, half = [], []
    for f, b in zip(feats, blocks):
        if keep is not None and not keep(b):
            continue
        x = np.hstack([f["XF"], f["XH"]])[::STEP]
        X.append(x)
        half.append(np.full(len(x), int(b["key"].rsplit("/h", 1)[1]) % 2, np.int8))
    return np.concatenate(X), np.concatenate(half)


def barath():
    out = []
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
    return out


def shift(Xs, Xt):
    with np.errstate(all="ignore"):
        ms, mt = np.nanmedian(Xs, 0), np.nanmedian(Xt, 0)
        iqr = lambda X: np.nanpercentile(X, 75, 0) - np.nanpercentile(X, 25, 0)  # noqa: E731
        qs, qt = iqr(Xs), iqr(Xt)
        smd = (mt - ms) / np.maximum(0.5 * (qs + qt), 1e-6)
        ratio = qt / np.maximum(qs, 1e-6)
    return smd, ratio


def domain_auc(Xs, hs, Xt, ht, cols):
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([Xs, Xt])
    y = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    h = np.concatenate([hs, ht])
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=0, num_threads=bench.THREADS,
                  deterministic=True, force_row_wise=True)
    oof = np.zeros(len(y))
    for k in (0, 1):
        tr = h != k
        w = np.where(y[tr] == 1, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0)
        mdl = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w), 200)
        oof[~tr] = mdl.predict(X[~tr], num_threads=bench.THREADS)
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    mdl = lgb.train(params, lgb.Dataset(X, y, weight=w), 200)
    g = mdl.feature_importance("gain")
    top = np.argsort(-g)[:12]
    return roc_auc_score(y, oof), [(cols[i], g[i] / g.sum()) for i in top]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cols = names()
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    Xd, hd = home_rows(exp_combo.aug_feats("dfl@v3"), bd)
    Xm, hm = home_rows(exp_combo.aug_feats("metrica@v3"), bm)
    Xs, hs = np.concatenate([Xd, Xm]), np.concatenate([hd, hm])
    Xb, hb = home_rows(barath(), bench.blocks("barath"))
    print("columns %d (names %d); frames: training %d, BAR-ATH %d" % (Xs.shape[1], len(cols), len(Xs), len(Xb)))
    assert Xs.shape[1] == len(cols)
    smd, ratio = shift(Xs, Xb)
    order = np.argsort(-np.abs(np.nan_to_num(smd)))
    print("largest standardised median shifts BAR-ATH vs training (median diff / IQR; IQR ratio BAR-ATH/training):")
    for i in order[:15]:
        print("  %-44s %+6.2f  IQR x%.2f" % (cols[i][:44], smd[i], ratio[i]))
    sd, rd = shift(Xd, Xm)
    print("ball-free columns: BAR-ATH vs training | (Metrica vs DFL, for scale)")
    for i, c in enumerate(cols):
        if c.startswith("bf:"):
            print("  %-34s %+6.2f  IQR x%.2f | %+6.2f  IQR x%.2f" % (c[:34], smd[i], ratio[i], sd[i], rd[i]))
    print("median |shift| all columns: BAR-ATH %.2f, Metrica-vs-DFL %.2f" % (
        np.nanmedian(np.abs(smd)), np.nanmedian(np.abs(sd))))
    auc, top = domain_auc(Xs, hs, Xb, hb, cols)
    print("domain classifier training vs BAR-ATH: AUC %.3f; top: %s" % (
        auc, ", ".join("%s %.2f" % (c[:30], g) for c, g in top)), flush=True)
    del Xd, Xm
    Xc, hc = home_rows(exp_combo.aug_feats("sc@v3"), bench.blocks("sc@v3"))
    auc, top = domain_auc(Xs, hs, Xc, hc, cols)
    print("domain classifier training vs SkillCorner: AUC %.3f; top: %s" % (
        auc, ", ".join("%s %.2f" % (c[:30], g) for c, g in top)), flush=True)


if __name__ == "__main__":
    main()
