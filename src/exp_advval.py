"""Adversarial validation (idea E1): can a classifier tell degraded-DFL frames from BAR-ATH frames?
Per feature: univariate AUC (|AUC - 0.5| = shift) and the gain of a joint LightGBM separator.
Also degraded Metrica vs BAR-ATH. AGGREGATE output only (feature names, AUCs, gains).

  python -I src/exp_advval.py [version] [sim variant]
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import bench  # noqa: E402


def rows(feats, stride):
    XF = np.concatenate([f["XF"][::stride] for f in feats])
    XH = np.concatenate([f["XH"][::stride] for f in feats])
    XA = np.concatenate([f["XA"][::stride] for f in feats])
    return XF, XH, XA


def separate(Xs, Xt, names, label):
    X = np.concatenate([Xs, Xt])
    y = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    uni = []
    for j, nm in enumerate(names):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        miss = abs(np.mean(~np.isfinite(Xs[:, j])) - np.mean(~np.isfinite(Xt[:, j])))
        uni.append((abs(auc - 0.5), auc, miss, nm))
    uni.sort(reverse=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    half = len(X) // 2
    tr, te = idx[:half], idx[half:]
    bst = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=500,
                         feature_fraction=0.8, verbose=-1, seed=0, num_threads=bench.THREADS),
                    lgb.Dataset(X[tr], y[tr]), 150)
    auc = roc_auc_score(y[te], bst.predict(X[te]))
    gain = bst.feature_importance("gain")
    top = np.argsort(-gain)[:15]
    print("\n[%s] joint separator AUC %.4f (0.5 = indistinguishable)" % (label, auc))
    print("  top features by separator gain: " + ", ".join("%s %.0f%%" % (names[j], 100 * gain[j] / gain.sum())
                                                        for j in top))
    print("  top univariate shifts (AUC; missing-share difference):")
    for s, a, miss, nm in uni[:20]:
        print("    %-26s AUC %.3f  miss diff %.3f" % (nm, a, miss))
    return auc, uni


def peel(Xs, Xt, names, steps=12):
    """Remove the strongest separating feature one at a time: how many features carry the domain gap?"""
    X = np.concatenate([Xs, Xt])
    y = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X))
    tr, te = idx[:len(X) // 2], idx[len(X) // 2:]
    keep = list(range(X.shape[1]))
    out = []
    for _ in range(steps):
        bst = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=500,
                             feature_fraction=0.8, verbose=-1, seed=0, num_threads=bench.THREADS),
                        lgb.Dataset(X[tr][:, keep], y[tr]), 100)
        auc = roc_auc_score(y[te], bst.predict(X[te][:, keep]))
        gain = bst.feature_importance("gain")
        j = int(np.argmax(gain))
        out.append("%s (AUC %.3f, %.0f%%)" % (names[keep[j]], auc, 100 * gain[j] / gain.sum()))
        del keep[j]
    print("  peel (feature removed next, joint AUC before removal): " + " > ".join(out))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    version = sys.argv[1] if len(sys.argv) > 1 else "base"
    var = sys.argv[2] if len(sys.argv) > 2 else "v2"
    fb = bench.features(version, "barath")
    fd = bench.features(version, bench.src_name("dfl", var))
    fm = bench.features(version, bench.src_name("metrica", var))
    names = ["F:" + k for k in fb[0]["fk"]] + ["H:" + k for k in fb[0]["tk"]]
    XFb, XHb, XAb = rows(fb, 2)
    tgt = np.hstack([XFb, XHb])
    for src, feats, stride in (("degraded DFL vs BAR-ATH", fd, 10), ("degraded Metrica vs BAR-ATH", fm, 12)):
        XF, XH, XA = rows(feats, stride)
        separate(np.hstack([XF, XH]), tgt, names, src)
        peel(np.hstack([XF, XH]), tgt, names)


if __name__ == "__main__":
    main()
