"""Blend of tempo + attention-to-ball cues for dead-ball detection, LOMO on open DFL (clean).
Logistic vs LightGBM, single +-1 s window vs multi-window. AGGREGATE output."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, lightgbm as lgb
import dfl
from mlfeat import win_mean
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.stdout.reconfigure(encoding="utf-8")
WINS = [(-25, 25), (-75, 0), (0, 75), (-150, 150)]
X, Y, M, names = [], [], [], None
for mi, m in enumerate(dfl.MATCHES):
    d = dfl.parse(m); L, W = d["pitch"]
    for h in (1, 2):
        t = "h%d" % h
        P = d[t + "_players"][:, :, :2] * np.array([L, W], np.float32)
        B = d[t + "_ball"][:, :2] * np.array([L, W], np.float32)
        lab = d[t + "_labels"]
        v = np.full(P.shape, np.nan, np.float32); v[1:-1] = (P[2:] - P[:-2]) * 12.5
        sp = np.linalg.norm(v, axis=2); to = B[:, None, :] - P; dist = np.linalg.norm(to, axis=2)
        cos = np.sum(v * to, axis=2) / np.maximum(sp * dist, 1e-6); mov = sp > 1.0
        with np.errstate(invalid="ignore"):
            base = {"spd": np.nanmean(sp, 1), "align": np.nanmean(np.where(mov, cos, np.nan), 1),
                    "toward": np.nanmean(np.where(np.isfinite(sp), mov & (cos > 0.7), np.nan), 1)}
        cols, nm = [], []
        for k, x in base.items():
            for a, b in WINS:
                cols.append(win_mean(x, a, b)); nm.append("%s_%d_%d" % (k, a, b))
        names = nm
        g = lab < 3
        X.append(np.stack(cols, 1)[g][::4]); Y.append((lab[g] == 2).astype(int)[::4]); M.append(np.full(g[::4].sum() if False else len(lab[g][::4]), mi))
X, Y, M = np.concatenate(X), np.concatenate(Y), np.concatenate(M)
ok = np.isfinite(X).all(1); X, Y, M = X[ok], Y[ok], M[ok]
def lomo(cols, kind):
    idx = [names.index(c) for c in cols]; aucs = []
    for m in range(len(dfl.MATCHES)):
        tr, te = M != m, M == m
        if kind == "lr":
            clf = LogisticRegression(max_iter=1000).fit(X[tr][:, idx], Y[tr]); p = clf.predict_proba(X[te][:, idx])[:, 1]
        else:
            bst = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, seed=0, num_threads=8),
                            lgb.Dataset(X[tr][:, idx], Y[tr]), 300)
            p = bst.predict(X[te][:, idx])
        aucs.append(roc_auc_score(Y[te], p))
    return np.mean(aucs)
one = lambda ks: ["%s_-25_25" % k for k in ks]
multi = lambda ks: [n for n in names if n.split("_")[0] in ks]
for label, cols in (("speed, 1 window", one(["spd"])), ("speed+attention, 1 window", one(["spd", "align", "toward"])),
                    ("speed, 4 windows", multi(["spd"])), ("speed+attention, 4 windows", multi(["spd", "align", "toward"]))):
    print("%-28s logistic AUC %.4f | LightGBM AUC %.4f" % (label, lomo(cols, "lr"), lomo(cols, "gbm")), flush=True)
