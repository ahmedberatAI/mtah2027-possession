"""Do players' movements stay 'locked' on the ball while play is alive? Open DFL data (clean).
Per frame: alignment = mean over moving players (>1 m/s) of cos(angle between velocity and the
direction to the ball); also the share of players moving towards the ball. Compared with tempo.
AGGREGATE output."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import dfl
from mlfeat import win_mean
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.stdout.reconfigure(encoding="utf-8")
rows = {"spd": [], "align": [], "toward": [], "y": [], "m": []}
curves = {"spd": [], "align": [], "toward": []}
for mi, m in enumerate(dfl.MATCHES):
    d = dfl.parse(m); L, W = d["pitch"]
    for h in (1, 2):
        t = "h%d" % h
        P = d[t + "_players"][:, :, :2] * np.array([L, W], np.float32)
        B = d[t + "_ball"][:, :2] * np.array([L, W], np.float32)
        lab = d[t + "_labels"]
        v = np.full(P.shape, np.nan, np.float32); v[1:-1] = (P[2:] - P[:-2]) * 12.5
        sp = np.linalg.norm(v, axis=2)
        to = B[:, None, :] - P
        dist = np.linalg.norm(to, axis=2)
        cos = np.sum(v * to, axis=2) / np.maximum(sp * dist, 1e-6)
        mov = sp > 1.0
        with np.errstate(invalid="ignore"):
            align = np.nanmean(np.where(mov, cos, np.nan), 1)
            toward = np.nanmean(np.where(np.isfinite(sp), mov & (cos > 0.7), np.nan), 1)
            spd = np.nanmean(sp, 1)
        F = {"spd": win_mean(spd, -25, 25), "align": win_mean(align, -25, 25), "toward": win_mean(toward, -25, 25)}
        g = lab < 3
        for k in F: rows[k].append(F[k][g])
        rows["y"].append((lab[g] == 2).astype(int)); rows["m"].append(np.full(g.sum(), mi))
        starts = np.flatnonzero((lab[1:] == 2) & (lab[:-1] < 2)) + 1
        for s in starts:
            if 125 <= s < len(lab) - 125:
                for k in curves: curves[k].append(F[k][s - 125:s + 126:25])
R = {k: np.concatenate(v) for k, v in rows.items()}
ok = np.isfinite(R["spd"]) & np.isfinite(R["align"]) & np.isfinite(R["toward"])
y = R["y"][ok]
print("frames %d, dead share %.2f" % (ok.sum(), y.mean()))
for k in ("spd", "align", "toward"):
    x = R[k][ok]
    print("%-6s alive mean %.3f | dead mean %.3f | AUC(dead) %.3f" % (k, x[y == 0].mean(), x[y == 1].mean(), roc_auc_score(y, -x)))
# leave-one-match-out logistic: speed only vs speed + attention
for feats in (["spd"], ["spd", "align"], ["spd", "align", "toward"]):
    aucs = []
    X = np.stack([R[k][ok] for k in feats], 1); mm = R["m"][ok]
    for m in range(len(dfl.MATCHES)):
        tr, te = mm != m, mm == m
        clf = LogisticRegression(max_iter=500).fit(X[tr][::5], y[tr][::5])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    print("LOMO logistic %-22s AUC %.4f" % ("+".join(feats), np.mean(aucs)))
print("\naround D starts (n=%d): mean value at offset (s)" % len(curves["spd"]))
print("offset  " + " ".join("%6.0f" % o for o in range(-5, 6)))
for k in curves:
    c = np.nanmean(np.array(curves[k]), 0)
    print("%-6s  " % k + " ".join("%6.3f" % x for x in c))
