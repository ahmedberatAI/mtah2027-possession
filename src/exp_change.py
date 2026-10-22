"""Round 6, idea 2 (docs/FIKIRLER.md): a possession-change model and a time-dependent team-switch cost.

A LightGBM learns "is there an in-play change of possession within +-K frames of t?" on degraded DFL with the
controlled-possession labels (trk), leave one match out. Inputs: v7's features plus direction-free cues from
v7's team probability around t (|mean logit after - before| at several widths, local spread, sign flips). The
decoder (v7's B2) then pays pen_team - lam * (logit p_change(t) - bias) for switching team at t (clipped so the
cost stays >= 0): a switch is cheap where a change is likely and expensive elsewhere. AGGREGATE output only.

  python -I src/exp_change.py
"""
import itertools
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402
from mlfeat import win_mean  # noqa: E402

K = 6
STRIDE = 2
OUT = os.path.join(bench.CACHE, "change")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
              force_row_wise=True, num_threads=bench.THREADS)
ROUNDS = 300
SETS = exp_oracle.SETS
EPS = 1e-4


def zfeat(ph):
    """Direction-free change cues from a team-probability track."""
    z = np.log(np.clip(ph, EPS, 1 - EPS) / np.clip(1 - ph, EPS, 1 - EPS))
    cols = [np.abs(z)]
    for w in (6, 12, 25, 50, 100):
        before, after = win_mean(z, -w, -1), win_mean(z, 1, w)
        cols += [np.abs(after - before), -(before * after)]
    sgn = np.sign(z)
    flip = np.concatenate(([0.0], (sgn[1:] != sgn[:-1]).astype(np.float64)))
    cols += [win_mean(flip, -25, 25), win_mean(flip, -100, 100), win_mean(np.abs(np.diff(z, prepend=z[0])), -12, 12)]
    return np.stack(cols, 1).astype(np.float32)


def target(lab):
    y = np.zeros(len(lab), np.int8)
    for t in exp_oracle.changes(lab):
        y[max(0, t - K):t + K + 1] = 1
    return y


def rows(f, ph, lab=None, swap=False):
    zf = zfeat(ph)
    XH, XA = (f["XA"], f["XH"]) if swap else (f["XH"], f["XA"])
    X = np.hstack([f["XF"], XH, XA, zf])
    return X if lab is None else (X, target(lab))


def fit(feats, phs, labs):
    Xs, ys = [], []
    for f, ph, lab in zip(feats, phs, labs):
        alive = lab < 2
        for sw in (False, True):
            X, y = rows(f, ph, lab, sw)
            sel = alive & (np.arange(len(lab)) % STRIDE == 0)
            Xs.append(X[sel]), ys.append(y[sel])
    return lgb.train(PARAMS, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), ROUNDS)


def predict(model, f, ph):
    return 0.5 * (model.predict(rows(f, ph)) + model.predict(rows(f, ph, swap=True)))


def spec_change(pen_team, dec):
    ramp = exp_v7.ramp(175, 48.0)
    macros = [dict(emis=0, out=0, L=len(ramp), min_exit=dec[1] - 1, exit_cost=ramp),
              dict(emis=1, out=1, L=len(ramp), min_exit=dec[1] - 1, exit_cost=ramp),
              dict(emis=2, out=2, L=dec[2])]
    pen = [[0, pen_team, dec[0]], [pen_team, 0, dec[0]], [dec[0], dec[0], 0]]
    return decode2.Spec(macros, pen, [[-1, 0, -1], [0, -1, -1], [-1, -1, -1]])


def chan(pc, pen_team, lam, bias):
    lg = np.log(np.clip(pc, EPS, 1 - EPS) / np.clip(1 - pc, EPS, 1 - EPS))
    return np.maximum(-lam * (lg - bias), -pen_team)[:, None]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    V = model_v7.VERSION
    data = {}
    for src, conv, dec in SETS:
        phl, pdl = exp_oracle.probs(src)
        data[src] = dict(feats=bench.features(V, src), ph=phl, pd=pdl, gts=bench.labels(src, conv), dec=dec,
                         conv=conv)
    d = data["dfl@v3"]
    bl = bench.blocks("dfl@v3")
    path = os.path.join(OUT, "pc.pkl")
    if os.path.isfile(path):
        pcs = bench._load(path)
    else:
        pcs = {"dfl@v3": [None] * len(bl)}
        for m in exp_oracle_matches(bl):
            tr = [i for i, b in enumerate(bl) if b["group"] != m]
            mdl = fit([d["feats"][i] for i in tr], [d["ph"][i] for i in tr], [d["gts"][i] for i in tr])
            for i, b in enumerate(bl):
                if b["group"] == m:
                    pcs["dfl@v3"][i] = predict(mdl, d["feats"][i], d["ph"][i]).astype(np.float32)
            print("  change model fold", m, flush=True)
        full = fit(d["feats"], d["ph"], d["gts"])
        full.save_model(os.path.join(OUT, "change_all_dfl.txt"))
        for src in ("metrica@v3", "sc@v3"):
            pcs[src] = [predict(full, f, ph).astype(np.float32) for f, ph in zip(data[src]["feats"], data[src]["ph"])]
        bench._dump(pcs, path)
    # quality of the change model: AUC on alive frames (direction-free)
    from sklearn.metrics import roc_auc_score
    for src, x in data.items():
        y = np.concatenate([target(g)[g < 2] for g in x["gts"]])
        p = np.concatenate([pc[g < 2] for pc, g in zip(pcs[src], x["gts"])])
        print("%-11s change model AUC %.4f (positives %.3f)" % (src, roc_auc_score(y, p), y.mean()), flush=True)
    grid = list(itertools.product((12.0, 20.0, 32.0), (0.5, 1.0, 2.0, 4.0), (-2.0, 0.0, 2.0)))
    for src, x in data.items():
        bl_, gts = bench.blocks(src), x["gts"]
        for dname, pds in (("learned D", x["pd"]), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in gts])):
            if src == "dfl@v3" and dname == "true D":
                continue
            S = [td.emissions(a, b, 1.0, 1.0) for a, b in zip(pds, x["ph"])]
            ref = None
            res = []
            sp0 = decode2.duration_spec(12.0, x["dec"][0], x["dec"][1], x["dec"][2], exp_v7.ramp(175, 48.0))
            r = bench.spell_by_group(bl_, gts, [sp0.decode(s).astype(np.int8) for s in S])
            ref = np.array([r[k][0] for k in sorted(r)])
            for pt, lam, bias in grid:
                sp = spec_change(pt, x["dec"])
                preds = [sp.decode(s, chan(pc, pt, lam, bias)).astype(np.int8) for s, pc in zip(S, pcs[src])]
                r = bench.spell_by_group(bl_, gts, preds)
                v = np.array([r[k][0] for k in sorted(r)])
                res.append(((pt, lam, bias), v))
            res.sort(key=lambda t: -t[1].mean())
            print("%-11s %-9s v7 %.4f | best: %s" % (src, dname, ref.mean(), " | ".join(
                "pen %g lam %g bias %+g: %.4f (%+.4f, %d/%d)" % (k + (v.mean(), v.mean() - ref.mean(),
                                                                   int((v > ref).sum()), len(v))) for k, v in res[:3])),
                  flush=True)
            bench._dump(res, os.path.join(OUT, "grid_%s_%s.pkl" % (src.replace("@", "-"), dname.split()[0])))


def exp_oracle_matches(bl):
    out = []
    for b in bl:
        if b["group"] not in out:
            out.append(b["group"])
    return out


if __name__ == "__main__":
    main()
