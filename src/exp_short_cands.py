"""Round 15 diagnostic, deployable view: can v10's OWN short-possession candidates be verified? AGGREGATE output only.

src/exp_short_angles.py describes true short spells over their true intervals (optimistic: true boundaries and the
true restart time leak). Here every quantity comes from the model's output, as it would at inference. v10 seed 0,
+-25 smoothing, learned D, shipped B2 decoder (c 48); per half:
  ins  suppressed flips: runs >= 12 frames inside decoded possessions where the +-25 probability favours the other team
  del  decoded possessions shorter than 4 s
Label from the truth: positive when >= 50% of the candidate's frames are the candidate team's and a true spell of that
team overlaps it with IoU >= 0.3; negative when < 20% are; otherwise ambiguous (left out). Features are those of
exp_short_angles over the candidate interval, with the time since the last DECODED restart. Reports per context:
counts, how many true in-play short spells (between two opponent spells) a positive candidate reaches, univariate
AUCs, leave-one-match-out LightGBM AUC and precision at fixed recall, for ins and del separately.

  python -I src/exp_short_cands.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402

CONTEXTS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)),
            ("sc@v3", "phase", (30.0, 8, 50)), ("sc@v3", "pp", (30.0, 8, 50)))
FEATS = tuple(f for f in ea.FEATS if f != "since_restart") + ("since_dead",)


def best_iou(gt, s, t, k):
    best = 0.0
    for a, b, kk in eo.spells(gt[max(s - 500, 0):t + 500]):
        a, b = a + max(s - 500, 0), b + max(s - 500, 0)
        if kk == k and b > s and a < t:
            best = max(best, (min(b, t) - max(a, s)) / (max(b, t) - min(a, s)))
    return best


def collect(src, conv, dec):
    blocks, gts = bench.blocks(src), bench.labels(src, conv)
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))[src]
    smo = exp_smooth.smooth(raw, 25)
    pds = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
    preds = es.decode(dec, 48.0, pds, smo)
    gru = None
    if not src.startswith("dfl"):
        path = os.path.join(os.path.dirname(bench.CACHE), "shared_blend_v1", src.replace("@", "-") + "_members.pkl")
        gru = bench._load(path)["members"][0]
    rows, n_inplay, reach = [], 0, 0
    for i, (b, gt, pr) in enumerate(zip(blocks, gts, preds)):
        c = ea.frame_cues(b)
        ms, mr = ea.logit(smo[i]), ea.logit(raw[i])
        mg = ea.logit(gru[i]) if gru is not None else None
        fav = np.where(ms > 0, 0, 1)
        restart = np.flatnonzero((pr[1:] < 2) & (pr[:-1] == 2)) + 1
        cands = []
        for a, e, k in eo.spells(pr):
            for s, t in ea.runs(fav[a:e] != k):
                if t - s >= ea.RUN:
                    cands.append((a + s, a + t, 1 - k, "ins"))
            if e - a < ea.SHORT:
                cands.append((a, e, k, "del"))
        pos_iv = []
        for s, t, k, typ in cands:
            frac = float(np.mean(gt[s:t] == k))
            iou = best_iou(gt, s, t, k)
            y = 1 if (frac >= 0.5 and iou >= 0.3) else 0 if frac < 0.2 else None
            if y is None:
                continue
            f = ea.item(c, gt, ms, mr, mg, s, t, k)
            r = restart[restart <= s]
            f["since_dead"] = min((s - r[-1]) / ea.FPS, 60.0) if len(r) else 60.0
            f.update(y=y, typ=typ, group=str(b["group"]))
            rows.append(f)
            if y == 1:
                pos_iv.append((s, t, k))
        sp = eo.spells(gt)
        for j in range(1, len(sp) - 1):
            a, e, k = sp[j]
            if e - a < ea.SHORT and sp[j - 1][2] == 1 - k and sp[j + 1][2] == 1 - k and \
                    sp[j - 1][1] == a and sp[j + 1][0] == e:
                n_inplay += 1
                reach += any(kk == k and min(t, e) - max(s, a) > 0 and
                             (min(t, e) - max(s, a)) / (max(t, e) - min(s, a)) >= 0.3 for s, t, kk in pos_iv)
    return rows, n_inplay, reach


def lomo(rows, keys):
    y = np.array([r["y"] for r in rows])
    X = np.array([[r[k] for k in keys] for r in rows], float)
    g = np.array([r["group"] for r in rows])
    score = np.full(len(y), np.nan)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0)
    for grp in np.unique(g):
        tr, te = g != grp, g == grp
        if len(np.unique(y[tr])) < 2:
            continue
        score[te] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300).predict(X[te])
    ok = np.isfinite(score)
    order = np.argsort(-score[ok])
    yy = y[ok][order]
    tp = np.cumsum(yy)
    prec, rec = tp / np.arange(1, len(yy) + 1), tp / max(yy.sum(), 1)
    pts = ["R%.1f P%.2f" % (r, prec[np.searchsorted(rec, r)]) for r in (0.3, 0.5, 0.7, 0.9)
           if np.searchsorted(rec, r) < len(prec)]
    return ea.auc(score[ok][y[ok] == 1], score[ok][y[ok] == 0]), pts


def report(src, conv, rows, n_inplay, reach):
    print("\n=== %s %s: true in-play short spells %d, reached by a positive candidate %d (%.0f%%) ===" % (
        src.split("@")[0], conv, n_inplay, reach, 100 * reach / max(n_inplay, 1)))
    for typ in ("ins", "del"):
        rs = [r for r in rows if r["typ"] == typ]
        if not rs:
            continue
        pos = [r for r in rs if r["y"] == 1]
        neg = [r for r in rs if r["y"] == 0]
        print("  %s: %d candidates, positive %d (%.0f%%)" % (typ, len(rs), len(pos), 100 * len(pos) / len(rs)))
        cells = []
        for key in FEATS:
            a_ = np.array([r[key] for r in pos], float)
            b_ = np.array([r[key] for r in neg], float)
            if np.isfinite(a_).any():
                cells.append("%s %.2f|%.2f AUC %.2f" % (key, np.nanmedian(a_), np.nanmedian(b_), ea.auc(a_, b_)))
        for i in range(0, len(cells), 4):
            print("    " + " ; ".join(cells[i:i + 4]))
        keys = [k for k in FEATS if np.isfinite(np.array([r[k] for r in rs], float)).any()]
        a_all, pts = lomo(rs, keys)
        a_m, _ = lomo(rs, ["m_s"])
        print("    LOMO LightGBM AUC all features %.3f (v10 margin alone %.3f) | %s" % (a_all, a_m, " ".join(pts)))
    sys.stdout.flush()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, conv, dec in CONTEXTS:
        rows, n_inplay, reach = collect(src, conv, dec)
        report(src, conv, rows, n_inplay, reach)
        bench._blocks.clear()
        bench._labels.clear()


if __name__ == "__main__":
    main()
