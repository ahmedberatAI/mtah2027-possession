"""Round 7: the player-level controller model (src/ctrl_model.py) as extra team evidence for v7.

1. SkillCorner (20 matches, sim V3 inputs): candidate rows + labels from SkillCorner player possessions; 5-fold
   cross-validation by match -> candidate AUC, top-1 controller accuracy, team accuracy of the pooled evidence.
2. Blend: z = logit(p_v7) + alpha * z_ctrl (z_ctrl smoothed over +-w frames), v7 decoder (B2, pen 12), paired against
   v7 on SkillCorner (player-possession labels and phase labels; true D and learned D; controller out of fold),
   and -- with the controller trained on all of SkillCorner -- on DFL-trk and Metrica-ctrl.
AGGREGATE output only.

  python -I src/exp_ctrl.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import ctrl_model as cm  # noqa: E402
import decode2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_v7  # noqa: E402
import skillcorner as scr  # noqa: E402
import train_dfl as td  # noqa: E402
from mlfeat import win_mean  # noqa: E402

OUT = os.path.join(bench.CACHE, "ctrl")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
              force_row_wise=True, num_threads=bench.THREADS)
ROUNDS = 300
EPS = 1e-4


def _rows_job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    X, fr, sl, tm = cm.rows(b)
    y = None
    if src.startswith("sc"):
        ms_ = scr.matches()
        m, h = ms_[i // 2], i % 2 + 1
        ctrl = cm.sc_controller(m, h, len(b["P"]))
        y = (ctrl[fr] == sl).astype(np.int8)
    return i, X, fr, sl, tm, y


def build_rows(src):
    path = os.path.join(OUT, "rows_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    n = len(bench.blocks(src))
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_rows_job, [(src, i) for i in range(n)]), key=lambda r: r[0])
    out = [dict(X=r[1], fr=r[2], sl=r[3], tm=r[4], y=r[5]) for r in res]
    bench._dump(out, path)
    return out


def evidence(model, blk_rows, n):
    p = model.predict(blk_rows["X"])
    return cm.team_evidence(p, blk_rows["fr"], blk_rows["tm"], n), p


def smooth(z, w):
    return z if w == 0 else win_mean(z, -w, w)


def blend_scores(src, conv, dec, pds, phl, zc, alpha, w):
    sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
    preds = []
    for pd_, ph, z in zip(pds, phl, zc):
        p = np.clip(ph, EPS, 1 - EPS)
        zz = np.log(p / (1 - p)) + alpha * smooth(z, w)
        preds.append(sp.decode(td.emissions(pd_, 1 / (1 + np.exp(-zz)), 1.0, 1.0)).astype(np.int8))
    r = bench.spell_by_group(bench.blocks(src), bench.labels(src, conv), preds)
    return np.array([r[k][0] for k in sorted(r)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    sc_rows = build_rows("sc@v3")
    bl = bench.blocks("sc@v3")
    groups = np.array([i // 2 for i in range(len(bl))])
    folds = [list(range(k, 20, 5)) for k in range(5)]
    oof_p = [None] * len(bl)
    for test in folds:
        tr = [i for i in range(len(bl)) if groups[i] not in test]
        X = np.concatenate([sc_rows[i]["X"] for i in tr])
        y = np.concatenate([sc_rows[i]["y"] for i in tr])
        mdl = lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS)
        for i in range(len(bl)):
            if groups[i] in test:
                oof_p[i] = mdl.predict(sc_rows[i]["X"])
    from sklearn.metrics import roc_auc_score
    y_all = np.concatenate([r["y"] for r in sc_rows])
    p_all = np.concatenate(oof_p)
    top1 = tot = 0
    for r, p in zip(sc_rows, oof_p):
        for f in np.unique(r["fr"][r["y"] == 1]):
            sel = r["fr"] == f
            tot += 1
            top1 += int(r["y"][sel][np.argmax(p[sel])] == 1)
    print("controller model (SkillCorner, 5-fold by match): rows %d, positives %.3f, AUC %.4f, top-1 among candidates "
          "%.4f (frames with a controller %d)" % (len(y_all), y_all.mean(), roc_auc_score(y_all, p_all), top1 / tot, tot),
          flush=True)
    zc_sc = [cm.team_evidence(p, r["fr"], r["tm"], len(b["P"]))[0] for p, r, b in zip(oof_p, sc_rows, bl)]
    full = lgb.train(PARAMS, lgb.Dataset(np.concatenate([r["X"] for r in sc_rows]), y_all), ROUNDS)
    full.save_model(os.path.join(OUT, "ctrl_all_sc.txt"))
    # team accuracy of the pooled evidence alone (alive frames of the player-possession labels)
    labs = bench.labels("sc@v3", "pp")
    acc = np.mean(np.concatenate([((z > 0) == (l == 0))[l < 2] for z, l in zip(zc_sc, labs)]))
    print("pooled controller evidence alone, SkillCorner alive frames: team acc %.4f" % acc, flush=True)
    grid = [(a, w) for a in (0.25, 0.5, 1.0) for w in (0, 12, 25)]
    for src, conv, dec in (("sc@v3", "pp", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
                           ("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50))):
        phl, pdl = exp_oracle.probs(src)
        if src.startswith("sc"):
            zc = zc_sc
        else:
            rws = build_rows(src)
            zc = [cm.team_evidence(full.predict(r["X"]), r["fr"], r["tm"], len(b["P"]))[0]
                  for r, b in zip(rws, bench.blocks(src))]
        variants = [("learned D", pdl)]
        if src != "dfl@v3":
            variants.append(("true D", [np.where(g == 2, 1 - EPS, EPS) for g in bench.labels(src, conv)]))
        for dname, pds in variants:
            ref = blend_scores(src, conv, dec, pds, phl, zc, 0.0, 0)
            cells = []
            for a, w in grid:
                v = blend_scores(src, conv, dec, pds, phl, zc, a, w)
                cells.append("a%.2g w%d %+.4f (%d/%d)" % (a, w, v.mean() - ref.mean(), int((v > ref).sum()), len(v)))
            print("%-10s %-6s %-9s v7 %.4f | %s" % (src, conv, dname, ref.mean(), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
