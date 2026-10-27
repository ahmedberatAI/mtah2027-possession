"""Round 9: training-side variants of v10 (DFL native + Metrica ctrl weight 2, v7 + ball-free features, +-25-frame
logit smoothing, v7 decoder), one switch at a time via the environment:

  V10_SOFT=k    soft team targets: the share of H among the alive frames within +-k frames (cross-entropy objective)
                instead of the hard H / A label -- the model learns the smoothed target the decoder works best with
  V10_KEEP=s    Metrica training labels put into the controlled convention with keep_s = s (default 1.0; evaluation
                always uses the standard 1.0)
  V10_LGB=json  LightGBM parameter overrides, e.g. '{"extra_trees": true}' or '{"feature_fraction": 0.5}'
  V10_MONO=1    monotone constraint +1 on the team's own control share and its window means (T:ctrl, T:ctrl2,
                T:ctrl*_w*): more own control can never lower the possession probability

Same folds as src/exp_combo.py (DFL by match, Metrica by game, an all-data model for SkillCorner). Paired spell F1
against v10 and v7 on DFL-trk, Metrica-ctrl, SkillCorner phase / pp, learned and true D. AGGREGATE output only.

  V10_SOFT=12 python -I src/exp_v10var.py
"""
import functools
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import metrica_sample as ms  # noqa: E402
from mlfeat import win_mean  # noqa: E402

OUT = os.path.join(bench.CACHE, "v10var")
EPS = 1e-4
W_M = 2.0
SOFT = int(os.environ.get("V10_SOFT", "0"))
KEEP = float(os.environ.get("V10_KEEP", "1.0"))
LGB = json.loads(os.environ.get("V10_LGB", "{}"))
MONO = os.environ.get("V10_MONO", "0") == "1"
TAG = "soft%d_keep%g_%s%s" % (SOFT, KEEP, "_".join("%s%s" % kv for kv in sorted(LGB.items())) or "lgb",
                              "_mono" if MONO else "")


@functools.lru_cache(maxsize=None)
def monotone():
    import exp_domain
    return [1 if c in ("T:ctrl", "T:ctrl2") or (c.startswith("T:ctrl") and "_w" in c) else 0
            for c in exp_domain.names()]


def metrica_labels():
    if KEEP == 1.0:
        return bench.labels("metrica@v3", "ctrl")
    per = {}
    for g in ms.GAMES:
        d, ev = ms.parse(g), ms.events(g)
        per[g] = [ms.merge_uncontrolled(d["h%d_labels" % h], ev[ev["period"] == h], int(d["h%d_n0" % h][0]), KEEP)
                  for h in (1, 2)]
    return [per[g][h] for g in ms.GAMES for s in bench.M_SEEDS for h in (0, 1)]


def rows(feats, labs, w):
    Xt, yt, _ = bench.team_rows(feats, labs)
    if SOFT:
        y = np.concatenate([lab[::bench.STRIDE] for lab in labs])
        s = np.concatenate([win_mean(np.where(lab == bench.H, 1.0, np.where(lab == bench.A, 0.0, np.nan)),
                                     -SOFT, SOFT)[::bench.STRIDE] for lab in labs])
        alive = (y == bench.H) | (y == bench.A)
        yt = np.concatenate([s[alive], 1 - s[alive]])
    return Xt, yt, np.full(len(yt), w, np.float32)


def fit(parts):
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2] for p in parts])
    params = dict(bench.LGB_TEAM, num_threads=bench.THREADS, **LGB)
    if SOFT:
        params["objective"] = "cross_entropy"
    if MONO:
        params["monotone_constraints"] = list(monotone())
    return lgb.train(params, lgb.Dataset(X, y, weight=w), bench.ROUNDS)


def probs():
    path = os.path.join(OUT, "probs_%s.pkl" % TAG)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), metrica_labels()
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = fit([rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = fit([d_all, rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    print("variant", TAG, flush=True)
    raw = probs()
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    ws = (0, 12, 25) if SOFT else (25,)
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        acc = e2.team_acc(raw[src], labs)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            for w in ws:
                v = e2.score(src, conv, e2.decode_all(dec, pds, exp_smooth.smooth(raw[src], w)))
                cells.append("%s w%d %.4f (vs v10 %+.4f, %d/%d; vs v7 %+.4f)" % (
                    dname, w, v.mean(), v.mean() - r10.mean(), int((v > r10).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s acc %.4f | %s" % (src, conv, acc, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
