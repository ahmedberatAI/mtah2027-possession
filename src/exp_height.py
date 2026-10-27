"""Round 11, idea E7 (docs/RAPOR.md), oracle first: does knowing the BALL HEIGHT help the team model at the contested
moments (aerial duels and second balls are the hardest turnovers, tur 10)? Open DFL tracking has the real ball height
and SkillCorner an estimated one; Metrica has none (NaN). Height columns are appended to v10's frame features (XF),
v10's training (DFL native + Metrica ctrl weight 2, LOMO / leave-game-out folds as src/exp_touch.py) and +-25 smoothing,
paired spell F1 against v10 (same LightGBM seed) on DFL-trk and SkillCorner phase / pp, learned and true D.
Only if the oracle is clearly positive is a height estimate for BAR-ATH worth building (two-camera parallax between
the tactical and the broadcast ball). AGGREGATE output only.

    python -I src/exp_height.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import dfl  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import exp_touch  # noqa: E402
import skillcorner as scr  # noqa: E402

OUT = os.path.join(bench.CACHE, "height")
EPS = 1e-4
W_M = 2.0
SUF = "_seed%d" % exp_a2.SEED if exp_a2.SEED else ""


def _win(x, w, fn):
    n = len(x)
    out = np.full(n, np.nan, np.float32)
    xp = np.concatenate([np.full(w, np.nan), x, np.full(w, np.nan)])
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(xp, 2 * w + 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out[:] = fn(v, axis=1)
    return out


def height_cols(z):
    """z (n,) metres, NaN unknown -> (n, 7): z, max over +-12 / +-25, min over +-12, vertical speed, airborne share
    over +-12 (z > 1 m), frames since the ball was last low (z < 0.5 m, capped 50)."""
    z = np.asarray(z, np.float32)
    vz = np.full(len(z), np.nan, np.float32)
    vz[2:-2] = (z[4:] - z[:-4]) / (4 / 25.0)
    air = np.where(np.isfinite(z), (z > 1.0).astype(np.float32), np.nan)
    low = np.isfinite(z) & (z < 0.5)
    since = np.zeros(len(z), np.float32)
    run = 50
    for i in range(len(z)):
        run = 0 if low[i] else min(50, run + 1)
        since[i] = run
    since[~np.isfinite(z)] = np.nan
    return np.stack([z, _win(z, 12, np.nanmax), _win(z, 25, np.nanmax), _win(z, 12, np.nanmin), vz,
                     _win(air, 12, np.nanmean), since], 1)


def heights(src):
    path = os.path.join(OUT, "z_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    out = []
    for b in bench.blocks(src):
        n = len(b["P"])
        if src.startswith("dfl"):
            m, h = b["group"], int(b["key"].split("/h")[-1])
            z = dfl.parse(m)["h%d_ball" % h][:n, 3]
        elif src.startswith("sc"):
            m, h = b["group"], int(b["key"].split("/h")[-1])
            z = scr.parse(m)["h%d_ball" % h][:n, 3]
        else:
            z = np.full(n, np.nan)
        z = np.where(np.isfinite(b["B"][:, 0]), z, np.nan)     # height only where the (degraded) ball is seen
        out.append(height_cols(z))
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def aug_feats(src):
    return [dict(XF=np.hstack([f["XF"], hz]), XH=f["XH"], XA=f["XA"])
            for f, hz in zip(exp_combo.aug_feats(src), heights(src))]


def probs():
    path = os.path.join(OUT, "probs%s.pkl" % SUF)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = aug_feats("dfl@v3"), aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = exp_a2.rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  DFL fold %s done" % m, flush=True)
    d_all = exp_a2.rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  Metrica fold %s done" % g, flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in aug_feats("sc@v3")]
    g = mdl.feature_importance("gain")
    print("height columns' gain share in the all-data model: %.3f" % (g[-7:].sum() / g.sum()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    new = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    v10 = exp_touch.v10_ref()
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            rh = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v10 %.4f height %+.4f (%d/%d)" % (dname, r10.mean(), rh.mean() - r10.mean(),
                                                               int((rh > r10).sum()), len(rh)))
        print("%-18s %s" % ("%s %s" % (src.split("@")[0], conv), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
