"""Round 9: ball-touch event features (src/touch.py; 10 per team view) appended to v10's inputs (v7 + ball-free
features), v10's training (DFL native + Metrica ctrl weight 2) and +-25-frame smoothing. Same folds as
src/exp_combo.py; paired spell F1 against v10 and v7 on DFL-trk, Metrica-ctrl, SkillCorner phase / pp, learned and
true D. AGGREGATE output only.

  python -I src/exp_touch.py [sweep]      (sweep: decode-only smoothing windows 12 / 25 / 37 / 50)
  TOUCH_V=2 python -I src/exp_touch.py     (touch.touch_feats2: 16 columns per view)
  LGB_SEED=1 python -I src/exp_touch.py    (seed check against the seed-1 v10 run)
  TOUCH_DFL=v4 python -I src/exp_touch.py  (DFL touch columns from its pre-smoothed V4 ball)
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import touch  # noqa: E402

OUT = os.path.join(bench.CACHE, "touch")
EPS = 1e-4
W_M = 2.0
V2 = os.environ.get("TOUCH_V") == "2"      # touch.touch_feats2 (16 columns: + kick strength, gains, chain maxima)
DFL4 = os.environ.get("TOUCH_DFL") == "v4"  # DFL's touch columns from its pre-smoothed (V4) ball, as smooth as BAR-ATH's
FSUF = "2" if V2 else ""
SUF = FSUF + ("_seed%d" % exp_a2.SEED if exp_a2.SEED else "") + ("_dflv4" if DFL4 else "")   # LGB_SEED=k: seed check


def _job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    fn = touch.touch_feats2 if V2 else touch.touch_feats
    return (i,) + fn(b["P"], b["B"], b["team"], b["gk"], b["pitch"])


def extra(src):
    path = os.path.join(OUT, "feat%s_%s.pkl" % (FSUF, src.replace("@", "-")))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_job, [(src, i) for i in range(len(bench.blocks(src)))]), key=lambda r: r[0])
    out = [(r[1], r[2]) for r in res]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def aug_feats(src):
    tsrc = "dfl@v4" if DFL4 and src == "dfl@v3" else src      # V4 = V3's masks / noise on smoothed tracks: aligned
    return [dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a]))
            for f, (h, a) in zip(exp_combo.aug_feats(src), extra(tsrc))]


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
    print("touch columns' gain share in the all-data model: %.3f" % (g[-(16 if V2 else 10):].sum() / g.sum()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    mdl.save_model(os.path.join(OUT, "all%s.txt" % SUF))
    bench._dump(res, path)
    return res


def v10_ref():
    """v10's smoothed probabilities from the same LightGBM seed (seed 1: src/exp_combo.py LGB_SEED=1 COMBO_WM=2)."""
    if not exp_a2.SEED:
        return bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_seed%d_wm2.pkl" % exp_a2.SEED))
    return {k: exp_smooth.smooth(v, 25) for k, v in raw.items()}


def sweep():
    """Decode-only: the smoothing window for the touch model (v10's +-25 was chosen without touch features)."""
    raw = probs()
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            res = {"v10": e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))}
            for w in (12, 25, 37, 50):
                res["touch w%d" % w] = e2.score(src, conv, e2.decode_all(dec, pds, exp_smooth.smooth(raw[src], w)))
            rows[ctx] = {k: v.mean() for k, v in res.items()}
            print("%-24s v10 %.4f | %s" % (ctx, res["v10"].mean(), " | ".join(
                "%s %+.4f (%d/%d)" % (k, v.mean() - res["v10"].mean(), int((v > res["v10"]).sum()), len(v))
                for k, v in res.items() if k != "v10")), flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in rows[next(iter(rows))]}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        return sweep()
    new = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    v10 = v10_ref()
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v10 %.4f -> %.4f (vs v10 %+.4f, %d/%d; vs v7 %+.4f)" % (
                dname, r10.mean(), v.mean(), v.mean() - r10.mean(), int((v > r10).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s acc %.4f (v10 %.4f) | %s" % (src, conv, e2.team_acc(new[src], labs),
                                                      e2.team_acc(v10[src], labs), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
