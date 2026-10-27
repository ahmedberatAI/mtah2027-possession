"""Round 9: combine the two robust gains of the night -- A2 (Metrica games in the team model's training,
src/exp_a2.py) and the ball-free interaction features (src/bfree.py) whose spell gain appears once the team
probabilities are smoothed (src/exp_smooth.py: +-25 frames in logit space).

  A2           cached probabilities of exp_a2.py (v7 features, DFL + Metrica, W_M 1)
  A2+bfree     v7 + ball-free features, DFL + Metrica training (same protocol as exp_a2.py)
each decoded after logit smoothing over +-w frames (w = 0 / 12 / 25 / 50); paired spell F1 against UNSMOOTHED v7 on
DFL-trk, Metrica-ctrl, SkillCorner phase / player-possession (learned and true D). AGGREGATE output only.

  python -I src/exp_combo.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_bfree  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

OUT = os.path.join(bench.CACHE, "combo")
EPS = 1e-4
WS = (0, 12, 25, 50)
W_M = float(os.environ.get("COMBO_WM", "1.0"))    # Metrica weight per game (v9: 1)
NORM = os.environ.get("COMBO_NORM") == "1"         # ball-free columns robust-scaled within each half
DFL_CONV = os.environ.get("COMBO_DFLCONV", "native")   # DFL training labels (v9 / v10: native)
M_CONV = os.environ.get("COMBO_MCONV", "ctrl")         # Metrica training labels (v9 / v10: ctrl; round 12: events)
TAG = ("" if W_M == 1.0 else "_wm%g" % W_M) + ("_norm" if NORM else "") + ("" if DFL_CONV == "native" else "_" + DFL_CONV)
TAG += "" if M_CONV == "ctrl" else "_m" + M_CONV
TAG += ("_l%d" % exp_a2.LEAVES if exp_a2.LEAVES else "") + ("_r%d" % exp_a2.ROUNDS if exp_a2.ROUNDS else "")


def _norm(x):
    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    return ((x - med) / np.where(iqr > 1e-6, iqr, 1.0)).astype(np.float32)


def aug_feats(src):
    ex = exp_bfree.extra(src)
    out = []
    for b, (h, a) in zip(bench.blocks(src), ex):
        f = bench._load(e2._feat_path(b))
        if NORM:
            h, a = _norm(h), _norm(a)
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
    return out


def probs_a2_bfree():
    path = os.path.join(OUT, "probs_a2_bfree%s%s.pkl" % ("" if exp_a2.SEED == 0 else "_seed%d" % exp_a2.SEED, TAG))
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = aug_feats("dfl@v3"), aug_feats("metrica@v3")
    if DFL_CONV.startswith("tau0"):          # e.g. tau04: native labels with spells < 0.4 s merged (micro-spells)
        import metrica_sample as ms
        ld = [ms.merge_short(l, float("0." + DFL_CONV[4:])) for l in bench.labels("dfl", "native")]
    else:
        ld = bench.labels("dfl", DFL_CONV)
    lm = bench.labels("metrica@v3", M_CONV)
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
    mdl.save_model(os.path.join(OUT, "a2_bfree_all%s%s.txt" % ("" if exp_a2.SEED == 0 else "_seed%d" % exp_a2.SEED,
                                                              TAG)))
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in aug_feats("sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    models = {"A2+bfree seed %d%s" % (exp_a2.SEED, TAG): probs_a2_bfree()}
    if exp_a2.SEED == 0 and not TAG:
        models["A2"] = bench._load(os.path.join(bench.CACHE, "a2", "probs_wm1.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            cells = []
            for name, res in models.items():
                for w in WS:
                    v = e2.score(src, conv, e2.decode_all(dec, pds, exp_smooth.smooth(res[src], w)))
                    cells.append("%s w%d %+.4f (%d/%d)" % (name, w, v.mean() - ref.mean(), int((v > ref).sum()),
                                                         len(v)))
            print("%-10s %-6s %-9s v7 %.4f | %s" % (src, conv, dname, ref.mean(), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
