"""Round 9: team-model TRAINING variants (no new features), v7 features / sim V3 / v7 decoder:

  trk       DFL tracking-derived controlled labels instead of native (idea A1, first tried with sim V2 + old decoder)
  bsmooth   native labels, frames within +-6 of an in-play team change weighted 0.3 (boundary-label uncertainty, a16)
  spellbal  native labels, frame weight (median spell length / own spell length)^0.5 clipped to [0.5, 3] (R-8, a03)
DFL leave-one-match-out, the all-DFL model on Metrica / SkillCorner; paired spell F1 against v7 (learned and true D)
on DFL-trk, Metrica-ctrl, SkillCorner phase / player-possession. AGGREGATE output only.

  python -I src/exp_trainvar.py <variant>
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import model_v7  # noqa: E402

OUT = os.path.join(bench.CACHE, "trainvar")
EPS = 1e-4
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("sc@v3", "pp", (30.0, 8, 50)))


def weights(lab, variant):
    w = np.ones(len(lab), np.float32)
    if variant == "bsmooth":
        t = np.flatnonzero((lab[1:] < 2) & (lab[:-1] < 2) & (lab[1:] != lab[:-1])) + 1
        cover = np.zeros(len(lab) + 1, np.int32)
        np.add.at(cover, np.maximum(t - 6, 0), 1)
        np.add.at(cover, np.minimum(t + 6, len(lab)), -1)
        w[np.cumsum(cover[:-1]) > 0] = 0.3
    elif variant == "spellbal":
        for a, b, _ in exp_oracle.spells(lab):
            w[a:b] = np.clip((175.0 / max(b - a, 1)) ** 0.5, 0.5, 3.0)
    return w


def fit(feats, labs, variant):
    XF, XH, XA, y, wt = [], [], [], [], []
    for f, lab in zip(feats, labs):
        sl = slice(0, None, bench.STRIDE)
        XF.append(f["XF"][sl]), XH.append(f["XH"][sl]), XA.append(f["XA"][sl]), y.append(lab[sl])
        wt.append(weights(lab, variant)[sl])
    XF, XH, XA, y, wt = (np.concatenate(XF), np.concatenate(XH), np.concatenate(XA), np.concatenate(y),
                         np.concatenate(wt))
    alive = y < 2
    Xt = np.concatenate([np.hstack([XF[alive], XH[alive]]), np.hstack([XF[alive], XA[alive]])])
    yt = np.concatenate([y[alive] == 0, y[alive] == 1]).astype(int)
    w = np.concatenate([wt[alive], wt[alive]])
    return lgb.train(dict(bench.LGB_TEAM, num_threads=bench.THREADS), lgb.Dataset(Xt, yt, weight=w), bench.ROUNDS)


def probs(variant):
    path = os.path.join(OUT, "probs_%s.pkl" % variant)
    if os.path.isfile(path):
        return bench._load(path)
    fd = bench.features(model_v7.VERSION, "dfl@v3")
    labs = bench.labels("dfl", "trk" if variant == "trk" else "native")
    bl = bench.blocks("dfl@v3")
    res = {"dfl@v3": [None] * len(bl)}
    for m in sorted({b["group"] for b in bl}):
        tr = [i for i, b in enumerate(bl) if b["group"] != m]
        mdl = fit([fd[i] for i in tr], [labs[i] for i in tr], variant)
        for i, b in enumerate(bl):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  %s fold %s done" % (variant, m), flush=True)
    mdl = fit(fd, labs, variant)
    del fd
    for src in ("metrica@v3", "sc@v3"):
        res[src] = [bench.predict_team(mdl, f) for f in bench.features(model_v7.VERSION, src)]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    variant = sys.argv[1]
    new = probs(variant)
    for src, conv, dec in SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v7 %.4f -> %.4f (%+.4f, better in %d/%d)" % (dname, ref.mean(), v.mean(),
                                                                         v.mean() - ref.mean(), int((v > ref).sum()),
                                                                         len(v)))
        print("%-9s %-10s %-6s frame team acc v7 %.4f -> %.4f | %s" % (
            variant, src, conv, e2.team_acc(phl, labs), e2.team_acc(new[src], labs), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
