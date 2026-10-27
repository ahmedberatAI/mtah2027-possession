"""Round 9 diagnostic: is v10's team evidence less lively on BAR-ATH than on the external sets? Per source, over alive
frames (external: true labels; BAR-ATH: v10's decoded labels), with v10's all-data team model
(cache/bench/combo/a2_bfree_all_wm2.txt; for the external sets the out-of-fold probabilities): mean |logit|, share
of uncertain frames (|p - 0.5| < 0.2), sign changes per alive minute of the raw and the +-25-smoothed logit, and
v10's decoded possession rate (and the true one where known). AGGREGATE output only.

  python -I src/exp_sticky.py
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
import exp_domain  # noqa: E402
import exp_oracle  # noqa: E402
import make_submission  # noqa: E402
import model_v10  # noqa: E402
from mlfeat import win_mean  # noqa: E402

FPM = 1500.0


def stats(name, phs, alive, n_pred=None, n_true=None):
    z = np.concatenate([e2.logit(p)[a] for p, a in zip(phs, alive)])
    flips_raw = flips_s = 0
    for p, a in zip(phs, alive):
        zr = e2.logit(p)
        zs = win_mean(zr, -25, 25)
        for zz in (zr, zs):
            s = np.sign(zz)
            ch = (s[1:] != s[:-1]) & a[1:] & a[:-1]
            if zz is zr:
                flips_raw += int(ch.sum())
            else:
                flips_s += int(ch.sum())
    mins = sum(int(a.sum()) for a in alive) / FPM
    line = "%-12s alive %6.1f min | mean |logit| %.2f | uncertain %.3f | sign changes / min raw %.2f, smoothed %.2f" % (
        name, mins, np.abs(z).mean(), (np.abs(e2.sig(z) - 0.5) < 0.2).mean(), flips_raw / mins, flips_s / mins)
    if n_pred is not None:
        line += " | v10 spells / min %.2f" % (n_pred / mins)
    if n_true is not None:
        line += " (true %.2f)" % (n_true / mins)
    print(line, flush=True)


def n_spells(labs):
    return sum(len(exp_oracle.spells(y)) for y in labs)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src, conv, dec in (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)),
                           ("sc@v3", "phase", (30.0, 8, 50))):
        labs = bench.labels(src, conv)
        pdl = exp_oracle.probs(src)[1]
        pred = e2.decode_all(dec, pdl, v10[src])
        stats(src.split("@")[0] + " " + conv, raw[src], [g < 2 for g in labs], n_spells(pred), n_spells(labs))
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "combo", "a2_bfree_all_wm2.txt"))
    f = make_submission.merged_features()
    labels = model_v10.predict(f)
    phs, alive, preds = [], [], []
    for b, fa in zip(bench.blocks("barath"), exp_domain.barath()):
        p = bench.predict_team(mdl, fa)
        y = labels[b["a"]:b["a"] + len(p)]
        phs.append(p)
        alive.append(y < 2)
        preds.append(y)
    stats("BAR-ATH", phs, alive, n_spells(preds))


if __name__ == "__main__":
    main()
