"""Round 9: the players-only dead-ball model (src/podead.py, all-DFL model of src/exp_podead.py) on BAR-ATH.

probs() -> per-frame P(dead) on the tactical timeline (NaN outside the graded halves), cached in
cache/v7/barath_podead.pkl; used by model_v7's po_lam / po_c option (bounded fusion into the dead-ball channel).
main() prints aggregate diagnostics: the scale of v7's own dead-ball log-odds on BAR-ATH, the players-only model's
frame agreement with v7's labels, and the label-free indicators plus the public window (information only) of v7 with
the fusion. AGGREGATE output only.

  python -I src/podead_barath.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import podead  # noqa: E402

ROOT = os.path.dirname(HERE)
PATH = os.path.join(ROOT, "cache", "v7", "barath_podead.pkl")
MODEL = os.path.join(bench.CACHE, "podead", "podead_all_dfl.txt")


def probs():
    if os.path.isfile(PATH):
        return bench._load(PATH)
    mdl = lgb.Booster(model_file=MODEL)
    out = np.full(atd.N_FRAMES["tactical"], np.nan)
    for b in bench.blocks("barath"):
        X, _ = podead.po_feats(b["P"], b["team"], b["gk"], b["pitch"])
        out[b["a"]:b["a"] + len(X)] = mdl.predict(X, num_threads=8)
    bench._dump(out, PATH)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    import evaluate as ev
    import exp_v4
    import model_v7
    from make_submission import merged_features
    f = merged_features()
    po = probs()
    p = dict(model_v7.DEFAULT)
    S = model_v7.scores(f, p, model_v7.ensure_probs())
    lb = S[:, 2] - np.logaddexp(S[:, 0], S[:, 1])
    gt = ev.public_gt()
    v7 = model_v7.predict(f)
    g = atd.graded_mask(len(v7))
    for name, m in (("public GT live", gt < 2), ("public GT dead", gt == 2), ("v7 live (full)", g & (v7 < 2)),
                    ("v7 dead (full)", g & (v7 == 2))):
        q = np.nanpercentile(lb[m & (np.abs(lb) < 1e5)], [10, 50, 90])
        qp = np.nanpercentile(po[m], [10, 50, 90])
        print("%-16s v7 dead log-odds p10/50/90 %6.2f %6.2f %6.2f | players-only P(dead) %.2f %.2f %.2f" % (
            name, q[0], q[1], q[2], qp[0], qp[1], qp[2]), flush=True)
    ok = g & np.isfinite(po)
    print("players-only (> 0.5) agrees with v7's dead/alive on %.3f of graded frames" % np.mean(
        (po[ok] > 0.5) == (v7[ok] == 2)), flush=True)
    for lam, c in ((0.0, 0.0), (0.5, 2.0), (0.5, 4.0), (1.0, 2.0)):
        lab = model_v7.predict(f, {"po_lam": lam, "po_c": c})
        sc, dg = ev.score(lab), ev.diagnostics(lab)
        print("po_lam %.1f po_c %.0f | dead segs %d (median %.1f s, share %.3f) | n_poss %d (%.2f/alive min) | "
              "self-lag %.3f | public (info) spellF1 %.4f macroF1 %.4f alive/dead F1 %.4f | frames changed %.4f" % (
                  lam, c, dg["n_dead_segs"], dg["dead_median_s"], dg["dead_share"], dg["n_poss"],
                  dg["n_poss"] / (np.sum(lab[g] < 2) / 1500), exp_v4.self_lag25(lab), sc["spellF1"], sc["macroF1"],
                  sc["alive_dead_f1"], np.mean(lab[g] != v7[g])), flush=True)


if __name__ == "__main__":
    main()
