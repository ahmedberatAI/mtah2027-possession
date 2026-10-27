"""Round 6, idea 8 (docs/RAPOR.md): BAR-ATH ball frames that the ATD filled in by prediction.

Over-smooth runs of the tactical ball (|2nd difference| < 2 mm for >= 8 frames while moving) are 3.1% of the seen
graded frames, and there the two cameras disagree by 1.66 m (median; 3.15 m for runs >= 25 frames) against
0.75 m elsewhere -- they are extrapolations, not detections. Variant: drop them (use the broadcast ball where it
is valid and live, else missing), recompute v7's features on BAR-ATH, same team model, v7 decoder. Only label-free
indicators (spell count, 1 s self-lag, team-model confidence) plus the public window for information.
AGGREGATE output only.

  python -I src/exp_fabball.py
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
import evaluate as ev  # noqa: E402
import exp_v4  # noqa: E402
import mlfeat  # noqa: E402
import model_v7  # noqa: E402

from ballclean import fabricated  # noqa: E402


def block_features(b, version):
    F, Th, Ta = mlfeat.compute(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
    for fn in bench._feat_fns(version):
        f2, th2, ta2 = fn(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        F.update(f2), Th.update(th2), Ta.update(ta2)
    fk, tk = sorted(F), sorted(Th)
    return {"XF": np.stack([F[k] for k in fk], 1).astype(np.float32),
            "XH": np.stack([Th[k] for k in tk], 1).astype(np.float32),
            "XA": np.stack([Ta[k] for k in tk], 1).astype(np.float32)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    from make_submission import merged_features
    f = merged_features()
    d = atd.load("tactical")
    tac = atd.to_metres(d["ball"][:, :2])
    flag = fabricated(tac)
    bro_ok = f["b_seen"] & f["b_live"]
    V = model_v7.VERSION
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", "%s_native_v3.txt" % bench._safe(V)))
    out = {}
    for name in ("as is", "fabricated dropped"):
        probs = {}
        for b in bench.blocks("barath"):
            a, e = b["a"], b["b"]
            bb = dict(b)
            if name != "as is":
                B = f["bxy"][a:e + 1].copy()
                fl = flag[a:e + 1]
                B[fl] = np.where(bro_ok[a:e + 1, None][fl], f["b_ball"][a:e + 1][fl], np.nan)
                bb["B"] = (B / np.array(atd.PITCH, np.float32)).astype(np.float32)
                feats = block_features(bb, V)
            else:
                feats = bench.features(V, "barath")[0 if a == atd.HALVES[0][1] else 1]
            probs[a] = (None, bench.predict_team(mdl, feats))
        path = os.path.join(bench.CACHE, "fabball_probs_%s.pkl" % name.replace(" ", "_"))
        bench._dump(probs, path)
        lab = model_v7.predict(f, probs_path=path)
        sc, dg = ev.score(lab), ev.diagnostics(lab)
        g = atd.graded_mask(len(lab))
        alive = g & (lab < 2)
        ph = np.full(len(lab), 0.5)
        for a, (_, p) in probs.items():
            ph[a:a + len(p)] = p
        conf = np.abs(np.log(np.clip(ph, 1e-4, 1 - 1e-4) / np.clip(1 - ph, 1e-4, 1 - 1e-4)))
        out[name] = lab
        print("%-19s flagged frames changed %d | self-lag@25 %.3f | full n_poss %d (%.2f/alive min) <2s %d | median "
              "|logit| alive %.2f, on flagged frames %.2f | public (info) spellF1 %.4f macroF1 %.4f team %.4f n_pred %d/17" % (
                  name, int((flag & g).sum()), exp_v4.self_lag25(lab), dg["n_poss"],
                  dg["n_poss"] / (np.sum(lab[g] < 2) / 1500), dg["poss_lt2s"], np.median(conf[alive]),
                  np.median(conf[alive & flag]), sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["n_pred"]), flush=True)
    a, b = out["as is"], out["fabricated dropped"]
    g = atd.graded_mask(len(a))
    print("frame agreement %.4f; on flagged frames %.4f" % (np.mean(a[g] == b[g]), np.mean(a[g & flag] == b[g & flag])))


if __name__ == "__main__":
    main()
