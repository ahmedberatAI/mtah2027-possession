"""Round 7b on BAR-ATH: the player-level controller (src/ctrl_model.py, trained on all SkillCorner) applied to the
match, as (a) the bounded blend logit(p_v7) + alpha * log((q_H + c) / (q_A + c)) and (b) the controller-feature team
model of src/exp_ctrl2.py (all-DFL). With and without the fabricated-ball cleaning (model_v7 clean_ball). Writes
probability files in model_v7's format (cache/bench/ctrl/barath_probs_<variant>.pkl) and prints label-free
indicators plus the public window for information only. AGGREGATE output only.

  python -I src/ctrl_barath.py [--alpha 0.5] [--c 0.2]
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import ctrl_model as cm  # noqa: E402
import evaluate as ev  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_v4  # noqa: E402
import model_v7  # noqa: E402


def indicators(name, f, path, ref=None):
    lab = model_v7.predict(f, probs_path=path)
    sc, dg = ev.score(lab), ev.diagnostics(lab)
    g = atd.graded_mask(len(lab))
    agree = "" if ref is None else " | frames changed vs v7 %.4f" % np.mean(lab[g] != ref[g])
    print("%-24s self-lag@25 %.3f | full n_poss %d (%.2f/alive min) <2s %d | public (info) spellF1 %.4f macroF1 %.4f "
          "team %.4f n_pred %d/17%s" % (name, exp_v4.self_lag25(lab), dg["n_poss"], dg["n_poss"] / (np.sum(lab[g] < 2) / 1500),
                                        dg["poss_lt2s"], sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["n_pred"], agree),
          flush=True)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.2)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    from make_submission import merged_features
    f = merged_features()
    ctrl = lgb.Booster(model_file=os.path.join(e2.OUT, "ctrl_all_sc.txt"))
    team = lgb.Booster(model_file=os.path.join(e2.OUT, e2.TAG + ".txt"))
    for clean in (False, True):
        tag = "cleanball" if clean else "asis"
        blocks = bench.blocks("barath")
        if clean:
            import ballclean
            blocks = ballclean.clean_blocks(blocks, f)
            feats = [model_v7._features(b) for b in blocks]
        else:
            feats = bench.features(model_v7.VERSION, "barath")
        v7 = bench._load(model_v7.ensure_probs(clean_ball=clean))
        ref = indicators("v7 " + tag, f, model_v7.ensure_probs(clean_ball=clean))
        blend, featp = {}, {}
        for b, fb in zip(blocks, feats):
            X, fr, sl, tm = cm.rows(b)
            p = ctrl.predict(X)
            Q, M = e2.per_frame(p, dict(fr=fr, tm=tm), len(b["P"]))
            blend[b["a"]] = (None, e2.sig(e2.logit(v7[b["a"]][1]) + args.alpha * e2.bounded(Q, args.c)))
            ch, ca = e2.ctrl_feats(Q, M)
            featp[b["a"]] = (None, bench.predict_team(team, dict(XF=fb["XF"], XH=np.hstack([fb["XH"], ch]),
                                                                 XA=np.hstack([fb["XA"], ca]))))
            print("  block %d: controller confident (q > 0.5) on %.3f of frames" % (b["a"], np.mean(Q.max(1) > 0.5)),
                  flush=True)
        for name, probs in (("blend", blend), ("ctrl features", featp)):
            path = os.path.join(e2.OUT, "barath_probs_%s_%s.pkl" % (name.replace(" ", ""), tag))
            bench._dump(probs, path)
            indicators("%s %s" % (name, tag), f, path, ref)


if __name__ == "__main__":
    main()
