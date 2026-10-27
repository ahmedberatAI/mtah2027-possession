"""Round 9: the deletion edit-utility rescorer (src/exp_edit2.py) on BAR-ATH. Trained on the candidates of all three
external sets (DFL-trk, Metrica-ctrl, SkillCorner-phase; learned-D decodes), applied to v7's (or a variant's) decoded
BAR-ATH labels: candidates = decoded spells < 15 s with an H/A neighbour; delete those with predicted gain > 0,
skipping neighbours of a deleted spell. Dead-ball probability features come from the learned all-DFL dead model on
BAR-ATH (bench.dead_probs) unless EDIT_NO_D=1. Label-free indicators plus the public window for information only;
optionally writes a candidate CSV. AGGREGATE output only.

  python -I src/edit_barath.py [--clean-ball] [--wh] [--out submissions/candidate_v7e.csv]
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import bench  # noqa: E402
import evaluate as ev  # noqa: E402
import exp_edit2 as ed  # noqa: E402
import exp_v4  # noqa: E402
import model_v7  # noqa: E402


def indicators(name, lab, ref=None):
    sc, dg = ev.score(lab), ev.diagnostics(lab)
    g = atd.graded_mask(len(lab))
    agree = "" if ref is None else " | frames changed vs base %.4f" % np.mean(lab[g] != ref[g])
    print("%-22s self-lag@25 %.3f | full n_poss %d (%.2f/alive min) <2s %d | public (info) spellF1 %.4f macroF1 %.4f "
          "team %.4f n_pred %d/17%s" % (name, exp_v4.self_lag25(lab), dg["n_poss"],
                                        dg["n_poss"] / (np.sum(lab[g] < 2) / 1500), dg["poss_lt2s"], sc["spellF1"],
                                        sc["macroF1"], sc["team_acc"], sc["n_pred"], agree), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-ball", action="store_true")
    ap.add_argument("--wh", action="store_true", help="v6 whistle bonus (model_v7 wh_bonus 30)")
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    data = {src: ed.build(src, conv, dec) for src, conv, dec in ed.SETS}
    mdl = ed.fit([h for hs in data.values() for h in hs])
    from make_submission import merged_features
    f = merged_features()
    params = {"clean_ball": True} if args.clean_ball else {}
    if args.wh:
        params["wh_bonus"] = 30.0
    y = model_v7.predict(f, params)
    probs = bench._load(model_v7.ensure_probs(clean_ball=args.clean_ball))
    pdb = bench.dead_probs("barath@v3", "base")
    seen = np.isfinite(f["bxy"][:, 0])
    out = y.copy()
    n_cand = 0
    for (_, a, b), pd_ in zip(atd.HALVES, pdb):
        p = probs[a][1]
        yh = y[a:b + 1].copy()
        h = dict(y0=yh)
        X, ids = [], []
        segs = ed.segments(yh)
        last_dead = -10 ** 9
        z = ed.e2.logit(p)
        for i, (s, e, k) in enumerate(segs):
            if k == 2:
                last_dead = e
                continue
            if k > 2 or e - s >= ed.MAX_DUR or ed.delete(yh, i, segs) is None:
                continue
            zk = z[s:e] if k == 0 else -z[s:e]
            L = e - s
            q = max(1, L // 5)
            prev = segs[i - 1] if i > 0 else (s, s, 3)
            nxt = segs[i + 1] if i + 1 < len(segs) else (e, e, 3)

            def side(sg, own):
                c, d, kk = sg
                if kk >= 2:
                    return [float(kk == 2), 0.0, 0.0, np.nan]
                zz = z[c:d] if kk == 0 else -z[c:d]
                return [0.0, float(kk == own), float(d - c), float(np.mean(zz)) if d > c else np.nan]

            row = [L, np.log1p(L), np.mean(zk), np.min(zk), np.max(zk), np.std(zk), np.sum(zk), np.mean(zk > 0),
                   np.mean(zk[:q]), np.mean(zk[-q:]), np.mean(pd_[s:e]), np.max(pd_[s:e]), pd_[max(0, s - 1)],
                   pd_[min(len(pd_) - 1, e)], np.mean(seen[a + s:a + e]), float(min(s - last_dead, 5000))]
            X.append(row + side(prev, k) + side(nxt, k))
            ids.append(i)
        if not X:
            continue
        X = np.array(X, np.float32)
        h["ids"] = ids
        pred = mdl.predict(ed.cols(X))
        n_cand += len(pred)
        yh2 = ed.apply(h, pred, 1.0)
        out[a:b + 1] = yh2
    print("candidates %d, frames changed %d, possession spells %d -> %d" % (
        n_cand, int(np.sum(out != y)), ev.diagnostics(y)["n_poss"], ev.diagnostics(out)["n_poss"]), flush=True)
    indicators("base" + (" cleanball" if args.clean_ball else "") + (" wh" if args.wh else ""), y)
    indicators("deletion rescorer", out, y)
    if args.out:
        ev.write(out, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
