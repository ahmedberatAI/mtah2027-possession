"""Round 6, idea N26 (docs/RAPOR.md): train v7's team model on DFL degraded with sim.V4 (V3 on pre-smoothed
tracks, matching the frame-to-frame smoothness of BAR-ATH's ATD, SkillCorner and Metrica) instead of V3.
External sets keep their V3 inputs (already as smooth as BAR-ATH); dead-ball evidence = base V3 DFL model or true D.
Also BAR-ATH: v7 decoder with the V4 team probabilities -> full-match spell rate, self-lag at 1 s (label-free
"stickiness" indicator, cf. round 6), public window for information only. AGGREGATE output only.

  python -I src/exp_v4.py
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
import decode2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402

V = model_v7.VERSION
TAG3 = "%s_native_v3" % bench._safe(V)
OUT = os.path.join(bench.CACHE, "v4")


def self_lag25(lab):
    pm = td.pm
    hs = [lab[x:y + 1] for _, x, y in atd.HALVES]
    g = np.concatenate([np.concatenate([h, np.full(10, 3, np.int8)]) for h in hs]).astype(np.int8)
    s = np.concatenate([np.concatenate([np.concatenate([h[:25], h[:-25]]), np.full(10, 3, np.int8)]) for h in hs])
    return pm.evaluate(g, s.astype(np.int8))["spell_f1_mean"]


def per_group(src, conv, dec, pds, phl):
    sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
    r = bench.spell_by_group(bench.blocks(src), bench.labels(src, conv),
                             [sp.decode(td.emissions(a, b, 1.0, 1.0)).astype(np.int8) for a, b in zip(pds, phl)])
    return np.array([r[k][0] for k in sorted(r)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    fd = bench.features(V, "dfl@v4")
    ld = bench.labels("dfl@v4", "native")
    bl = bench.blocks("dfl@v4")
    oof_path = os.path.join(OUT, "oof.pkl")
    if os.path.isfile(oof_path):
        oof = bench._load(oof_path)
    else:
        oof = [None] * len(bl)
        for m in sorted({b["group"] for b in bl}):
            tr = [i for i, b in enumerate(bl) if b["group"] != m]
            mdl = bench.fit_team([fd[i] for i in tr], [ld[i] for i in tr])
            for i, b in enumerate(bl):
                if b["group"] == m:
                    oof[i] = bench.predict_team(mdl, fd[i])
            print("  v4 fold", m, flush=True)
        bench._dump(oof, oof_path)
    mpath = os.path.join(OUT, "team_all_dfl_v4.txt")
    if os.path.isfile(mpath):
        m4 = lgb.Booster(model_file=mpath)
    else:
        m4 = bench.fit_team(fd, ld)
        m4.save_model(mpath)
    m3 = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", TAG3 + ".txt"))
    # DFL-trk in-sim: each model on its own degradation (different inputs; indicative only)
    gts = bench.labels("dfl@v4", "trk")
    true_d = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in gts]
    oof3 = bench._load(os.path.join(bench.CACHE, "oof", TAG3 + ".pkl"))
    v3 = per_group("dfl@v3", "trk", (32.0, 5, 25), true_d, oof3)
    v4 = per_group("dfl@v4", "trk", (32.0, 5, 25), true_d, oof)
    print("DFL-trk true D (own degradation each): v3-trained %.4f | v4-trained %.4f (%+.4f, %d/%d)" % (
        v3.mean(), v4.mean(), v4.mean() - v3.mean(), int((v4 > v3).sum()), len(v4)), flush=True)
    for src, conv in (("metrica@v3", "ctrl"), ("sc@v3", "phase")):
        feats = bench.features(V, src)
        p3 = [bench.predict_team(m3, f) for f in feats]
        p4 = [bench.predict_team(m4, f) for f in feats]
        g = bench.labels(src, conv)
        acc = lambda phl: np.mean([np.mean((ph[l < 2] > 0.5) == (l[l < 2] == 0)) for ph, l in zip(phl, g)])
        learned = bench.dead_ext("base", "v3", src) if src.startswith("sc") else bench.dead_probs(src, "base")
        for dname, pds in (("learned D", learned), ("true D", [np.where(x == 2, 1 - 1e-4, 1e-4) for x in g])):
            a, b = per_group(src, conv, (30.0, 8, 50), pds, p3), per_group(src, conv, (30.0, 8, 50), pds, p4)
            print("%-11s %-9s team acc v3 %.4f v4 %.4f | spellF1 v3-trained (v7) %.4f | v4-trained %.4f (%+.4f, %d/%d)" % (
                src, dname, acc(p3), acc(p4), a.mean(), b.mean(), b.mean() - a.mean(), int((b > a).sum()), len(b)),
                flush=True)
    import evaluate as ev
    from make_submission import merged_features
    f = merged_features()
    probs = {blk["a"]: (None, bench.predict_team(m4, x)) for blk, x in zip(bench.blocks("barath"), bench.features(V, "barath"))}
    path = os.path.join(OUT, "barath_probs_v4.pkl")
    bench._dump(probs, path)
    for name, pp in (("v7 (V3 team model)", None), ("V4 team model", path)):
        lab = model_v7.predict(f, probs_path=pp)
        sc, dg = ev.score(lab), ev.diagnostics(lab)
        gm = lab[atd.graded_mask(len(lab))]
        print("BAR-ATH %-20s self-lag@25 %.3f | full n_poss %d (%.2f/alive min) | public (info) spellF1 %.4f macroF1 "
              "%.4f team %.4f n_pred %d/17" % (name, self_lag25(lab), dg["n_poss"], dg["n_poss"] / (np.sum(gm < 2) / 1500),
                                               sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["n_pred"]), flush=True)


if __name__ == "__main__":
    main()
