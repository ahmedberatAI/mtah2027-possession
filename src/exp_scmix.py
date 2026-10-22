"""Idea A3b (docs/FIKIRLER.md): train the team model on degraded DFL + degraded SkillCorner instead of DFL
alone. SkillCorner is evaluated out of fold (4 folds over its 20 matches, all DFL always in training),
Metrica-ctrl and BAR-ATH with the model trained on DFL + all of SkillCorner. Same features as v7 (sim V3),
dead-ball evidence = the all-DFL base V3 model for every variant. AGGREGATE output only.

  python -I src/exp_scmix.py [sc_weight ...]      # weight of a SkillCorner frame relative to a DFL frame
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402

V = model_v7.VERSION
DEC = dict(pen_dead=30.0, min_poss=8, min_dead=50)
B2_PENS = (4, 8, 12, 20)
OUT = os.path.join(bench.CACHE, "scmix")


def fit(feats, labs, weights):
    Xs, ys, ws = [], [], []
    for f, l, w in zip(feats, labs, weights):
        X, y, _ = bench.team_rows([f], [l])
        Xs.append(X), ys.append(y), ws.append(np.full(len(y), w, np.float32))
    ds = lgb.Dataset(np.concatenate(Xs), np.concatenate(ys), weight=np.concatenate(ws))
    return lgb.train(dict(bench.LGB_TEAM, num_threads=bench.THREADS), ds, bench.ROUNDS)


def decode_all(pdl, phl, pt, b2):
    sp = (decode2.duration_spec(pt, DEC["pen_dead"], DEC["min_poss"], DEC["min_dead"], exp_v7.ramp(175, 48.0))
          if b2 else decode2.std_spec(pt, DEC["pen_dead"], DEC["min_poss"], DEC["min_dead"]))
    return [sp.decode(td.emissions(pd_, ph, 1.0, 1.0)).astype(np.int8) for pd_, ph in zip(pdl, phl)]


def scores(src, conv, pdl, phl):
    bl, gts = bench.blocks(src), bench.labels(src, conv)
    out = {}
    for b2 in (False, True):
        for pt in (B2_PENS if b2 else (20, 32, 48, 64)):
            r = bench.spell_by_group(bl, gts, decode_all(pdl, phl, pt, b2))
            out[("b2" if b2 else "std", pt)] = np.array([r[k][0] for k in sorted(r)])
    return out


def report(name, res, ref):
    for kind in ("std", "b2"):
        keys = [k for k in res if k[0] == kind]
        kb = max(keys, key=lambda k: res[k].mean())
        kr = max((k for k in ref if k[0] == kind), key=lambda k: ref[k].mean())
        d = res[kb] - ref[kr]
        print("  %-34s %-3s best @%-2d %.4f | vs DFL-only %s best %.4f: %+.4f (wins %d/%d)" % (
            name, kind, kb[1], res[kb].mean(), kind, ref[kr].mean(), d.mean(), int((d > 0).sum()), len(d)), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    weights = [float(x) for x in sys.argv[1:]] or [1.0]
    os.makedirs(OUT, exist_ok=True)
    fd = bench.features(V, "dfl@v3")
    ld = bench.labels("dfl@v3", "native")
    fs = bench.features(V, "sc@v3")
    ls = bench.labels("sc@v3", "phase")
    bls = bench.blocks("sc@v3")
    matches = sorted({b["group"] for b in bls})
    folds = [matches[k::4] for k in range(4)]
    pd_sc = bench.dead_ext("base", "v3", "sc@v3")
    pd_m = bench.dead_probs("metrica@v3", "base")
    fm = bench.features(V, "metrica@v3")
    dfl_only = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", "%s_native_v3.txt" % bench._safe(V)))
    ref_sc = scores("sc@v3", "phase", pd_sc, [bench.predict_team(dfl_only, f) for f in fs])
    ref_m = scores("metrica@v3", "ctrl", pd_m, [bench.predict_team(dfl_only, f) for f in fm])
    for w in weights:
        print("\nSkillCorner frame weight %.2g (DFL frame = 1)" % w, flush=True)
        ph_sc = [None] * len(bls)
        for k, test in enumerate(folds):
            tr = [i for i, b in enumerate(bls) if b["group"] not in test]
            mdl = fit(fd + [fs[i] for i in tr], ld + [ls[i] for i in tr], [1.0] * len(fd) + [w] * len(tr))
            for i, b in enumerate(bls):
                if b["group"] in test:
                    ph_sc[i] = bench.predict_team(mdl, fs[i])
        report("SkillCorner phase (out of fold)", scores("sc@v3", "phase", pd_sc, ph_sc), ref_sc)
        full = fit(fd + fs, ld + ls, [1.0] * len(fd) + [w] * len(fs))
        full.save_model(os.path.join(OUT, "dfl_sc_w%g.txt" % w))
        report("Metrica ctrl", scores("metrica@v3", "ctrl", pd_m, [bench.predict_team(full, f) for f in fm]), ref_m)
        probs = {b["a"]: (None, bench.predict_team(full, f))
                 for b, f in zip(bench.blocks("barath"), bench.features(V, "barath"))}
        path = os.path.join(OUT, "barath_probs_w%g.pkl" % w)
        bench._dump(probs, path)
        import evaluate as ev
        import atd
        from make_submission import merged_features
        f = merged_features()
        for pt in (8, 12):
            lab = model_v7.predict(f, {"pen_team": float(pt)}, probs_path=path)
            sc, dg = ev.score(lab), ev.diagnostics(lab)
            g = lab[atd.graded_mask(len(lab))]
            print("  BAR-ATH v7 decoder pen %2d | public spellF1 %.4f macroF1 %.4f team_acc %.4f n_pred %d/%d | full "
                  "n_poss %d (%.2f/alive min)" % (pt, sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["n_pred"],
                                                 sc["n_gt"], dg["n_poss"], dg["n_poss"] / (np.sum(g < 2) / 1500)),
                  flush=True)


if __name__ == "__main__":
    main()
