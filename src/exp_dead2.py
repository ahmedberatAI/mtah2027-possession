"""Round 46 (docs/RAPOR.md, pre-registered): more out of the dead-ball channel, on top of v19's evidence (A25 = 0.25 open rules + 0.75 multi-source model + view).

  TD     the sequence model of src/exp_tdead.py (mean of three seeds, held out by match)
  M2     the multi-source LightGBM with team-side features too: XF + (XH + XA) / 2 + |XH - XA| of v7's feature set (set-piece formation, team shape, control), equal provider weight, held out by match
  V2     a richer view channel fitted on SkillCorner alone, whose flag is real: share of invalid view over +-2 s, +-5 s, +-15 s, length of the current invalid run, on top of the two view terms

  python -I src/exp_dead2.py rich          M2's held-out probabilities on the three providers
  python -I src/exp_dead2.py eval K        all candidates on one provider (dfl, metrica, sc); decide pools them
"""
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_blend  # noqa: E402
import exp_deadmix as dm  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_rules_open as ro  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402
import exp_tdead as td  # noqa: E402
import model_v7  # noqa: E402

SRC = ro.SOURCES
OUT = os.path.join(bench.CACHE, "dead2")
RUNS = os.path.join(bench.ROOT, "tools", "runs")


def rich_x(f):
    xs, xa = f["XH"], f["XA"]
    return np.hstack([f["XF"], (xs + xa) / 2, np.abs(xs - xa)]).astype(np.float32)


def rich():
    """M2, held out by match (SkillCorner by five folds); also the all-data model."""
    import lightgbm as lgb
    os.makedirs(OUT, exist_ok=True)
    srcs = {k: (bench.features(model_v7.VERSION, v[0]), [np.asarray(x) for x in bench.labels(v[1], {"dfl": "native"}.get(k, v[2]))], [b["group"] for b in bench.blocks(v[0])]) for k, v in SRC.items()}
    rows = {k: [(rich_x(f)[::bench.STRIDE], l[::bench.STRIDE], g) for f, l, g in zip(*srcs[k])] for k in srcs}

    def fit(skip):
        Xs, ys, ws = [], [], []
        for k in rows:
            x = np.concatenate([r[0] for r in rows[k] if r[2] not in skip])
            y = np.concatenate([r[1] for r in rows[k] if r[2] not in skip])
            m = y < 3
            Xs.append(x[m])
            ys.append((y[m] == 2).astype(int))
            ws.append(np.full(int(m.sum()), 1.0 / int(m.sum())))
        w = np.concatenate(ws)
        return lgb.train(bench.LGB_DEAD, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys), weight=w * len(w) / w.sum()), bench.ROUNDS)

    for k, (fs, ls, gs) in srcs.items():
        path = os.path.join(OUT, "M2_%s.pkl" % SRC[k][0].replace("@", "-"))
        if os.path.isfile(path):
            continue
        un = sorted(set(gs), key=str)
        folds = [un[i::5] for i in range(5)] if k == "sc" else [[g] for g in un]
        res = [None] * len(fs)
        for fold in folds:
            mdl = fit(fold)
            for i, (f, g) in enumerate(zip(fs, gs)):
                if g in fold:
                    res[i] = mdl.predict(rich_x(f), num_threads=bench.THREADS).astype(np.float32)
            print("  M2 %s fold %s done" % (k, fold), flush=True)
        bench._dump(res, path)
    mdl = fit([])
    mdl.save_model(os.path.join(OUT, "M2_all.txt"))
    print("  M2 all done", flush=True)


def view2_terms(valid, ndet):
    """Richer view features of one half: the two terms, share of invalid frames over +-2 s, +-5 s, +-15 s, log length of the current invalid run."""
    n = len(valid)
    inv = (~valid).astype(np.float64)
    c = np.r_[0.0, np.cumsum(inv)]

    def share(h):
        a, b = np.clip(np.arange(n) - h, 0, n), np.clip(np.arange(n) + h + 1, 0, n)
        return (c[b] - c[a]) / np.maximum(b - a, 1)

    st = np.flatnonzero(np.r_[inv[0] > 0, (inv[1:] > 0) & (inv[:-1] == 0)])
    en = np.flatnonzero(np.r_[(inv[:-1] > 0) & (inv[1:] == 0), inv[-1] > 0]) + 1
    run = np.zeros(n)
    for a, b in zip(st, en):
        run[a:b] = np.log1p((b - a) / 25.0)
    return np.stack([inv, (valid & (ndet >= 4)).astype(np.float64), share(50), share(125), share(375), run], 1)


def fit_view2(track, cues, labs, groups, skip):
    """The view weights on SkillCorner's matches outside skip, with the tracking evidence as a fixed offset."""
    X, y, off = [], [], []
    for t, (v, n), g, grp in zip(track, cues, labs, groups):
        if grp in skip:
            continue
        m = g[::ro.STRIDE] < 3
        X.append(view2_terms(v, n)[::ro.STRIDE][m])
        y.append((g[::ro.STRIDE][m] < 2).astype(float))
        off.append(t[::ro.STRIDE][m])
    beta, _ = ro.logistic(np.concatenate(X), np.concatenate(y), np.ones(sum(len(a) for a in y)), offset=np.concatenate(off))
    return beta


def transplant2(g, rng):
    """(valid, ndet) transplanted from SkillCorner; ndet 4 or 0 stands for live or not (only the threshold is read)."""
    v, l = dm.transplant(g, rng)
    return v, np.where(l, 4, 0)


def evaluate(k):
    with open(ro.OUT) as fh:
        conf = __import__("json").load(fh)
    fsrc, src, _ = SRC[k]
    convs = dm.CONVS[src]
    track, view = ro.evidence(k, conf)
    blocks = bench.blocks(fsrc)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    truth = labs[convs[0]]
    a_m = [exp_blend.logit_alive(x) for x in dm.balanced_probs("DMSb", fsrc)]
    a_t = [exp_blend.logit_alive(x) for x in td.held(fsrc)]
    a_r = [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M2_%s.pkl" % fsrc.replace("@", "-")))]
    groups = [b["group"] for b in blocks]
    # the richer view channel: fitted on SkillCorner, held out by fold there
    sc_blocks = bench.blocks("sc@v3")
    sc_cues = sd.view_cues()
    sc_labs = [np.asarray(g) for g in bench.labels("sc@v3", "evctrl")]
    sc_groups = [b["group"] for b in sc_blocks]
    sc_track, _ = ro.evidence("sc", conf)
    sc_a = [exp_blend.logit_alive(x) for x in dm.balanced_probs("DMSb", "sc@v3")]
    sc_off = [0.25 * t + 0.75 * m for t, m in zip(sc_track, sc_a)]
    if k == "sc":
        un = sorted(set(sc_groups), key=str)
        folds = [un[i::5] for i in range(5)]
        v2 = [None] * len(blocks)
        for fold in folds:
            b2 = fit_view2(sc_off, sc_cues, sc_labs, sc_groups, fold)
            for i, g in enumerate(sc_groups):
                if g in fold:
                    v2[i] = view2_terms(*sc_cues[i]) @ b2
    else:
        b2 = fit_view2(sc_off, sc_cues, sc_labs, sc_groups, ())
        v2 = [view2_terms(*transplant2(g, np.random.default_rng([dm.SEED, 0, i]))) @ b2 for i, g in enumerate(truth)]
    print("  view2 weights (last fit): %s" % np.round(b2, 3), flush=True)
    ev = {"v19": [0.25 * t + 0.75 * m + v for t, m, v in zip(track, a_m, view)],
          "TD": [0.25 * t + 0.75 * x + v for t, x, v in zip(track, a_t, view)],
          "M+TD": [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(track, a_m, a_t, view)],
          "M2": [0.25 * t + 0.75 * x + v for t, x, v in zip(track, a_r, view)],
          "M2+TD": [0.25 * t + 0.375 * r + 0.375 * x + v for t, r, x, v in zip(track, a_r, a_t, view)],
          "M+M2+TD": [0.25 * t + 0.25 * m + 0.25 * r + 0.25 * x + v for t, m, r, x, v in zip(track, a_m, a_r, a_t, view)],
          "v19 V2": [0.25 * t + 0.75 * m + v for t, m, v in zip(track, a_m, v2)],
          "M2+TD V2": [0.25 * t + 0.375 * r + 0.375 * x + v for t, r, x, v in zip(track, a_r, a_t, v2)]}
    smos = dm.team(src)
    cells = {}
    names = list(ev)
    for s in es.SEEDS:
        for name, e in ev.items():
            preds = sd.decode_scores(e, smos[s])
            for c in convs:
                cells.setdefault((src, c, dm.TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            if s == es.SEEDS[0]:
                print("  %-10s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
        print("  seed %d done" % s, flush=True)
    er.summarize(cells, names, ref="v19")
    with open(os.path.join(RUNS, "r46_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)


def decide():
    import exp_blend_weight as bw
    cells = {}
    for k in SRC:
        with open(os.path.join(RUNS, "r46_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    names = sorted({c[3] for c in cells}, key=lambda n: (n != "v19", n))
    mean, _ = er.summarize(cells, names, ref="v19")
    for cand in names[1:]:
        bw.decide(mean, "v19", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"rich": rich, "eval": evaluate, "decide": decide}[sys.argv[1]](*sys.argv[2:])
