"""Round 17b: the round-13 blend weight grid (BiGRU probability share .25 / .5 / .75, pre-registered in
src/exp_shared_blend.py) with the short-possession verifier on top, judged on all ten contexts now that DFL has a
leave-one-match-out BiGRU (round 17, src/exp_bigru_lomo.py). Round 13 chose .75 on Metrica and SkillCorner alone.
Each weight gets its own candidates (src/exp_verify.py build3 / build4) and its own pooled leave-one-match-out
verifier (trainpool4); .75 reuses round 17's files. AGGREGATE output only.

The current model is judged as it meets BAR-ATH: v13's verifier (trained as in round 15, DFL rows without a BiGRU
margin) scores each held-out DFL match with the held-out BiGRU's margin ("v13 (DFL margin)"); round 15 left DFL's
margin missing. Decision: worst-case regret over the ten contexts within {current} + the six candidates, with a
paired match bootstrap of the regret difference (also shown pairwise).

  python -I src/exp_blend_weight.py build W                  candidates of the W blend on all three sources
  python -I src/exp_blend_weight.py trainpool W SEED VSEED   pooled leave-one-match-out verifier probabilities
  python -I src/exp_blend_weight.py build13                  v13's DFL candidates with the held-out BiGRU margin
  python -I src/exp_blend_weight.py trainpool13 SEED VSEED   v13's verifier scoring them
  python -I src/exp_blend_weight.py eval VSEED[,VSEED...]    current model and every weight, +- the verifier
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_verify as ev  # noqa: E402

WEIGHTS = (0.25, 0.5, 0.75)
CUR = "v13 (DFL margin)"      # v13 as it meets BAR-ATH: its verifier sees a BiGRU margin that never saw the match
_path = ev._path


def use_weight(w):
    """Point exp_short's blend and exp_verify's blend caches at weight w (.75 = round 17's untagged files)."""
    tag = "" if w == 0.75 else "w%02d" % round(100 * w)
    es.W_BIGRU = w

    def tagged(kind, *parts):
        return _path(kind + tag if kind in ("rows3", "rows4", "probs3", "probs4") else kind, *parts)
    ev._path = tagged


def build(w):
    use_weight(w)
    ev.build3()
    ev.build4()


def trainpool(w, seed, vseed):
    use_weight(w)
    ev.train_pooled4(seed, vseed)


def build_v13m():
    """v13's DFL candidates (rows2) again with the held-out BiGRU's margin: on BAR-ATH v13's verifier sees the shared
    BiGRU's margin, while round 15 scored DFL without it (the shared BiGRU trained on every DFL match)."""
    members = ev.lomo_bigru()
    lomo_mean = [np.mean([m[i] for m in members], axis=0) for i in range(len(members[0]))]
    gm = ev.gru_mean
    ev.gru_mean = lambda src: lomo_mean
    ev._path = lambda kind, *parts: _path("rows2m" if kind == "rows2" else kind, *parts)
    try:
        ev.build2("dfl@v3")
    finally:
        ev.gru_mean, ev._path = gm, _path
    for seed in es.SEEDS:
        for dname in ev.DEADS:
            a, b = (bench._load(_path(k, "dfl@v3", seed, dname))["rows"] for k in ("rows2", "rows2m"))
            assert [(r["i"], r["s"], r["t"], r["typ"]) for r in a] == [(r["i"], r["s"], r["t"], r["typ"]) for r in b]
    print("  v13's DFL candidates with the held-out BiGRU margin built", flush=True)


def trainpool_v13m(seed, vseed):
    """v13's pooled verifier trained exactly as in round 15 (DFL rows without the margin), each held-out DFL match
    scored with its held-out margin; the other sources reproduce round 15's probabilities (checked)."""
    import lightgbm as lgb
    tr_rows, te_rows, cut = [], [], [0]
    for dname in ev.DEADS:
        for src, *_ in ev.SOURCES:
            r = bench._load(_path("rows2", src, seed, dname))["rows"]
            tr_rows += r
            te_rows += bench._load(_path("rows2m", src, seed, dname))["rows"] if src.startswith("dfl") else r
        cut.append(len(tr_rows))
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    keys = ev.FEATSETS["G1"]
    p = np.full(len(tr_rows), np.nan)
    for typ in ("ins", "del"):
        idx = np.array([n for n, r in enumerate(tr_rows) if r["typ"] == typ])
        X = np.array([[tr_rows[n][k] for k in keys] for n in idx], float)
        Xt = np.array([[te_rows[n][k] for k in keys] for n in idx], float)
        y = np.array([tr_rows[n]["y"] for n in idx])
        g = np.array([tr_rows[n]["group"] for n in idx])
        for grp in np.unique(g):
            fit = (g != grp) & (y >= 0)
            m = lgb.train(params, lgb.Dataset(X[fit], y[fit]), ev.ROUNDS)
            p[idx[g == grp]] = m.predict(Xt[g == grp])
    old = np.concatenate([bench._load(_path("probs2", seed, d, "q%d" % vseed))["G1"] for d in ev.DEADS])
    dfl = np.array([r["src"].startswith("dfl") for r in tr_rows])
    assert np.allclose(p[~dfl], old[~dfl]), "round-15 probabilities not reproduced"
    for k, dname in enumerate(ev.DEADS):
        bench._dump({"G1": p[cut[k]:cut[k + 1]]}, _path("probs2m", seed, dname, "q%d" % vseed))
    print("    seed %d q%d: v13 with the DFL margin, %.1f%% of DFL decisions' probabilities moved by > 0.05" % (
        seed, vseed, 100 * float(np.mean(np.abs(p[dfl] - old[dfl]) > 0.05))), flush=True)


def _verified(src, seed, dname, vseed, loader, probs_kind):
    n0 = 0
    for s2, *_ in ev.SOURCES:
        if s2 == src:
            break
        n0 += len(loader(s2)["rows"])
    d = loader(src)
    pp = bench._load(ev._path(probs_kind, seed, dname, "q%d" % vseed))["G1"]
    by_half = {}
    for n, r in enumerate(d["rows"]):
        by_half.setdefault(r["i"], []).append(n)
    return d["base"], [ev.apply(pr, [d["rows"][n] for n in by_half.get(i, [])],
                                pp[[n0 + n for n in by_half.get(i, [])]], True, True, 0.65, 0.35)
                       for i, pr in enumerate(d["base"])]


def evaluate(vseeds):
    """Every (v10 seed, verifier seed) pair is one run; the blends without the verifier repeat across verifier seeds."""
    cells = {}
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                for vseed in vseeds:
                    outs = {}
                    ev._path = _path
                    for name, probs_kind in (("v13", "probs2"), (CUR, "probs2m")):
                        _, outs[name] = _verified(src, seed, dname, vseed,
                                                  lambda s2: bench._load(_path("rows2", s2, seed, dname)), probs_kind)
                    for w in WEIGHTS:
                        use_weight(w)
                        outs["blend %.2f" % w], outs["blend %.2f + verifier" % w] = _verified(
                            src, seed, dname, vseed, lambda s2: ev._rows4(s2, seed, dname), "probs4")
                    for c in convs:
                        for name, preds in outs.items():
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    er.summarize(cells, ["v13", CUR], ref="v13")           # what the DFL margin changes in v13 itself
    cands = ["blend %.2f%s" % (w, x) for w in WEIGHTS for x in ("", " + verifier")]
    mean, regret = er.summarize(cells, [CUR] + cands, ref=CUR)
    for name in cands:
        regret_boot(mean, [CUR] + cands, CUR, name)
    for name in cands:
        regret_boot(mean, [CUR, name], CUR, name)
    return mean, regret


def regret_boot(mean, names, cur, cand, n=10000, seed=0):
    """Paired match bootstrap of worst-case regret(cur) - regret(cand) within the set `names`: matches are resampled
    per source, the same draw for every context of that source (DFL trk / evctrl, learned / true D)."""
    ctxs = sorted({k[:3] for k in mean})
    rng = np.random.default_rng(seed)
    srcs = sorted({c[0] for c in ctxs})
    ms = {s: sorted(mean[[c for c in ctxs if c[0] == s][0] + (cur,)]) for s in srcs}
    picks = {s: rng.integers(0, len(ms[s]), size=(n, len(ms[s]))) for s in srcs}
    reg = {nm: np.zeros(n) for nm in names}
    point = {nm: 0.0 for nm in names}
    for ctx in ctxs:
        mat = np.array([[mean[ctx + (nm,)][m] for m in ms[ctx[0]]] for nm in names])   # names x matches
        vals = mat[:, picks[ctx[0]]].mean(axis=2)                                         # names x draws
        full = mat.mean(axis=1)
        for k, nm in enumerate(names):
            reg[nm] = np.maximum(reg[nm], vals.max(axis=0) - vals[k])
            point[nm] = max(point[nm], full.max() - full[k])
    d = reg[cur] - reg[cand]
    lo, hi = np.percentile(d, [5, 95])
    print("regret %s - %s within %d names: %+.4f, 90%% [%+.4f, %+.4f], P(>= 0.005) %.2f, P(> 0) %.2f" % (
        cur, cand, len(names), point[cur] - point[cand], lo, hi, float(np.mean(d >= 0.005)), float(np.mean(d > 0))),
        flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage = sys.argv[1]
    if stage == "build":
        return build(float(sys.argv[2]))
    if stage == "trainpool":
        return trainpool(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    if stage == "build13":
        return build_v13m()
    if stage == "trainpool13":
        return trainpool_v13m(int(sys.argv[2]), int(sys.argv[3]))
    if stage == "eval":
        return evaluate(tuple(int(x) for x in sys.argv[2].split(",")))
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
