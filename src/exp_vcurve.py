"""Round 21 diagnostic: does v14's verifier still learn from more matches?

v14's pooled verifier (G1, thresholds 0.65 / 0.35) learns, for every held-out match, from the candidates of the other
29 matches (DFL 7, Metrica 3, SkillCorner 20; both dead channels). Here each held-out match's model sees only a random
fraction of the other matches of every source (25 / 50 / 75 %, three draws; the same draw for the insertion and the
deletion model); 100 % is v14's own verifier (seed 0). A curve that still rises at 100 % says that more matches of this
kind (PFF's 64) could pay; a flat one says they would not. Three v10 seeds, ten contexts. AGGREGATE output only.

  python -I src/exp_vcurve.py run SEED FRAC REP     leave-one-match-out probabilities of one draw
  python -I src/exp_vcurve.py eval                  the curve on the ten contexts
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_sb  # noqa: E402
import exp_short as es  # noqa: E402
import exp_verify as ev  # noqa: E402
import exp_vpool  # noqa: E402

FRACS = (0.25, 0.5, 0.75)
REPS = (0, 1, 2)


def _out(seed, frac, rep):
    return os.path.join(ev.CACHE, "vcurve_%d_f%d_r%d.pkl" % (seed, round(100 * frac), rep))


def _pooled(seed):
    parts = {d: [r for src, *_ in ev.SOURCES for r in ev._rows4(src, seed, d)["rows"]] for d in ev.DEADS}
    return parts[ev.DEADS[0]] + parts[ev.DEADS[1]], len(parts[ev.DEADS[0]])


def kept(order, frac, held):
    """Training matches for held-out `held`: the first round(frac x n) of every source's drawn order, held excluded."""
    out = set()
    for groups in order.values():
        others = [g for g in groups if g != held]
        out |= set(others[:max(1, int(round(frac * len(others))))])
    return out


def run(seed, frac, rep):
    import lightgbm as lgb
    exp_sb.use_variant("v14")
    rows, n = _pooled(seed)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["src"], set()).add(r["group"])
    rng = np.random.default_rng(20261101 + rep)
    order = {s: [gs[i] for i in rng.permutation(len(gs))] for s, gs in ((s, sorted(g)) for s, g in sorted(by_src.items()))}
    params = dict(ev.PARAMS, seed=0, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    keys = ev.FEATSETS["G1"]
    p = np.full(len(rows), np.nan)
    sizes = []
    for typ in ("ins", "del"):
        idx = np.array([k for k, r in enumerate(rows) if r["typ"] == typ])
        X = np.array([[rows[k][f] for f in keys] for k in idx], float)
        y = np.array([rows[k]["y"] for k in idx])
        g = np.array([rows[k]["group"] for k in idx])
        for grp in np.unique(g):
            tr = np.isin(g, sorted(kept(order, frac, grp))) & (y >= 0)
            te = g == grp
            sizes.append(int(tr.sum()))
            p[idx[te]] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), ev.ROUNDS).predict(X[te])
    bench._dump({ev.DEADS[0]: p[:n], ev.DEADS[1]: p[n:]}, _out(seed, frac, rep))
    print("  seed %d fraction %.2f draw %d: mean training rows %d" % (seed, frac, rep, np.mean(sizes)), flush=True)


def evaluate():
    import exp_blend_weight as bw
    exp_sb.use_variant("v14")
    names = ["100%"] + ["%d%%" % round(100 * f) for f in FRACS]
    cells, aucs = {}, {}
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            rows, n = _pooled(seed)
            for k, dname in enumerate(ev.DEADS):
                part = rows[:n] if k == 0 else rows[n:]
                runs = {"100%": [bench._load(ev._path("probs4", seed, dname, "q0"))["G1"]]}
                for f in FRACS:
                    runs["%d%%" % round(100 * f)] = [bench._load(_out(seed, f, r))[dname] for r in REPS]
                for name, plist in runs.items():
                    for pp in plist:
                        preds = exp_vpool._apply(src, seed, dname, pp)
                        for c in convs:
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
                        for typ in ("ins", "del"):
                            sel = np.array([r["typ"] == typ and r["src"] == src and r["y"] >= 0 for r in part])
                            yy = np.array([r["y"] for r in part])
                            aucs.setdefault((name, typ, src), []).append(
                                ev.ea.auc(pp[sel & (yy == 1)], pp[sel & (yy == 0)]))
        bench._blocks.clear()
        bench._labels.clear()
    print("\nverifier AUC (mean over v10 seeds, dead channels and draws)")
    for name in names:
        print("  %-5s " % name + " | ".join("%s %s %.3f" % (typ, src.split("@")[0], np.mean(aucs[(name, typ, src)]))
                                          for typ in ("ins", "del") for src, *_ in ev.SOURCES))
    mean, regret = er.summarize(cells, names, ref="100%")
    for name in names[1:]:
        bw.regret_boot(mean, ["100%", name], "100%", name)
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "run":
        return run(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    if sys.argv[1] == "eval":
        return evaluate()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
