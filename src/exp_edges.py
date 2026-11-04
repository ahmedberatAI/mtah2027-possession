"""Round 49 (docs/RAPOR.md, pre-registered): the ball around stops and restarts. sim.V2 / V3 cut every dead-ball gap exactly at the restart, so in the open sets the ball reappears on the true restart frame at about half of the
restarts (DFL 0.55, Metrica 0.72, and Metrica's ball is gone within 2 frames of 0.66 of the stops). Raw PFF tracking: 0.03, and the ball is visible through the last 2 s before 0.65 of the restarts. Every dead-ball model of the
project was trained and chosen on those masks. V6 (sim.V6) replaces the edges of each dead stretch by a draw of three states with the shares of the target's own label-free statistics. Here: how much of v19's dead-ball score
was that artefact, which evidence is right when it is gone, and does the answer carry over to real tracking (PFF raw, no simulated mask)?

  python -I src/exp_edges.py inputs        blocks and 'base' features of dfl@v6, metrica@v6g, sc@v6; the edge table on the true boundaries
  python -I src/exp_edges.py models        multi-source dead models per fold: trained on V3 or on V6, tested on V6 (and PFF raw)
  python -I src/exp_edges.py eval K        all arms on provider K (dfl, metrica, sc), V6 harness, frozen team channels, 3 seeds; prints v19's own edge table for comparison with the target
  python -I src/exp_edges.py decide        pools the providers
  python -I src/exp_edges.py pff           the arms' dead evidence on PFF raw, the independent test
"""
import json
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

V3 = {"dfl": ("dfl@v3", "dfl@v3", "evctrl"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}
V6 = {"dfl": ("dfl@v6", "dfl@v3", "evctrl"), "metrica": ("metrica@v6g", "metrica@v3", "ctrl"), "sc": ("sc@v6", "sc@v3", "evctrl")}
OUT = os.path.join(bench.CACHE, "edges")
RUNS = os.path.join(bench.ROOT, "tools", "runs")


def edges(lab):
    d = (np.asarray(lab) == 2).astype(np.int8)
    return np.flatnonzero((d[1:] == 1) & (d[:-1] == 0)) + 1, np.flatnonzero((d[1:] == 0) & (d[:-1] == 1)) + 1


def _nearest(ev, x, w=125):
    if not len(ev):
        return None
    j = int(np.argmin(np.abs(ev - x)))
    o = int(ev[j] - x)
    return o if abs(o) <= w else None


def edge_table(name, seens, labs):
    """The ball against stops and restarts (the same table as the target's in docs/RAPOR.md)."""
    R, S = [], []
    bef, aft = [0, 0, 0], [0, 0, 0]
    ma = md = na = nd = 0
    for seen, lab in zip(seens, labs):
        n = min(len(seen), len(lab))
        seen, lab = seen[:n], np.asarray(lab[:n])
        ma += int(((lab < 2) & ~seen).sum())
        na += int((lab < 2).sum())
        md += int(((lab == 2) & ~seen).sum())
        nd += int((lab == 2).sum())
        on, off = np.flatnonzero(seen[1:] & ~seen[:-1]) + 1, np.flatnonzero((~seen[1:]) & seen[:-1]) + 1
        st, re = edges(lab)
        for r in re:
            if 50 <= r <= n - 50 and lab[r - 1] < 3 and lab[r] < 3:
                R.append(_nearest(on, r))
                s0 = seen[r - 50:r]
                bef[0 if s0.all() else 1 if not s0.any() else 2] += 1
        for s in st:
            if 50 <= s <= n - 50 and lab[s - 1] < 3 and lab[s] < 3:
                S.append(_nearest(off, s))
                s1 = seen[s:s + 50]
                aft[0 if s1.all() else 1 if not s1.any() else 2] += 1

    def show(o):
        m = max(len(o), 1)
        v = np.array([x for x in o if x is not None])
        return "%.3f / %.3f / %.3f / %.3f" % (np.sum(np.abs(v) <= 2) / m, np.sum((np.abs(v) > 2) & (np.abs(v) <= 12)) / m, np.sum((np.abs(v) > 12) & (np.abs(v) <= 50)) / m, 1 - len(v) / m)

    print("%-26s missing live %.3f dead %.3f" % (name, ma / max(na, 1), md / max(nd, 1)))
    print("    restart: onset within 0-2 / 3-12 / 13-50 frames / none in 5 s: %s | last 2 s all seen / all gone / mixed: %.3f / %.3f / %.3f" % (show(R), bef[0] / max(sum(bef), 1), bef[1] / max(sum(bef), 1), bef[2] / max(sum(bef), 1)))
    print("    stop:    loss  within 0-2 / 3-12 / 13-50 frames / none in 5 s: %s | next 2 s all seen / all gone / mixed: %.3f / %.3f / %.3f" % (show(S), aft[0] / max(sum(aft), 1), aft[1] / max(sum(aft), 1), aft[2] / max(sum(aft), 1)), flush=True)


def inputs():
    for k, (fsrc, lsrc, conv) in V6.items():
        bench.features("base", fsrc)
        blocks = bench.blocks(fsrc)
        edge_table(fsrc + " true", [exp_blend.rule_f(b)["seen"] for b in blocks], bench.labels(lsrc, conv))
        bench._blocks.clear()
        bench._labels.clear()


def _groups(blocks):
    return [str(b["group"]) for b in blocks]


def fit_parts(srcs, skip):
    parts = {}
    for k, (fs, ls, gs) in srcs.items():
        keep = [i for i, g in enumerate(gs) if g not in skip]
        parts[k] = ([fs[i] for i in keep], [ls[i] for i in keep])
    return parts


def train_table(table):
    return {k: (bench.features("base", v[0]), bench.labels(v[1], v[2]), _groups(bench.blocks(v[0]))) for k, v in table.items()}


def fold_models(tag, table, with_pff=False):
    """Held-out P(dead) of the multi-source model trained on `table`, per fold, on every test source in V6 and on PFF raw (all models averaged there). Cached by tag."""
    os.makedirs(OUT, exist_ok=True)
    done = os.path.join(OUT, "M_%s_done" % tag)
    if os.path.isfile(done):
        return
    train = train_table(table)
    extra = dm.pff_part() if with_pff else None
    tests = {k: (bench.features("base", V6[k][0]), _groups(bench.blocks(V6[k][0]))) for k in V6}
    pff_x = None
    res = {k: [None] * len(tests[k][0]) for k in V6}
    for k in V6:
        un = sorted(set(tests[k][1]))
        folds = [un[i::5] for i in range(5)] if k == "sc" else [[g] for g in un]
        for fold in folds:
            parts = fit_parts(train, fold)
            if extra is not None:
                parts["pff"] = extra
            model = dm.fit_balanced(parts)
            for i, (x, g) in enumerate(zip(tests[k][0], tests[k][1])):
                if g in fold:
                    res[k][i] = model.predict(x["XF"], num_threads=bench.THREADS).astype(np.float32)
            print("  M %s %s fold %s done" % (tag, k, fold[:1]), flush=True)
    for k in V6:
        bench._dump(res[k], os.path.join(OUT, "M_%s_%s.pkl" % (tag, V6[k][0].replace("@", "-"))))
    parts = fit_parts(train, ())
    if extra is not None:
        parts["pff"] = extra
    model = dm.fit_balanced(parts)
    model.save_model(os.path.join(OUT, "M_%s_all.txt" % tag))
    with open(done, "w") as fh:
        fh.write("ok")


def models():
    fold_models("v3", V3)
    fold_models("v6", V6)
    fold_models("v6p", V6, with_pff=True)


def rule_evidence(k, table, conf):
    """Track terms and view terms per half of V6's provider k, weights fitted on `table` (held out by match)."""
    ro.SOURCES = table
    ro._f.clear()
    t, b = conf["thresholds"], conf["bounds"]
    d = ro.data()
    fsrc, lsrc, conv = V6[k]
    blocks = bench.blocks(fsrc)
    labs = [np.asarray(g) for g in bench.labels(lsrc, conv)]
    grp = _groups(blocks)
    cues = sd.view_cues() if k == "sc" else [(None, None)] * len(blocks)
    un = sorted(set(grp))
    folds = [un[i::5] for i in range(5)] if k == "sc" else [[g] for g in un]
    track, view = [None] * len(blocks), [None] * len(blocks)
    fields = [exp_blend.rule_f(x) for x in blocks]
    conv_grp = {str(g): g for g in {x[2] for x in d[k]}}
    for fold in folds:
        skip = [conv_grp.get(g, g) for g in fold]
        X, y, w = ro.design(t, b, skip=skip)
        beta, _ = ro.logistic(X, y, w)
        gamma = ro.fit_view(t, b, beta, skip=skip if k == "sc" else ())
        for i, g in enumerate(grp):
            if g in fold:
                track[i] = ro.terms(fields[i], t, b) @ beta
                if cues[i][0] is None:
                    v2, l2 = dm.transplant(labs[i], np.random.default_rng([dm.SEED, 0, i]))
                else:
                    v2, l2 = cues[i][0], cues[i][0] & (cues[i][1] >= 4)
                view[i] = gamma[0] * (~v2) + gamma[1] * l2
    print("  rule weights (last fold, %s): %s; view %s" % (table["dfl"][0], np.round(beta, 3).tolist(), np.round(gamma, 3).tolist()), flush=True)
    ro.SOURCES = V3
    ro._f.clear()
    bench._blocks.clear()
    bench._labels.clear()
    return track, view


def evidence_arms(k, conf):
    fsrc = V6[k][0]
    tr3, vi3 = rule_evidence(k, V3, conf)
    tr6, vi6 = rule_evidence(k, V6, conf)
    load = lambda tag: [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M_%s_%s.pkl" % (tag, fsrc.replace("@", "-"))))]
    m3, m6, m6p = load("v3"), load("v6"), load("v6p")
    arms = {"v19": [0.25 * t + 0.75 * m + v for t, m, v in zip(tr3, m3, vi3)],
            "rules3": [t + v for t, v in zip(tr3, vi3)],
            "M6 A25": [0.25 * t + 0.75 * m + v for t, m, v in zip(tr6, m6, vi6)],
            "M6 A50": [0.5 * t + 0.5 * m + v for t, m, v in zip(tr6, m6, vi6)],
            "M6 A75": [0.75 * t + 0.25 * m + v for t, m, v in zip(tr6, m6, vi6)],
            "M6 only": [m + v for m, v in zip(m6, vi6)],
            "rules6": [t + v for t, v in zip(tr6, vi6)],
            "M6P A25": [0.25 * t + 0.75 * m + v for t, m, v in zip(tr6, m6p, vi6)],
            "M6P A50": [0.5 * t + 0.5 * m + v for t, m, v in zip(tr6, m6p, vi6)]}
    return arms


def evaluate(k):
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    arms = evidence_arms(k, conf)
    fsrc, lsrc, _ = V6[k]
    src = V3[k][1]
    convs = dm.CONVS[src]
    blocks = bench.blocks(fsrc)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    truth = labs[convs[0]]
    seens = [exp_blend.rule_f(b)["seen"] for b in blocks]
    smos = dm.team(src)
    cells = {}
    names = list(arms)
    for s in es.SEEDS:
        for name, e in arms.items():
            preds = sd.decode_scores(e, smos[s])
            for c in convs:
                cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            if s == es.SEEDS[0]:
                print("  %-9s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
                if name in ("v19", "M6 A25"):
                    edge_table("%s pred %s" % (fsrc, name), seens, preds)
        print("  seed %d done" % s, flush=True)
    er.summarize(cells, names, ref="v19")
    with open(os.path.join(RUNS, "r49_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)


def decide():
    import exp_blend_weight as bw
    cells = {}
    for k in V6:
        with open(os.path.join(RUNS, "r49_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    names = sorted({c[3] for c in cells}, key=lambda n: (n != "v19", n))
    mean, _ = er.summarize(cells, names, ref="v19")
    for cand in names[1:]:
        bw.decide(mean, "v19", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"inputs": inputs, "models": models, "eval": evaluate, "decide": decide}[sys.argv[1]](*sys.argv[2:])
