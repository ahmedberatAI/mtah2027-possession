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
import exp_deglue as eg  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_rules_open as ro  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402

V3 = {"dfl": ("dfl@v3", "dfl@v3", "evctrl"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}
VAR = os.environ.get("EDGES_VAR", "v6")      # v6b: the sensitivity test, shares like raw PFF's
V6 = {"dfl": ("dfl@%s" % VAR, "dfl@v3", "evctrl"), "metrica": ("metrica@%sg" % VAR, "metrica@v3", "ctrl"), "sc": ("sc@%s" % VAR, "sc@v3", "evctrl")}
OUT = os.path.join(bench.CACHE, "edges" if VAR == "v6" else "edges_" + VAR)
TD_FOLDER = "tdead6" if VAR == "v6" else "tdead6b"
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
    if VAR == "v6":
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
    m3, m6 = load("v3"), load("v6")
    m6p = load("v6p") if os.path.isfile(os.path.join(OUT, "M_v6p_done")) else None
    arms = {"v19": [0.25 * t + 0.75 * m + v for t, m, v in zip(tr3, m3, vi3)],
            "rules3": [t + v for t, v in zip(tr3, vi3)],
            "M6 A25": [0.25 * t + 0.75 * m + v for t, m, v in zip(tr6, m6, vi6)],
            "M6 A50": [0.5 * t + 0.5 * m + v for t, m, v in zip(tr6, m6, vi6)],
            "M6 A75": [0.75 * t + 0.25 * m + v for t, m, v in zip(tr6, m6, vi6)],
            "M6 only": [m + v for m, v in zip(m6, vi6)],
            "rules6": [t + v for t, v in zip(tr6, vi6)]}
    if m6p is not None:
        arms["M6P A25"] = [0.25 * t + 0.75 * m + v for t, m, v in zip(tr6, m6p, vi6)]
        arms["M6P A50"] = [0.5 * t + 0.5 * m + v for t, m, v in zip(tr6, m6p, vi6)]
    paths = [os.path.join(bench.CACHE, TD_FOLDER, "held_%d_%s.pkl" % (s, fsrc.replace("@", "-"))) for s in (20261104, 20261105, 20261106)]
    if all(os.path.isfile(x) for x in paths):       # the sequence model of src/exp_tdead.py on V6's masks, mean of three seeds
        runs = [bench._load(x) for x in paths]
        td = [exp_blend.logit_alive(np.mean(np.stack([r[i] for r in runs]), axis=0)) for i in range(len(runs[0]))]
        arms["TD6 A25"] = [0.25 * t + 0.75 * x + v for t, x, v in zip(tr6, td, vi6)]
        arms["M6+TD6 A25"] = [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, td, vi6)]
        arms["M6+TD6 A50"] = [0.5 * t + 0.25 * m + 0.25 * x + v for t, m, x, v in zip(tr6, m6, td, vi6)]
        if m6p is not None:
            arms["M6P+TD6 A25"] = [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6p, td, vi6)]
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
            preds = eg.decode_par(e, smos[s])
            for c in convs:
                cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            if s == es.SEEDS[0]:
                print("  %-9s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
                if name in ("v19", "M6 A25"):
                    edge_table("%s pred %s" % (fsrc, name), seens, preds)
        print("  seed %d done" % s, flush=True)
    er.summarize(cells, names, ref="v19")
    with open(os.path.join(RUNS, "r49_cells_%s%s.pkl" % ("" if VAR == "v6" else VAR + "_", k)), "wb") as fh:
        pickle.dump(cells, fh)
    if len(arms) > 2:
        import exp_blend_weight as bw
        mean = {}
        for c, runs in cells.items():
            mean[c] = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
        print("(one provider: per-context deltas above)", flush=True)


def decide():
    import exp_blend_weight as bw
    cells = {}
    for k in V6:
        with open(os.path.join(RUNS, "r49_cells_%s%s.pkl" % ("" if VAR == "v6" else VAR + "_", k)), "rb") as fh:
            cells.update(pickle.load(fh))
    names = sorted({c[3] for c in cells}, key=lambda n: (n != "v19", n))
    mean, _ = er.summarize(cells, names, ref="v19")
    for cand in names[1:]:
        bw.decide(mean, "v19", cand)


def weights_on(table, conf):
    """The rule and view weights fitted on all of `table`."""
    ro.SOURCES = table
    ro._f.clear()
    t, c = conf["thresholds"], conf["bounds"]
    X, y, w = ro.design(t, c)
    beta, _ = ro.logistic(X, y, w)
    gamma = ro.fit_view(t, c, beta)
    ro.SOURCES = V3
    ro._f.clear()
    bench._blocks.clear()
    bench._labels.clear()
    return beta, gamma


def pff():
    """PFF raw: real tracking with its real ball gaps, no simulated mask, and no model below saw these matches (the view flag is transplanted, the team model is the all-DFL LightGBM, as in round 42). Is what wins on V6 also better on real gaps?"""
    import gradient
    import lightgbm as lgb
    import model_v7
    import pathlib
    import exp_smooth
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b3, g3 = weights_on(V3, conf)
    b6, g6 = weights_on(V6, conf)
    print("rule weights V3 %s view %s" % (np.round(b3, 3).tolist(), np.round(g3, 3).tolist()))
    print("rule weights V6 %s view %s" % (np.round(b6, 3).tolist(), np.round(g6, 3).tolist()), flush=True)
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    m3 = bench.load_booster(os.path.join(OUT, "M_v3_all.txt"))
    m6 = bench.load_booster(os.path.join(OUT, "M_v6_all.txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    names = ("v19", "rules3", "M6 A25", "M6 A50", "M6 A75", "M6 only", "rules6", "TD3 A25", "M3+TD3 A25", "TD6 A25", "M6+TD6 A25", "M6+TD6 A50")
    import exp_tdead as etd
    import torch

    def td_nets(folder):
        nets = []
        for seed in (20261104, 20261105, 20261106):
            ck = torch.load(os.path.join(bench.CACHE, folder, "all_%d.pt" % seed), map_location="cpu", weights_only=False)
            net = etd.Net(ck["n_in"]).to(etd.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            nets.append((net, ck["mean"], ck["scale"]))
        return nets

    def td_alive(nets, x):
        return exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, x) for n_, m_, s_ in nets], axis=0))

    td3, td6 = td_nets("tdead"), td_nets("tdead6")
    cells = {}
    n = 0
    dead = {k: [0, 0, 0] for k in names}
    for match in gradient.matches():
        blocks, labs, ev, smo = [], [], {k: [] for k in names}, []
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                t0 = "h%d" % int(per)
                b = dict(P=d[t0 + "_players"], B=d[t0 + "_ball"], team=d["team"], gk=d["gk"], pitch=tuple(d["pitch"]), group=match)
                lab = np.asarray(d[t0 + "_labels"])
                x = f["XF"][:, [f["fk"].index(k) for k in base_names]]
                rf = exp_blend.rule_f(b)
                v, l = dm.transplant(lab, np.random.default_rng([dm.SEED, n]))
                n += 1
                T = ro.terms(rf, t, c)
                tr3, tr6 = T @ b3, T @ b6
                vi3, vi6 = g3[0] * (~v) + g3[1] * l, g6[0] * (~v) + g6[1] * l
                a3 = exp_blend.logit_alive(m3.predict(x, num_threads=bench.THREADS))
                a6 = exp_blend.logit_alive(m6.predict(x, num_threads=bench.THREADS))
                blocks.append(b)
                labs.append(lab)
                ev["v19"].append(0.25 * tr3 + 0.75 * a3 + vi3)
                ev["rules3"].append(tr3 + vi3)
                ev["M6 A25"].append(0.25 * tr6 + 0.75 * a6 + vi6)
                ev["M6 A50"].append(0.5 * tr6 + 0.5 * a6 + vi6)
                ev["M6 A75"].append(0.75 * tr6 + 0.25 * a6 + vi6)
                ev["M6 only"].append(a6 + vi6)
                ev["rules6"].append(tr6 + vi6)
                x3, x6 = td_alive(td3, x), td_alive(td6, x)
                ev["TD3 A25"].append(0.25 * tr3 + 0.75 * x3 + vi3)
                ev["M3+TD3 A25"].append(0.25 * tr3 + 0.375 * a3 + 0.375 * x3 + vi3)
                ev["TD6 A25"].append(0.25 * tr6 + 0.75 * x6 + vi6)
                ev["M6+TD6 A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * x6 + vi6)
                ev["M6+TD6 A50"].append(0.5 * tr6 + 0.25 * a6 + 0.25 * x6 + vi6)
                smo.append(bench.predict_team(team_model, f))
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        for name in names:
            preds = eg.decode_par(ev[name], smo)
            cells.setdefault(("pff", "ctrl", "edges D", name), []).append(er.by_match(blocks, labs, preds))
            for p_, g_ in zip(preds, labs):
                ok = g_ < 3
                dead[name][0] += int(((p_ == 2) & (g_ == 2) & ok).sum())
                dead[name][1] += int(((p_ == 2) & (g_ < 2) & ok).sum())
                dead[name][2] += int(((p_ < 2) & (g_ == 2) & ok).sum())
        print("  %s done" % match, flush=True)
    merged = {}
    for k, v in cells.items():
        d = {}
        for x in v:
            d.update(x)
        merged[k] = [d]
    er.summarize(merged, list(names), ref="v19")
    for name in names:
        tp, fp, fn = dead[name]
        print("  %-9s dead F1 %.3f, precision %.3f, recall %.3f" % (name, 2 * tp / (2 * tp + fp + fn), tp / max(tp + fp, 1), tp / max(tp + fn, 1)), flush=True)


def boundary_heads():
    """The sequence model's stop and restart heads as time-varying costs of entering and leaving D: the timing pool (+0.014 / +0.016 / +0.054). M6+TD6 A25's evidence, V6 harness."""
    from concurrent.futures import ThreadPoolExecutor
    import decode2
    import exp_v7
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    ramp = exp_v7.ramp(175, 48.0)
    macros = [dict(emis=0, out=0, L=len(ramp), min_exit=7, exit_cost=ramp), dict(emis=1, out=1, L=len(ramp), min_exit=7, exit_cost=ramp), dict(emis=2, out=2, L=50)]
    pen = [[0, 12.0, 30.0], [12.0, 0, 30.0], [30.0, 30.0, 0]]
    spec = decode2.Spec(macros, pen, [[-1, -1, 0], [-1, -1, 0], [1, 1, -1]])
    lams = (10.0, 20.0, 30.0)
    names = ["M6+TD6 A25"] + ["h%d" % l for l in lams]
    cells = {}
    pool = ThreadPoolExecutor(int(os.environ.get("DECODE_WORKERS", "5")))
    for k in V6:
        fsrc = V6[k][0]
        src = V3[k][1]
        ev = evidence_arms(k, conf)["M6+TD6 A25"]
        heads = [bench._load(os.path.join(bench.CACHE, "tdead6h", "heads_%d_%s.pkl" % (s, fsrc.replace("@", "-")))) for s in (20261104, 20261105, 20261106)]
        heat = [np.mean(np.stack([np.asarray(h[i], np.float32) for h in heads]), axis=0) for i in range(len(heads[0]))]
        convs = dm.CONVS[src]
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        smos = dm.team(src)
        for s in es.SEEDS:
            ems = [np.stack([a / 2 + np.log(np.clip(q, sd.EPS, 1 - sd.EPS)), a / 2 + np.log(np.clip(1 - q, sd.EPS, 1 - sd.EPS)), -a / 2], 1) for a, q in zip(ev, smos[s])]
            for name in names:
                lam = 0.0 if name == names[0] else float(name[1:])

                def one(i):
                    return spec.decode(ems[i], -lam * heat[i][:len(ems[i])].astype(np.float64)).astype(np.int8)
                preds = list(pool.map(one, range(len(ems))))
                for c in convs:
                    cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (k, s), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, names, ref=names[0])
    import exp_blend_weight as bw
    for cand in names[1:]:
        bw.decide(mean, names[0], cand)


def stress():
    """As round 41: what if a channel is much worse or better on the target than measured? kappa: the rules know the truth by +-kappa log-odds; sigma: slow AR(1) noise on each learned channel (independent). One seed; DFL, Metrica, half of SkillCorner; max regret across the cells."""
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    names = ("v19", "M6 A25", "M6 A50", "TD6 A25", "M6+TD6 A25", "M6+TD6 A50")
    grid = ((0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (0, 4), (1, 4), (2, 4))
    res = {}
    for k in V6:
        fsrc = V6[k][0]
        src = V3[k][1]
        conv = dm.CONVS[src][0]
        tr3, vi3 = rule_evidence(k, V3, conf)
        tr6, vi6 = rule_evidence(k, V6, conf)
        load = lambda tag: [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M_%s_%s.pkl" % (tag, fsrc.replace("@", "-"))))]
        m3, m6 = load("v3"), load("v6")
        runs = [bench._load(os.path.join(bench.CACHE, "tdead6", "held_%d_%s.pkl" % (s, fsrc.replace("@", "-")))) for s in (20261104, 20261105, 20261106)]
        td = [exp_blend.logit_alive(np.mean(np.stack([r[i] for r in runs]), axis=0)) for i in range(len(runs[0]))]
        blocks = bench.blocks(fsrc)
        truth = [np.asarray(g) for g in bench.labels(src, conv)]
        smo = dm.team(src)[es.SEEDS[0]]
        keep = list(range(20)) if k == "sc" else list(range(len(blocks)))
        for kappa, sigma in grid:
            def noisy(x, tag):
                return [x[i] + dm._ar1(len(x[i]), np.random.default_rng([dm.SEED, tag, i]), sigma) if sigma else x[i] for i in keep]
            n3, n6, nt = noisy(m3, 31), noisy(m6, 32), noisy(td, 33)
            bias = [kappa * np.where(truth[i] == 2, -1.0, 1.0) for i in keep]
            t3 = [tr3[i] + b for i, b in zip(keep, bias)]
            t6 = [tr6[i] + b for i, b in zip(keep, bias)]
            v3 = [vi3[i] for i in keep]
            v6 = [vi6[i] for i in keep]
            ev = {"v19": [0.25 * t + 0.75 * m + v for t, m, v in zip(t3, n3, v3)],
                  "M6 A25": [0.25 * t + 0.75 * m + v for t, m, v in zip(t6, n6, v6)],
                  "M6 A50": [0.5 * t + 0.5 * m + v for t, m, v in zip(t6, n6, v6)],
                  "TD6 A25": [0.25 * t + 0.75 * x + v for t, x, v in zip(t6, nt, v6)],
                  "M6+TD6 A25": [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(t6, n6, nt, v6)],
                  "M6+TD6 A50": [0.5 * t + 0.25 * m + 0.25 * x + v for t, m, x, v in zip(t6, n6, nt, v6)]}
            b_, g_ = [blocks[i] for i in keep], [truth[i] for i in keep]
            sm = [smo[i] for i in keep]
            sc = {n: float(np.mean(list(er.by_match(b_, g_, eg.decode_par(ev[n], sm)).values()))) for n in names}
            res.setdefault((kappa, sigma), {})[k] = sc
            print("  %s kappa %d sigma %d | %s" % (k, kappa, sigma, " ".join("%s %.4f" % (n, sc[n]) for n in names)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    reg = {n: 0.0 for n in names}
    reg2 = {n: 0.0 for n in names}
    for cell, per in res.items():
        for k, sc in per.items():
            best = max(sc.values())
            for n in names:
                reg[n] = max(reg[n], best - sc[n])
                if cell[1] <= 2:
                    reg2[n] = max(reg2[n], best - sc[n])
    print("max regret, all cells: %s" % " | ".join("%s %.4f" % (n, reg[n]) for n in names))
    print("max regret, sigma <= 2: %s" % " | ".join("%s %.4f" % (n, reg2[n]) for n in names), flush=True)


def _held(folder, fsrc):
    runs = [bench._load(os.path.join(bench.CACHE, folder, "held_%d_%s.pkl" % (s, fsrc.replace("@", "-")))) for s in (20261104, 20261105, 20261106)]
    return [exp_blend.logit_alive(np.mean(np.stack([r[i] for r in runs]), axis=0)) for i in range(len(runs[0]))]


R51 = ("M6+TD6 A25", "TD6 A25", "TD6n A25", "M6+TD6n A25", "TD6+TD6n A25", "TD6P A25", "M6+TD6P A25", "M6P+TD6P A25", "v19")


def r51(k):
    """Round 51 on one provider, V6 harness."""
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    fsrc = V6[k][0]
    src = V3[k][1]
    tr3, vi3 = rule_evidence(k, V3, conf)
    tr6, vi6 = rule_evidence(k, V6, conf)
    load = lambda tag: [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M_%s_%s.pkl" % (tag, fsrc.replace("@", "-"))))]
    m3, m6, m6p = load("v3"), load("v6"), load("v6p")
    td, tn, tp = _held("tdead6", fsrc), _held("tdead6n", fsrc), _held("tdead6p", fsrc)
    ev = {"v19": [0.25 * t + 0.75 * m + v for t, m, v in zip(tr3, m3, vi3)],
          "M6+TD6 A25": [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, td, vi6)],
          "TD6 A25": [0.25 * t + 0.75 * x + v for t, x, v in zip(tr6, td, vi6)],
          "TD6n A25": [0.25 * t + 0.75 * x + v for t, x, v in zip(tr6, tn, vi6)],
          "M6+TD6n A25": [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, tn, vi6)],
          "TD6+TD6n A25": [0.25 * t + 0.375 * y + 0.375 * x + v for t, y, x, v in zip(tr6, td, tn, vi6)],
          "TD6P A25": [0.25 * t + 0.75 * x + v for t, x, v in zip(tr6, tp, vi6)],
          "M6+TD6P A25": [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, tp, vi6)],
          "M6P+TD6P A25": [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6p, tp, vi6)]}
    convs = dm.CONVS[src]
    blocks = bench.blocks(fsrc)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    truth = labs[convs[0]]
    smos = dm.team(src)
    cells = {}
    for s in es.SEEDS:
        for name in R51:
            preds = eg.decode_par(ev[name], smos[s])
            for c in convs:
                cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            if s == es.SEEDS[0]:
                print("  %-13s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
        print("  seed %d done" % s, flush=True)
    er.summarize(cells, list(R51), ref="M6+TD6 A25")
    with open(os.path.join(RUNS, "r51_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)


def r51_decide():
    import exp_blend_weight as bw
    cells = {}
    for k in V6:
        with open(os.path.join(RUNS, "r51_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    mean, _ = er.summarize(cells, list(R51), ref="M6+TD6 A25")
    for cand in R51[1:-1]:
        bw.decide(mean, "M6+TD6 A25", cand)
    for cand in R51[:-1]:
        bw.decide(mean, "v19", cand)


def r51_pff():
    """PFF raw: TD6n and the all-data TD6 never saw it; TD6P and M6P held out by match within PFF (M6P: five folds here)."""
    import gradient
    import lightgbm as lgb
    import model_v7
    import pathlib
    import exp_smooth
    import exp_tdead as etd
    import torch
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b3, g3 = weights_on(V3, conf)
    b6, g6 = weights_on(V6, conf)
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    m3 = bench.load_booster(os.path.join(OUT, "M_v3_all.txt"))
    m6 = bench.load_booster(os.path.join(OUT, "M_v6_all.txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"

    def nets(folder):
        res = []
        for seed in (20261104, 20261105, 20261106):
            ck = torch.load(os.path.join(bench.CACHE, folder, "all_%d.pt" % seed), map_location="cpu", weights_only=False)
            net = etd.Net(ck["n_in"]).to(etd.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            res.append((net, ck["mean"], ck["scale"]))
        return res

    def alive(ns, x, norm=False):
        ps = []
        for n_, m_, s_ in ns:
            if norm:
                os.environ["TD_NORM"] = "1"
            ps.append(etd.score_half(n_, m_, s_, x))
            os.environ.pop("TD_NORM", None)
        return exp_blend.logit_alive(np.mean(ps, axis=0))

    td6, tdn = nets("tdead6"), nets("tdead6n")
    tp_runs = [bench._load(os.path.join(bench.CACHE, "tdead6p", "held_%d_pff.pkl" % s)) for s in (20261104, 20261105, 20261106)]
    matches = gradient.matches()
    # M6P with PFF held out: five folds over PFF's matches
    train = train_table(V6)
    folds = [matches[i::5] for i in range(5)]
    pff_rows = {}
    for match in matches:
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if path.exists():
                    f = bench._load(str(path))
                    pff_rows.setdefault(match, []).append(({"XF": f["XF"][:, [f["fk"].index(k) for k in base_names]].astype(np.float32)}, np.asarray(d["h%d_labels" % int(per)]), (match, int(per))))
    m6p = {}
    parts0 = fit_parts(train, ())
    for fi, fold in enumerate(folds):
        fs = [r[0] for m_ in matches if m_ not in fold for r in pff_rows.get(m_, [])]
        ls = [r[1] for m_ in matches if m_ not in fold for r in pff_rows.get(m_, [])]
        model = dm.fit_balanced(dict(parts0, pff=(fs, ls)))
        for m_ in fold:
            for x_, _, key in pff_rows.get(m_, []):
                m6p[key] = exp_blend.logit_alive(model.predict(x_["XF"], num_threads=bench.THREADS))
        print("  M6P fold %d done" % fi, flush=True)
    names = ("v19", "M6+TD6 A25", "TD6 A25", "TD6n A25", "M6+TD6n A25", "TD6+TD6n A25", "TD6P A25", "M6+TD6P A25", "M6P+TD6P A25", "rules3")
    cells = {}
    dead = {k: [0, 0, 0] for k in names}
    n = 0
    hi = 0
    for match in matches:
        blocks, labs, ev, smo = [], [], {k: [] for k in names}, []
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                t0 = "h%d" % int(per)
                b = dict(P=d[t0 + "_players"], B=d[t0 + "_ball"], team=d["team"], gk=d["gk"], pitch=tuple(d["pitch"]), group=match)
                lab = np.asarray(d[t0 + "_labels"])
                x = f["XF"][:, [f["fk"].index(k) for k in base_names]]
                rf = exp_blend.rule_f(b)
                v, l = dm.transplant(lab, np.random.default_rng([dm.SEED, n]))
                n += 1
                T = ro.terms(rf, t, c)
                tr3, tr6 = T @ b3, T @ b6
                vi3, vi6 = g3[0] * (~v) + g3[1] * l, g6[0] * (~v) + g6[1] * l
                a3 = exp_blend.logit_alive(m3.predict(x, num_threads=bench.THREADS))
                a6 = exp_blend.logit_alive(m6.predict(x, num_threads=bench.THREADS))
                x6, xn = alive(td6, x), alive(tdn, x, True)
                xp = exp_blend.logit_alive(np.mean([r[hi] for r in tp_runs], axis=0))
                hi += 1
                ap = m6p[(match, int(per))]
                blocks.append(b)
                labs.append(lab)
                ev["v19"].append(0.25 * tr3 + 0.75 * a3 + vi3)
                ev["rules3"].append(tr3 + vi3)
                ev["M6+TD6 A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * x6 + vi6)
                ev["TD6 A25"].append(0.25 * tr6 + 0.75 * x6 + vi6)
                ev["TD6n A25"].append(0.25 * tr6 + 0.75 * xn + vi6)
                ev["M6+TD6n A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * xn + vi6)
                ev["TD6+TD6n A25"].append(0.25 * tr6 + 0.375 * x6 + 0.375 * xn + vi6)
                ev["TD6P A25"].append(0.25 * tr6 + 0.75 * xp + vi6)
                ev["M6+TD6P A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * xp + vi6)
                ev["M6P+TD6P A25"].append(0.25 * tr6 + 0.375 * ap + 0.375 * xp + vi6)
                smo.append(bench.predict_team(team_model, f))
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        for name in names:
            preds = eg.decode_par(ev[name], smo)
            cells.setdefault(("pff", "ctrl", "edges D", name), []).append(er.by_match(blocks, labs, preds))
            for p_, g_ in zip(preds, labs):
                ok = g_ < 3
                dead[name][0] += int(((p_ == 2) & (g_ == 2) & ok).sum())
                dead[name][1] += int(((p_ == 2) & (g_ < 2) & ok).sum())
                dead[name][2] += int(((p_ < 2) & (g_ == 2) & ok).sum())
        print("  %s done" % match, flush=True)
    merged = {}
    for k, v in cells.items():
        d = {}
        for x in v:
            d.update(x)
        merged[k] = [d]
    er.summarize(merged, list(names), ref="v19")
    for name in names:
        tp_, fp, fn = dead[name]
        print("  %-13s dead F1 %.3f, precision %.3f, recall %.3f" % (name, 2 * tp_ / (2 * tp_ + fp + fn), tp_ / max(tp_ + fp, 1), tp_ / max(tp_ + fn, 1)), flush=True)


def combo():
    """v20's two parts together against v19: the specialist's weight (Metrica de-glued for every team channel) and the dead-ball evidence, V6 harness and true D, three seeds."""
    import exp_blend_weight as bw
    import exp_gate as eg2
    import exp_smooth
    import short_specialist as ss
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    names = ("v19", "v19 s.5", "D", "D s.5", "TD s.5")
    cells = {}
    for k in V6:
        fsrc = V6[k][0]
        src = V3[k][1]
        tr3, vi3 = rule_evidence(k, V3, conf)
        tr6, vi6 = rule_evidence(k, V6, conf)
        load = lambda tag: [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M_%s_%s.pkl" % (tag, fsrc.replace("@", "-"))))]
        m3, m6 = load("v3"), load("v6")
        td = _held("tdead6", fsrc)
        ev19 = [0.25 * t + 0.75 * m + v for t, m, v in zip(tr3, m3, vi3)]
        evd = [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, td, vi6)]
        evt = [0.25 * t + 0.75 * x + v for t, x, v in zip(tr6, td, vi6)]
        convs = dm.CONVS[src]
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        true_a = [np.where(g == 2, -12.0, 12.0) for g in labs[convs[0]]]
        for sd_ in es.SEEDS:
            v, g, o = eg2.parts(src, sd_, True)
            s25 = exp_smooth.smooth([0.375 * np.asarray(x, np.float64) + 0.375 * np.asarray(y, np.float64) + 0.25 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
            s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
            arms = {"v19": (ev19, s25), "v19 s.5": (ev19, s50), "D": (evd, s25), "D s.5": (evd, s50), "TD s.5": (evt, s50)}
            for name, (e, smo) in arms.items():
                preds = eg.decode_par(e, smo)
                for c in convs:
                    cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            for name, smo in (("v19", s25), ("v19 s.5", s50), ("D", s25), ("D s.5", s50), ("TD s.5", s50)):
                preds = eg.decode_par(true_a, smo)
                for c in convs:
                    cells.setdefault((src, c, "true D", name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (k, sd_), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="v19")
    with open(os.path.join(RUNS, "r52_combo_cells.pkl"), "wb") as fh:
        pickle.dump(cells, fh)
    for cand in names[1:]:
        bw.decide(mean, "v19", cand)


def decoder_screen():
    """The decoder's settings on v20's evidence (D s.5, V6 harness, Metrica de-glued): one setting at a time, three seeds."""
    import exp_blend_weight as bw
    import exp_gate as eg2
    import exp_smooth
    import short_specialist as ss
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    grid = {"base": {}, "team8": dict(pen_team=8.0), "team16": dict(pen_team=16.0), "dead20": dict(pen_dead=20.0), "dead45": dict(pen_dead=45.0), "mind25": dict(min_dead=25), "mind75": dict(min_dead=75), "minp5": dict(min_poss=5), "minp12": dict(min_poss=12)}
    cells = {}
    for k in V6:
        fsrc = V6[k][0]
        src = V3[k][1]
        tr6, vi6 = rule_evidence(k, V6, conf)
        m6 = [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M_v6_%s.pkl" % fsrc.replace("@", "-")))]
        td = _held("tdead6", fsrc)
        evd = [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, td, vi6)]
        convs = dm.CONVS[src]
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        for sd_ in es.SEEDS:
            v, g, o = eg2.parts(src, sd_, True)
            s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
            for name, kw in grid.items():
                preds = eg.decode_par(evd, s50, **kw)
                for c in convs:
                    cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (k, sd_), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    names = list(grid)
    mean, _ = er.summarize(cells, names, ref="base")
    for cand in names[1:]:
        bw.decide(mean, "base", cand)


def r51_st(var="v6"):
    """SoccerTrack, unseen by every model here, for round 51's arms; J1 at 0.25 and 0.5."""
    import exp_smooth
    import exp_st
    import exp_tdead as etd
    import short_specialist as ss
    import torch
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b3, g3 = weights_on(V3, conf)
    b6, g6 = weights_on(V6, conf)
    m3 = bench.load_booster(os.path.join(OUT, "M_v3_all.txt"))
    m6 = bench.load_booster(os.path.join(OUT, "M_v6_all.txt"))
    m6p = bench.load_booster(os.path.join(OUT, "M_v6p_all.txt"))

    def nets(folder):
        out = []
        for seed in (20261104, 20261105, 20261106):
            ck = torch.load(os.path.join(bench.CACHE, folder, "all_%d.pt" % seed), map_location="cpu", weights_only=False)
            net = etd.Net(ck["n_in"]).to(etd.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            out.append((net, ck["mean"], ck["scale"]))
        return out

    def alive(ns, x, norm=False):
        ps = []
        for n_, m_, s_ in ns:
            if norm:
                os.environ["TD_NORM"] = "1"
            ps.append(etd.score_half(n_, m_, s_, x))
            os.environ.pop("TD_NORM", None)
        return exp_blend.logit_alive(np.mean(ps, axis=0))

    td6, tdn, tdp = nets("tdead6"), nets("tdead6n"), nets("tdead6p")
    src = "st@%s" % var
    blocks = bench.blocks(src)
    labs = {cv: [np.asarray(g) for g in bench.labels(exp_st.SRC, cv)] for cv in exp_st.CONVS}
    truth = labs["evctrl"]
    fs = bench.features("base", src)
    names = ("v19", "M6+TD6 A25", "TD6n A25", "M6+TD6n A25", "TD6P A25", "M6+TD6P A25", "M6P+TD6P A25")
    ev = {n: [] for n in names}
    for i, (b, f, g) in enumerate(zip(blocks, fs, truth)):
        T = ro.terms(exp_blend.rule_f(b), t, c)
        v, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        tr3, tr6, vi3, vi6 = T @ b3, T @ b6, g3[0] * (~v) + g3[1] * l, g6[0] * (~v) + g6[1] * l
        x = f["XF"]
        a3, a6, ap = [exp_blend.logit_alive(mm.predict(x, num_threads=bench.THREADS)) for mm in (m3, m6, m6p)]
        x6, xn, xp = alive(td6, x), alive(tdn, x, True), alive(tdp, x)
        ev["v19"].append(0.25 * tr3 + 0.75 * a3 + vi3)
        ev["M6+TD6 A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * x6 + vi6)
        ev["TD6n A25"].append(0.25 * tr6 + 0.75 * xn + vi6)
        ev["M6+TD6n A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * xn + vi6)
        ev["TD6P A25"].append(0.25 * tr6 + 0.75 * xp + vi6)
        ev["M6+TD6P A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * xp + vi6)
        ev["M6P+TD6P A25"].append(0.25 * tr6 + 0.375 * ap + 0.375 * xp + vi6)
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))
    cells = {}
    for k in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
        smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
        for name in names:
            preds = eg.decode_par(ev[name], smo)
            if k == es.SEEDS[0]:
                print("  %-13s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "edges D", name), []).append(er.by_match(blocks, labs[cv], preds))
        print("  seed %d done" % k, flush=True)
    er.summarize(cells, list(names), ref="v19")


def norm_table(fs):
    """Per-half normalised copies of the 'base' features for the LightGBM (round 54)."""
    import exp_tdead as etd
    return [{"XF": etd.half_norm(np.asarray(f["XF"], np.float64)).astype(np.float32)} for f in fs]


def fold_models_n(tag, table):
    """fold_models on per-half normalised features; PFF raw held out entirely. Cached by tag."""
    os.makedirs(OUT, exist_ok=True)
    done = os.path.join(OUT, "M_%s_done" % tag)
    if os.path.isfile(done):
        return
    train = {k: (norm_table(bench.features("base", v[0])), bench.labels(v[1], v[2]), _groups(bench.blocks(v[0]))) for k, v in table.items()}
    tests = {k: (train[k][0], train[k][2]) for k in table}
    res = {k: [None] * len(tests[k][0]) for k in table}
    for k in table:
        un = sorted(set(tests[k][1]))
        folds = [un[i::5] for i in range(5)] if k == "sc" else [[g] for g in un]
        for fold in folds:
            model = dm.fit_balanced(fit_parts(train, fold))
            for i, (x, g) in enumerate(zip(tests[k][0], tests[k][1])):
                if g in fold:
                    res[k][i] = model.predict(x["XF"], num_threads=bench.THREADS).astype(np.float32)
            print("  M %s %s fold %s done" % (tag, k, fold[:1]), flush=True)
    for k in table:
        bench._dump(res[k], os.path.join(OUT, "M_%s_%s.pkl" % (tag, table[k][0].replace("@", "-"))))
    dm.fit_balanced(fit_parts(train, ())).save_model(os.path.join(OUT, "M_%s_all.txt" % tag))
    with open(done, "w") as fh:
        fh.write("ok")


def r54_models():
    fold_models_n("v6n", V6)


R54 = ("M6+TD6n A25", "M6n+TD6n A25", "M6+TDmix A25", "M6+TDmixn A25", "TDmixn A25", "M6n+TDmixn A25", "R n M6+TD6n", "v19")


def norm_rules(fields):
    """The rule fields with tempo rescaled by the half's own median to the pooled median (label-free): the rules' only absolute-tempo inputs."""
    out = dict(fields)
    with np.errstate(invalid="ignore"):
        for key, ref in (("spd_all_s25", 1.7424), ("spd_all_s100", 1.8224)):
            med = np.nanmedian(fields[key])
            out[key] = fields[key] * ref / med
    return out


def r54(k, var=None):
    """Round 54 on one provider, on V6 (or V3 / V6B masks for the sensitivity: var)."""
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    fsrc = V6[k][0]
    src = V3[k][1]
    tr3, vi3 = rule_evidence(k, V3, conf)
    tr6, vi6 = rule_evidence(k, V6, conf)
    load = lambda tag: [exp_blend.logit_alive(x) for x in bench._load(os.path.join(OUT, "M_%s_%s.pkl" % (tag, fsrc.replace("@", "-"))))]
    m3, m6, mn = load("v3"), load("v6"), load("v6n")
    tn, tm, tmn = _held("tdead6n", fsrc), _held("tdeadmix", fsrc), _held("tdeadmixn", fsrc)
    blocks = bench.blocks(fsrc)
    # the rules with the half's tempo level removed, weights refitted on the same footing
    t, c = conf["thresholds"], conf["bounds"]
    ro.SOURCES = V6
    ro._f.clear()
    d = ro.data()
    un = sorted({str(x[2]) for x in d[k]})
    folds = [un[i::5] for i in range(5)] if k == "sc" else [[g] for g in un]
    fields = [norm_rules(exp_blend.rule_f(x)) for x in blocks]
    trn = [None] * len(blocks)
    grp = _groups(blocks)
    for fold in folds:
        Xs, ys, ws = [], [], []
        for p_ in d:
            xk, yk = [], []
            for f, g, gr, _, _ in d[p_]:
                if str(gr) in fold:
                    continue
                m = g[::ro.STRIDE] < 3
                xk.append(ro.terms(norm_rules(f), t, c)[::ro.STRIDE][m])
                yk.append((g[::ro.STRIDE][m] < 2).astype(float))
            x_, y_ = np.concatenate(xk), np.concatenate(yk)
            Xs.append(x_)
            ys.append(y_)
            ws.append(np.full(len(y_), 1.0 / len(y_)))
        beta, _ = ro.logistic(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))
        for i, g in enumerate(grp):
            if g in fold:
                trn[i] = ro.terms(fields[i], t, c) @ beta
    ro.SOURCES = V3
    ro._f.clear()
    bench._blocks.clear()
    bench._labels.clear()
    blocks = bench.blocks(fsrc)
    ev = {"v19": [0.25 * t_ + 0.75 * m + v for t_, m, v in zip(tr3, m3, vi3)],
          "M6+TD6n A25": [0.25 * t_ + 0.375 * m + 0.375 * x + v for t_, m, x, v in zip(tr6, m6, tn, vi6)],
          "M6n+TD6n A25": [0.25 * t_ + 0.375 * m + 0.375 * x + v for t_, m, x, v in zip(tr6, mn, tn, vi6)],
          "M6+TDmix A25": [0.25 * t_ + 0.375 * m + 0.375 * x + v for t_, m, x, v in zip(tr6, m6, tm, vi6)],
          "M6+TDmixn A25": [0.25 * t_ + 0.375 * m + 0.375 * x + v for t_, m, x, v in zip(tr6, m6, tmn, vi6)],
          "TDmixn A25": [0.25 * t_ + 0.75 * x + v for t_, x, v in zip(tr6, tmn, vi6)],
          "M6n+TDmixn A25": [0.25 * t_ + 0.375 * m + 0.375 * x + v for t_, m, x, v in zip(tr6, mn, tmn, vi6)],
          "R n M6+TD6n": [0.25 * t_ + 0.375 * m + 0.375 * x + v for t_, m, x, v in zip(trn, m6, tn, vi6)]}
    convs = dm.CONVS[src]
    labs = {c_: [np.asarray(g) for g in bench.labels(src, c_)] for c_ in convs}
    truth = labs[convs[0]]
    smos = dm.team(src)
    cells = {}
    for s_ in es.SEEDS:
        for name in R54:
            preds = eg.decode_par(ev[name], smos[s_])
            for c_ in convs:
                cells.setdefault((src, c_, "edges D", name), []).append(er.by_match(blocks, labs[c_], preds))
            if s_ == es.SEEDS[0]:
                print("  %-15s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
        print("  seed %d done" % s_, flush=True)
    er.summarize(cells, list(R54), ref="M6+TD6n A25")
    with open(os.path.join(RUNS, "r54_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)


def r54_decide():
    import exp_blend_weight as bw
    cells = {}
    for k in V6:
        with open(os.path.join(RUNS, "r54_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    mean, _ = er.summarize(cells, list(R54), ref="M6+TD6n A25")
    for cand in R54[1:-1]:
        bw.decide(mean, "M6+TD6n A25", cand)
    for cand in R54[:-1]:
        bw.decide(mean, "v19", cand)


def r54_unseen():
    """PFF raw and SoccerTrack for round 54's arms: all-data models, none saw these sets."""
    import gradient
    import lightgbm as lgb
    import model_v7
    import pathlib
    import exp_smooth
    import exp_st
    import exp_tdead as etd
    import short_specialist as ss
    import torch
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b3, g3 = weights_on(V3, conf)
    b6, g6 = weights_on(V6, conf)
    m3 = bench.load_booster(os.path.join(OUT, "M_v3_all.txt"))
    m6 = bench.load_booster(os.path.join(OUT, "M_v6_all.txt"))
    mn = bench.load_booster(os.path.join(OUT, "M_v6n_all.txt"))

    def nets(folder):
        res = []
        for seed in (20261104, 20261105, 20261106):
            ck = torch.load(os.path.join(bench.CACHE, folder, "all_%d.pt" % seed), map_location="cpu", weights_only=False)
            net = etd.Net(ck["n_in"]).to(etd.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            res.append((net, ck["mean"], ck["scale"]))
        return res

    def alive(ns, x, norm=False):
        ps = []
        for n_, m_, s_ in ns:
            if norm:
                os.environ["TD_NORM"] = "1"
            ps.append(etd.score_half(n_, m_, s_, x))
            os.environ.pop("TD_NORM", None)
        return exp_blend.logit_alive(np.mean(ps, axis=0))

    tn, tm, tmn = nets("tdead6n"), nets("tdeadmix"), nets("tdeadmixn")
    names = ("v19", "M6+TD6n A25", "M6n+TD6n A25", "M6+TDmix A25", "M6+TDmixn A25", "TDmixn A25", "M6n+TDmixn A25")

    def arms(T, x, v, l):
        tr3, tr6, vi3, vi6 = T @ b3, T @ b6, g3[0] * (~v) + g3[1] * l, g6[0] * (~v) + g6[1] * l
        a3 = exp_blend.logit_alive(m3.predict(x, num_threads=bench.THREADS))
        a6 = exp_blend.logit_alive(m6.predict(x, num_threads=bench.THREADS))
        an = exp_blend.logit_alive(mn.predict(etd.half_norm(np.asarray(x, np.float64)), num_threads=bench.THREADS))
        xn, xm, xmn = alive(tn, x, True), alive(tm, x), alive(tmn, x, True)
        return {"v19": 0.25 * tr3 + 0.75 * a3 + vi3, "M6+TD6n A25": 0.25 * tr6 + 0.375 * a6 + 0.375 * xn + vi6, "M6n+TD6n A25": 0.25 * tr6 + 0.375 * an + 0.375 * xn + vi6,
                "M6+TDmix A25": 0.25 * tr6 + 0.375 * a6 + 0.375 * xm + vi6, "M6+TDmixn A25": 0.25 * tr6 + 0.375 * a6 + 0.375 * xmn + vi6, "TDmixn A25": 0.25 * tr6 + 0.75 * xmn + vi6,
                "M6n+TDmixn A25": 0.25 * tr6 + 0.375 * an + 0.375 * xmn + vi6}

    # SoccerTrack
    src = "st@v6"
    blocks = bench.blocks(src)
    labs = {cv: [np.asarray(g) for g in bench.labels(exp_st.SRC, cv)] for cv in exp_st.CONVS}
    truth = labs["evctrl"]
    ev = {n: [] for n in names}
    for i, (b, f, g) in enumerate(zip(blocks, bench.features("base", src), truth)):
        v, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        res = arms(ro.terms(exp_blend.rule_f(b), t, c), f["XF"], v, l)
        for n in names:
            ev[n].append(res[n])
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))
    cells = {}
    for k in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
        smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
        for name in names:
            preds = eg.decode_par(ev[name], smo)
            if k == es.SEEDS[0]:
                print("  ST %-15s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "edges D", name), []).append(er.by_match(blocks, labs[cv], preds))
    er.summarize(cells, list(names), ref="v19")
    # PFF raw
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    cells = {}
    dead = {k: [0, 0, 0] for k in names}
    n = 0
    for match in gradient.matches():
        blocks, labs, ev, smo = [], [], {k: [] for k in names}, []
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                t0 = "h%d" % int(per)
                b = dict(P=d[t0 + "_players"], B=d[t0 + "_ball"], team=d["team"], gk=d["gk"], pitch=tuple(d["pitch"]), group=match)
                lab = np.asarray(d[t0 + "_labels"])
                v, l = dm.transplant(lab, np.random.default_rng([dm.SEED, n]))
                n += 1
                res = arms(ro.terms(exp_blend.rule_f(b), t, c), f["XF"][:, [f["fk"].index(k) for k in base_names]], v, l)
                for k in names:
                    ev[k].append(res[k])
                blocks.append(b)
                labs.append(lab)
                smo.append(bench.predict_team(team_model, f))
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        for name in names:
            preds = eg.decode_par(ev[name], smo)
            cells.setdefault(("pff", "ctrl", "edges D", name), []).append(er.by_match(blocks, labs, preds))
            for p_, g_ in zip(preds, labs):
                ok = g_ < 3
                dead[name][0] += int(((p_ == 2) & (g_ == 2) & ok).sum())
                dead[name][1] += int(((p_ == 2) & (g_ < 2) & ok).sum())
                dead[name][2] += int(((p_ < 2) & (g_ == 2) & ok).sum())
        print("  %s done" % match, flush=True)
    merged = {}
    for k, v_ in cells.items():
        d_ = {}
        for x in v_:
            d_.update(x)
        merged[k] = [d_]
    er.summarize(merged, list(names), ref="v19")
    for name in names:
        tp_, fp, fn = dead[name]
        print("  %-15s dead F1 %.3f, precision %.3f, recall %.3f" % (name, 2 * tp_ / (2 * tp_ + fp + fn), tp_ / max(tp_ + fp, 1), tp_ / max(tp_ + fn, 1)), flush=True)


def soccertrack(var="v6"):
    """SoccerTrack v2: a full-pitch camera and the provider's own per-frame ball status; no model saw it. Team channels frozen as in round 44; masks V3 or V6; all-data dead models."""
    import exp_smooth
    import exp_st
    import exp_tdead as etd
    import short_specialist as ss
    import torch
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b3, g3 = weights_on(V3, conf)
    b6, g6 = weights_on(V6, conf)
    m3 = bench.load_booster(os.path.join(OUT, "M_v3_all.txt"))
    m6 = bench.load_booster(os.path.join(OUT, "M_v6_all.txt"))

    def nets(folder):
        out = []
        for seed in (20261104, 20261105, 20261106):
            ck = torch.load(os.path.join(bench.CACHE, folder, "all_%d.pt" % seed), map_location="cpu", weights_only=False)
            net = etd.Net(ck["n_in"]).to(etd.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            out.append((net, ck["mean"], ck["scale"]))
        return out

    td3, td6 = nets("tdead"), nets("tdead6")
    src = "st@%s" % var
    blocks = bench.blocks(src)
    labs = {cv: [np.asarray(g) for g in bench.labels(exp_st.SRC, cv)] for cv in exp_st.CONVS}
    truth = labs["evctrl"]
    fs = bench.features("base", src)
    names = ("v19", "rules3", "M6 A25", "M6 A50", "TD3 A25", "M3+TD3 A25", "TD6 A25", "M6+TD6 A25", "M6+TD6 A50")
    ev = {n: [] for n in names}
    for i, (b, f, g) in enumerate(zip(blocks, fs, truth)):
        T = ro.terms(exp_blend.rule_f(b), t, c)
        v, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        tr3, tr6, vi3, vi6 = T @ b3, T @ b6, g3[0] * (~v) + g3[1] * l, g6[0] * (~v) + g6[1] * l
        a3, a6 = exp_blend.logit_alive(m3.predict(f["XF"], num_threads=bench.THREADS)), exp_blend.logit_alive(m6.predict(f["XF"], num_threads=bench.THREADS))
        x3 = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, f["XF"]) for n_, m_, s_ in td3], axis=0))
        x6 = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, f["XF"]) for n_, m_, s_ in td6], axis=0))
        ev["v19"].append(0.25 * tr3 + 0.75 * a3 + vi3)
        ev["rules3"].append(tr3 + vi3)
        ev["M6 A25"].append(0.25 * tr6 + 0.75 * a6 + vi6)
        ev["M6 A50"].append(0.5 * tr6 + 0.5 * a6 + vi6)
        ev["TD3 A25"].append(0.25 * tr3 + 0.75 * x3 + vi3)
        ev["M3+TD3 A25"].append(0.25 * tr3 + 0.375 * a3 + 0.375 * x3 + vi3)
        ev["TD6 A25"].append(0.25 * tr6 + 0.75 * x6 + vi6)
        ev["M6+TD6 A25"].append(0.25 * tr6 + 0.375 * a6 + 0.375 * x6 + vi6)
        ev["M6+TD6 A50"].append(0.5 * tr6 + 0.25 * a6 + 0.25 * x6 + vi6)
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))
    cells = {}
    for k in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
        smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
        s50 = exp_smooth.smooth([0.5 * r + 0.5 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
        for name in names:
            preds = eg.decode_par(ev[name], smo)
            if k == es.SEEDS[0]:
                print("  %-11s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "edges D", name), []).append(er.by_match(blocks, labs[cv], preds))
        for name, e in (("v19 s.5", ev["v19"]), ("D s.5", ev["M6+TD6 A25"]), ("TD s.5", ev["TD6 A25"])):
            preds = eg.decode_par(e, s50)
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "edges D", name), []).append(er.by_match(blocks, labs[cv], preds))
        print("  seed %d done" % k, flush=True)
    er.summarize(cells, list(names) + ["v19 s.5", "D s.5", "TD s.5"], ref="v19")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"inputs": inputs, "models": models, "eval": evaluate, "decide": decide, "pff": pff, "st": soccertrack, "heads": boundary_heads, "stress": stress, "r51": r51, "r51decide": r51_decide, "r51pff": r51_pff, "combo": combo, "decoder": decoder_screen, "r51st": r51_st, "r54models": r54_models, "r54": r54, "r54decide": r54_decide, "r54unseen": r54_unseen}[sys.argv[1]](*sys.argv[2:])
