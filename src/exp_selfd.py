"""Round 62 (docs/RAPOR.md): the dead-ball evidence adapted to the half itself. Per-half normalisation gave the largest gain on unseen data, so the domain gap is real; self-training had been tried only on the team channel. Each half: frames where v20's alive evidence is confident (|a| >= tau) become pseudo-labels, a small LightGBM learns them from the half's own tracking features, and the new evidence is (1 - w) a + w a2. Pseudo-labels come from an algorithm only (official RULES allow them). Aggregate output only.

  python -I src/exp_selfd.py eval K          K in dfl, metrica, sc: arms against v20, 3 seeds
  python -I src/exp_selfd.py st              SoccerTrack, unseen by every model
  python -I src/exp_selfd.py pff             PFF raw, unseen by the dead-ball models
  python -I src/exp_selfd.py decide          the decision on the first three
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = list(sys.argv)
sys.argv = [sys.argv[0], "", "none"]
import bench  # noqa: E402
import exp_blend  # noqa: E402
import exp_deadmix as dm  # noqa: E402
import exp_deglue as eg  # noqa: E402
import exp_edges as ee  # noqa: E402
import exp_gate as eg2  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_rules_open as ro  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import short_specialist as ss  # noqa: E402
sys.argv = _argv

import lightgbm as lgb  # noqa: E402

RUNS = os.path.join(bench.ROOT, "tools", "runs")
OUT = os.path.join(bench.CACHE, "selfd")
ARMS = {"t2 w.25": (2.0, 0.25), "t2 w.5": (2.0, 0.5), "t4 w.25": (4.0, 0.25), "t4 w.5": (4.0, 0.5)}
NAMES = ("v20",) + tuple(ARMS)
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=int(os.environ.get("SELFD_THREADS", "4")), seed=20261106, deterministic=True, force_col_wise=True)
ROUNDS = 60


def self_logit(xf, a, tau):
    """Alive log-odds of a model fitted on this half's own confident frames, 5 Hz; None when one class is too rare."""
    idx = np.arange(0, len(a), 5)
    sel = idx[np.abs(a[idx]) >= tau]
    y = (a[sel] > 0).astype(np.int8)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n0 < 100 or n1 < 100:
        return None
    w = np.where(y == 1, 0.5 / n1, 0.5 / n0) * len(y)
    model = lgb.train(PARAMS, lgb.Dataset(xf[sel], label=y, weight=w, free_raw_data=True), ROUNDS)
    z = model.predict(xf, raw_score=True, num_threads=PARAMS["num_threads"])
    return np.clip(z, -12, 12)


def adapted(xf, a):
    a = np.asarray(a, np.float64)
    out = {}
    cache = {}
    for name, (tau, w) in ARMS.items():
        if tau not in cache:
            cache[tau] = self_logit(xf, a, tau)
        z = cache[tau]
        out[name] = a if z is None else (1 - w) * a + w * z
    return out


def evidence_open(k):
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    fsrc = ee.V6[k][0]
    tr6, vi6 = ee.rule_evidence(k, ee.V6, conf)
    m6 = [exp_blend.logit_alive(x) for x in bench._load(os.path.join(ee.OUT, "M_v6_%s.pkl" % fsrc.replace("@", "-")))]
    tn = ee._held("tdead6n", fsrc)
    return [0.25 * t + 0.375 * m + 0.375 * x + v for t, m, x, v in zip(tr6, m6, tn, vi6)]


def evaluate(k):
    fsrc, src = ee.V6[k][0], ee.V3[k][1]
    blocks = bench.blocks(fsrc)
    convs = dm.CONVS[src]
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    feats = bench.features("base", fsrc)
    ev0 = evidence_open(k)
    ev = {"v20": ev0}
    for name in ARMS:
        ev[name] = []
    for i, (f, a) in enumerate(zip(feats, ev0)):
        res = adapted(f["XF"], a)
        for name in ARMS:
            ev[name].append(res[name])
        if i % 10 == 0:
            print("  %s half %d of %d" % (k, i, len(ev0)), flush=True)
    truth = labs[convs[0]]
    cells = {}
    for s_i in range(len(es.SEEDS)):
        v, g, o = eg2.parts(src, s_i, True)
        s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
        for name in NAMES:
            preds = eg.decode_par(ev[name], s50)
            for c in convs:
                cells.setdefault((src, c, "edges D", name), []).append(er.by_match(blocks, labs[c], preds))
            if s_i == 0:
                print("  %-8s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
        print("  %s seed %d done" % (k, s_i), flush=True)
    er.summarize(cells, list(NAMES), ref="v20")
    with open(os.path.join(RUNS, "r62_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)


def st():
    import exp_st
    import exp_tdead as etd
    import torch
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b6, g6 = ee.weights_on(ee.V6, conf)
    m6 = bench.load_booster(os.path.join(ee.OUT, "M_v6_all.txt"))
    nets = []
    for seed in (20261104, 20261105, 20261106):
        ck = torch.load(os.path.join(bench.CACHE, "tdead6n", "all_%d.pt" % seed), map_location="cpu", weights_only=False)
        net = etd.Net(ck["n_in"]).to(etd.DEVICE)
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append((net, ck["mean"], ck["scale"]))
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % exp_st.SRC.replace("@", "-")))
    labs = {cv: [np.asarray(g) for g in bench.labels(exp_st.SRC, cv)] for cv in exp_st.CONVS}
    truth = labs["evctrl"]
    src = "st@v6"
    blocks = bench.blocks(src)
    ev = {n: [] for n in NAMES}
    os.environ["TD_NORM"] = "1"
    for i, (b, f, g) in enumerate(zip(blocks, bench.features("base", src), truth)):
        T = ro.terms(exp_blend.rule_f(b), t, c)
        vv, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        a6 = exp_blend.logit_alive(m6.predict(f["XF"], num_threads=bench.THREADS))
        xn = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, f["XF"]) for n_, m_, s_ in nets], axis=0))
        a = 0.25 * (T @ b6) + 0.375 * a6 + 0.375 * xn + g6[0] * (~vv) + g6[1] * l
        ev["v20"].append(a)
        res = adapted(f["XF"], a)
        for name in ARMS:
            ev[name].append(res[name])
    os.environ.pop("TD_NORM")
    cells = {}
    for s_i in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(s_i), gru, 0.5)
        s50 = exp_smooth.smooth([0.5 * r + 0.5 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[s_i]])], 25)
        for name in NAMES:
            preds = eg.decode_par(ev[name], s50)
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "v6", name), []).append(er.by_match(blocks, labs[cv], preds))
            if s_i == es.SEEDS[0]:
                print("  %-8s dead share %.3f (true %.3f), dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
    er.summarize(cells, list(NAMES), ref="v20")


def pff():
    """As exp_edges.r51_pff for v20's dead evidence; v7's team model as in that test (never saw PFF)."""
    import gradient
    import model_v7
    import pathlib
    import exp_tdead as etd
    import torch
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    t, c = conf["thresholds"], conf["bounds"]
    b6, g6 = ee.weights_on(ee.V6, conf)
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    m6 = bench.load_booster(os.path.join(ee.OUT, "M_v6_all.txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    nets = []
    for seed in (20261104, 20261105, 20261106):
        ck = torch.load(os.path.join(bench.CACHE, "tdead6n", "all_%d.pt" % seed), map_location="cpu", weights_only=False)
        net = etd.Net(ck["n_in"]).to(etd.DEVICE)
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append((net, ck["mean"], ck["scale"]))
    cells = {}
    dead = {k: [0, 0, 0] for k in NAMES}
    n = 0
    for match in gradient.matches():
        blocks, labs, ev, smo = [], [], {k: [] for k in NAMES}, []
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
                vv, l = dm.transplant(lab, np.random.default_rng([dm.SEED, n]))
                n += 1
                T = ro.terms(exp_blend.rule_f(b), t, c)
                a6 = exp_blend.logit_alive(m6.predict(x, num_threads=bench.THREADS))
                os.environ["TD_NORM"] = "1"
                xn = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, x) for n_, m_, s_ in nets], axis=0))
                os.environ.pop("TD_NORM")
                a = 0.25 * (T @ b6) + 0.375 * a6 + 0.375 * xn + g6[0] * (~vv) + g6[1] * l
                ev["v20"].append(a)
                res = adapted(x, a)
                for name in ARMS:
                    ev[name].append(res[name])
                blocks.append(b)
                labs.append(lab)
                smo.append(bench.predict_team(team_model, f))
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        for name in NAMES:
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
        for x_ in v:
            d.update(x_)
        merged[k] = [d]
    er.summarize(merged, list(NAMES), ref="v20")
    for name in NAMES:
        tp_, fp, fn = dead[name]
        print("  %-8s dead F1 %.3f, precision %.3f, recall %.3f" % (name, 2 * tp_ / (2 * tp_ + fp + fn), tp_ / max(tp_ + fp, 1), tp_ / max(tp_ + fn, 1)), flush=True)


def decide():
    import exp_blend_weight as bw
    cells = {}
    for k in ee.V6:
        with open(os.path.join(RUNS, "r62_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    mean, _ = er.summarize(cells, list(NAMES), ref="v20")
    for cand in NAMES[1:]:
        bw.decide(mean, "v20", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "eval":
        evaluate(sys.argv[2])
    elif cmd == "st":
        st()
    elif cmd == "pff":
        pff()
    elif cmd == "decide":
        decide()
    else:
        raise SystemExit(__doc__)
