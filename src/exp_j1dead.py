"""Round 63 (docs/RAPOR.md): the specialist J1 has its own dead-ball head, at 25 Hz on a ball-centred view, trained partly on real tracking (PFF), and v20's dead-ball evidence ignores it. Does it add? No new training, decode level. Aggregate output only.

  python -I src/exp_j1dead.py eval K     K in dfl, metrica, sc: arms against v20, 3 seeds
  python -I src/exp_j1dead.py st         SoccerTrack, unseen by every model
  python -I src/exp_j1dead.py decide
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

RUNS = os.path.join(bench.ROOT, "tools", "runs")
NAMES = ("v20", "J.125", "J.25", "J+.25")


def mix(t, m, x, v, j):
    return {"v20": 0.25 * t + 0.375 * m + 0.375 * x + v,
            "J.125": 0.25 * t + 0.3125 * m + 0.3125 * x + 0.125 * j + v,
            "J.25": 0.25 * t + 0.25 * m + 0.25 * x + 0.25 * j + v,
            "J+.25": 0.25 * t + 0.375 * m + 0.375 * x + 0.25 * j + v}


def j_alive(o):
    return exp_blend.logit_alive(np.asarray(o, np.float32)[:, 2])


def evaluate(k):
    with open(ro.OUT) as fh:
        conf = json.load(fh)
    fsrc, src = ee.V6[k][0], ee.V3[k][1]
    tr6, vi6 = ee.rule_evidence(k, ee.V6, conf)
    m6 = [exp_blend.logit_alive(x) for x in bench._load(os.path.join(ee.OUT, "M_v6_%s.pkl" % fsrc.replace("@", "-")))]
    tn = ee._held("tdead6n", fsrc)
    blocks = bench.blocks(fsrc)
    convs = dm.CONVS[src]
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    truth = labs[convs[0]]
    cells = {}
    for s_i in range(len(es.SEEDS)):
        v, g, o = eg2.parts(src, s_i, True)
        s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
        evs = {n: [] for n in NAMES}
        for t, m, x, vv, oo in zip(tr6, m6, tn, vi6, o):
            res = mix(t, m, x, vv, j_alive(oo))
            for n in NAMES:
                evs[n].append(res[n])
        for n in NAMES:
            preds = eg.decode_par(evs[n], s50)
            for c in convs:
                cells.setdefault((src, c, "edges D", n), []).append(er.by_match(blocks, labs[c], preds))
            if s_i == 0:
                print("  %-6s dead share %.3f (true %.3f), dead F1 %.3f" % (n, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
        print("  %s seed %d done" % (k, s_i), flush=True)
    er.summarize(cells, list(NAMES), ref="v20")
    with open(os.path.join(RUNS, "r63_cells_%s.pkl" % k), "wb") as fh:
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
    parts = []
    os.environ["TD_NORM"] = "1"
    for i, (b, f, g) in enumerate(zip(blocks, bench.features("base", src), truth)):
        T = ro.terms(exp_blend.rule_f(b), t, c)
        vv, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        a6 = exp_blend.logit_alive(m6.predict(f["XF"], num_threads=bench.THREADS))
        xn = exp_blend.logit_alive(np.mean([etd.score_half(n_, m_, s_, f["XF"]) for n_, m_, s_ in nets], axis=0))
        parts.append((T @ b6, a6, xn, g6[0] * (~vv) + g6[1] * l))
    os.environ.pop("TD_NORM")
    cells = {}
    for s_i in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(s_i), gru, 0.5)
        oo = outs[ss.SEEDS[s_i]]
        s50 = exp_smooth.smooth([0.5 * r + 0.5 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, oo)], 25)
        evs = {n: [] for n in NAMES}
        for (tt, a6, xn, vw), o in zip(parts, oo):
            res = mix(tt, a6, xn, vw, j_alive(o))
            for n in NAMES:
                evs[n].append(res[n])
        for n in NAMES:
            preds = eg.decode_par(evs[n], s50)
            for cv in exp_st.CONVS:
                cells.setdefault((exp_st.SRC, cv, "v6", n), []).append(er.by_match(blocks, labs[cv], preds))
            if s_i == es.SEEDS[0]:
                print("  %-6s dead share %.3f (true %.3f), dead F1 %.3f" % (n, np.mean(np.concatenate(preds) == 2), np.mean(np.concatenate(truth) == 2), dm._dead_f1(preds, truth)), flush=True)
    er.summarize(cells, list(NAMES), ref="v20")


def decide():
    import exp_blend_weight as bw
    cells = {}
    for k in ee.V6:
        with open(os.path.join(RUNS, "r63_cells_%s.pkl" % k), "rb") as fh:
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
    elif cmd == "decide":
        decide()
    else:
        raise SystemExit(__doc__)
