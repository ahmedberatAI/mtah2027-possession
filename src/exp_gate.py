"""Round 48 (docs/RAPOR.md, pre-registered): does the mix of the team channels pay to depend on what is observable? Open-data diagnosis: the ball-centred specialist J1 is the best channel within 1 s of a true change and the worst
while the ball is unseen. Nothing here reads the truth; the gates are the ball's visibility and J1's own change heads. Decode level, v19's dead evidence and true D, three seeds.

  today   0.75 x (0.5 v10 + 0.5 BiGRU) + 0.25 x J1, +-1 s logit smoothing
  G1      J1's weight 0 while the ball has been unseen for > 1 s (the other two keep their ratio)
  G2      J1's weight 0.5 where its own gain / lose heads exceed 0.3 within +-0.5 s
  G3      G1 + G2
  L       the same weights, mean of logits instead of probabilities
  S6      J1 smoothed over +-6 frames only, the other two over +-25, mixed afterwards
  T8, T16 today's probabilities; a change of team costs lam x (gain of the new team + loss of the old) / 2 less (never below 0)

  python -I src/exp_gate.py eval [SEEDS]   e.g. eval 0 for one v10 seed
  python -I src/exp_gate.py decide
"""
import os
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_blend  # noqa: E402
import exp_deadmix as dm  # noqa: E402
import exp_deglue as eg  # noqa: E402
import exp_kpu  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_rules_open as ro  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import exp_v7  # noqa: E402
import decode2  # noqa: E402
import short_specialist as ss  # noqa: E402

RUNS = os.path.join(bench.ROOT, "tools", "runs")
WORKERS = int(os.environ.get("GATE_WORKERS", "6"))
EPS = 1e-4
W = dict(v=0.375, g=0.375, s=0.25)
NAMES = ("today", "G1", "G2", "G3", "L", "S6", "T8", "T16")
HEAD_TAU, HEAD_W = 0.3, 12
PEN_TEAM, PEN_DEAD, MIN_POSS, MIN_DEAD = 12.0, 30.0, 8, 50


def spec_chan():
    ramp = exp_v7.ramp(175, 48.0)
    macros = [dict(emis=0, out=0, L=len(ramp), min_exit=MIN_POSS - 1, exit_cost=ramp),
              dict(emis=1, out=1, L=len(ramp), min_exit=MIN_POSS - 1, exit_cost=ramp),
              dict(emis=2, out=2, L=MIN_DEAD)]
    pen = [[0, PEN_TEAM, PEN_DEAD], [PEN_TEAM, 0, PEN_DEAD], [PEN_DEAD, PEN_DEAD, 0]]
    return decode2.Spec(macros, pen, [[-1, 0, -1], [1, -1, -1], [-1, -1, -1]])


SPEC_PLAIN = decode2.duration_spec(PEN_TEAM, PEN_DEAD, MIN_POSS, MIN_DEAD, exp_v7.ramp(175, 48.0))
SPEC_CHAN = spec_chan()


def emis(a, ph):
    q = np.clip(ph, EPS, 1 - EPS)
    return np.stack([a / 2 + np.log(q), a / 2 + np.log(1 - q), -a / 2], 1)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def peaks(x, w):
    """Maximum over +-w frames."""
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(x, 2 * w + 1, mode="nearest")


def parts(src, k, deglued):
    """Per half: v10, BiGRU, J1's outputs. Metrica's first two on the de-glued input when asked (needs round 45's files)."""
    outs = exp_kpu.outputs("J1", src)[ss.SEEDS[k]]
    if src.startswith("metrica") and deglued:
        gru = bench._load(os.path.join(eg.OUT, "bigru_members_v3g.pkl"))["members"][k]
        v = bench._load(os.path.join(eg.OUT, "v10g_seed%d.pkl" % eg.SEED_OF[k]))["today"]
    else:
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
        gru = members[k]
        v = es.v10_raw(k, src)
    return v, gru, outs


def gates(blocks):
    res = []
    for b in blocks:
        f = exp_blend.rule_f(b)
        res.append((~f["seen"]) & (f["since_seen"] > 25))
    return res


def arms(v, g, o, gap):
    """{arm: (smoothed team probability, channel or None)} of one half."""
    v, g = np.asarray(v, np.float64), np.asarray(g, np.float64)
    o = np.asarray(o, np.float64)
    s = ss.pk(o, 0).astype(np.float64)
    heads = peaks(o[:, 3:7].max(1), HEAD_W) > HEAD_TAU

    def mix(ws):
        return ((W["v"] / 0.75) * v + (W["g"] / 0.75) * g) * (1 - ws) + ws * s

    w0 = np.full(len(v), W["s"])
    res = {"today": mix(w0)}
    res["G1"] = mix(np.where(gap, 0.0, W["s"]))
    res["G2"] = mix(np.where(heads, 0.5, W["s"]))
    res["G3"] = mix(np.where(gap, 0.0, np.where(heads, 0.5, W["s"])))
    res["L"] = sig((W["v"] * logit(v) + W["g"] * logit(g) + W["s"] * logit(s)))
    smo = {n: exp_smooth.smooth([p], 25)[0] for n, p in res.items()}
    base = exp_smooth.smooth([(W["v"] / 0.75) * v + (W["g"] / 0.75) * g], 25)[0]
    smo["S6"] = 0.75 * base + 0.25 * exp_smooth.smooth([s], 6)[0]
    # channels: 0 = into A from H, 1 = into H from A (gain H, gain A, lose H, lose A = columns 3, 4, 5, 6)
    into_a = (o[:, 4] + o[:, 5]) / 2.0
    into_h = (o[:, 3] + o[:, 6]) / 2.0
    out = {n: (p, None) for n, p in smo.items()}
    for name, lam in (("T8", 8.0), ("T16", 16.0)):
        chan = -np.minimum(lam * np.stack([into_a, into_h], 1), PEN_TEAM)
        out[name] = (smo["today"], chan)
    return out


def evaluate(seeds="0,1,2", deglued="0"):
    seeds = tuple(int(x) for x in seeds.split(","))
    deglued = deglued == "1"
    cells = {}
    pool = ThreadPoolExecutor(WORKERS)
    for src, convs in dm.CONVS.items():
        fsrc = eg.G if src.startswith("metrica") else src
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        a = ro.a25_evidence(src)
        gaps = gates(blocks)
        true_a = [np.where(g == 2, -12.0, 12.0) for g in labs[convs[0]]]
        for k in seeds:
            v, g, o = parts(src, k, deglued)
            halves = [arms(x, y, z, q) for x, y, z, q in zip(v, g, o, gaps)]
            for name in NAMES:
                for dname, ev2 in (("v19 D", a), ("true D", true_a)):
                    def one(i):
                        ph, chan = halves[i][name]
                        S = emis(ev2[i], ph)
                        return (SPEC_PLAIN.decode(S) if chan is None else SPEC_CHAN.decode(S, chan)).astype(np.int8)
                    preds = list(pool.map(one, range(len(halves))))
                    for c in convs:
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (src, k), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(NAMES), ref="today")
    with open(os.path.join(RUNS, "r48_cells%s.pkl" % ("_g" if deglued else "")), "wb") as fh:
        pickle.dump(cells, fh)
    import exp_blend_weight as bw
    for cand in NAMES[1:]:
        bw.decide(mean, "today", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"eval": evaluate}[sys.argv[1]](*sys.argv[2:])
