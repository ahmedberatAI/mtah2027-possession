"""Round 39 (docs/RAPOR.md): the verifier in BAR-ATH's condition. v16 decodes BAR-ATH with its rule-based dead ball; the verifier was trained and scored only on decodes with the learned or the true dead ball, and on BAR-ATH it nearly
stops working (1 insert accepted of 54 candidates; about 4 per match on the open sets). Here the open sets get a rules channel (exp_kpu build R1abETm: the decode with v3's rules, plus image cues on SkillCorner) and every arm is scored on it,
before and after the dead-ball timing step (DT), as v16 runs:
  decoded      no verifier
  shipped      verifier trained as v15's (learned + true dead ball rows of the other matches)
  added        the same plus the rules rows of the other matches
  rules only   only the rules rows of the other matches
Three v10 seeds; verifier seed 0, 1. AGGREGATE output only.

  python -I src/exp_rules_ctx.py eval [SEEDS] [VSEEDS]
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bench  # noqa: E402
import dead_timing  # noqa: E402
import exp_blend  # noqa: E402
import exp_dsplit as ds  # noqa: E402
import exp_kpu  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import short_specialist as ss  # noqa: E402

SHIPPED, RULES = "C1abETmS", "R1abETmS"
CONVS = {"dfl@v3": ("evctrl", "trk"), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp", "evctrl")}
ARMS = ("decoded", "shipped", "added", "rules only")


def lomo(train, test, keys, seed):
    """Scores of the test rows, each from models that never saw its match; deletion and insertion rows apart."""
    params = dict(ev.PARAMS, seed=seed, deterministic=True, force_row_wise=True, num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    p = np.full(len(test), np.nan)
    for typ in ("ins", "del"):
        tr = [r for r in train if r["typ"] == typ and r["y"] >= 0]
        idx = np.array([n for n, r in enumerate(test) if r["typ"] == typ])
        if not len(idx):
            continue
        X = np.array([[r[k] for k in keys] for r in tr], float)
        y = np.array([r["y"] for r in tr])
        g = np.array([r["group"] for r in tr])
        Xt = np.array([[test[n][k] for k in keys] for n in idx], float)
        gt = np.array([test[n]["group"] for n in idx])
        for grp in np.unique(gt):
            keep = g != grp
            m = lgb.train(params, lgb.Dataset(X[keep], y[keep]), ev.ROUNDS)
            te = gt == grp
            p[idx[te]] = m.predict(Xt[te])
    return p


def rows_of(name, seed, deads):
    exp_kpu.use(name)
    return {(src, d): ev._rows4(src, seed, d) for src in exp_kpu.SRCS for d in deads}


def evaluate(seeds=es.SEEDS, vseeds=(0, 1)):
    cells, acc = {}, {}
    keys = exp_kpu.feats(SHIPPED)
    learned_ref = {}
    for seed in seeds:
        shipped = rows_of(SHIPPED, seed, ev.DEADS)
        rules = rows_of(RULES, seed, (ev.DEADS[0],))             # the "learned D" channel of R1 holds the rules decode
        test = [r for src in exp_kpu.SRCS for r in rules[(src, ev.DEADS[0])]["rows"]]
        tr_ship = [r for v in shipped.values() for r in v["rows"]]
        tr_rule = test
        for q in vseeds:
            ps = {"shipped": lomo(tr_ship, test, keys, q), "added": lomo(tr_ship + tr_rule, test, keys, q), "rules only": lomo(tr_rule, test, keys, q)}
            n0 = 0
            for src in exp_kpu.SRCS:
                v = rules[(src, ev.DEADS[0])]
                idx = np.arange(n0, n0 + len(v["rows"]))
                n0 += len(v["rows"])
                blocks = bench.blocks(src)
                if (src, seed) not in learned_ref:
                    dl = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", src)
                    a_lrn = [exp_blend.logit_alive(x) for x in dl]
                    members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
                    outs = exp_kpu.outputs("C1", src)
                    raws = [0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(es.blend(es.v10_raw(seed, src), members[seed], 0.5), outs[ss.SEEDS[seed]])]
                    learned_ref[(src, seed)] = sd.decode_scores(a_lrn, exp_smooth.smooth(raws, 25))
                ref = learned_ref[(src, seed)]
                by = {}
                for n, r in enumerate(v["rows"]):
                    by.setdefault(r["i"], []).append(n)
                labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in CONVS[src]}
                ins = np.array([r["typ"] == "ins" for r in v["rows"]])
                for arm in ARMS:
                    if arm == "decoded":
                        preds = [np.asarray(b) for b in v["base"]]
                    else:
                        pp = ps[arm][idx]
                        preds = [ev.apply(v["base"][i], [v["rows"][n] for n in by.get(i, [])], pp[by.get(i, [])], True, True, 0.65, 0.35) for i in range(len(blocks))]
                        a = acc.setdefault((src, arm), [])
                        a.append(((pp[ins] >= 0.65).sum() / (len(blocks) / 2), (pp[~ins] < 0.35).sum() / (len(blocks) / 2), np.quantile(pp[ins], 0.9), np.quantile(pp[ins], 0.99)))
                    dt = [dead_timing.retime(np.asarray(x), np.asarray(y)) for x, y in zip(preds, ref)]
                    for c in CONVS[src]:
                        cells.setdefault((src, c, "rules", arm), []).append(np.mean(list(er.by_match(blocks, labs[c], preds).values())))
                        cells.setdefault((src, c, "rules + DT", arm), []).append(np.mean(list(er.by_match(blocks, labs[c], dt).values())))
                bench._blocks.clear()
                bench._labels.clear()
        print("  seed %d done" % seed, flush=True)
    exp_kpu.use("v14")
    print("\\n%-20s %-11s %s" % ("context", "dead ball", " ".join("%-16s" % a for a in ARMS)))
    for src, cs in CONVS.items():
        for c in cs:
            for d in ("rules", "rules + DT"):
                base = np.mean(cells[(src, c, d, "decoded")])
                print("%-20s %-11s %-16.4f %s" % (src.split("@")[0] + " " + c, d, base, " ".join("%-+16.4f" % (np.mean(cells[(src, c, d, a)]) - base) for a in ARMS[1:])), flush=True)
    print("\\naccepted per match: inserts, deletes, insert p 90% / 99%")
    for (src, arm), v in acc.items():
        print("  %-8s %-11s %.1f %.1f %.2f %.2f" % ((src.split("@")[0], arm) + tuple(np.mean(v, axis=0))), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if len(sys.argv) < 2 or sys.argv[1] != "eval":
        raise SystemExit(__doc__)
    evaluate(tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else es.SEEDS,
             tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1))
