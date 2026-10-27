"""Round 9: ensembles of the night's v10-regime team models (all out of fold, raw probabilities cached): v10 itself,
extra-trees v10, soft-target v10 (+-12), self-trained v10, v10 + controller features, Metrica keep-2 labels. Logit
mean of the members, then v10's +-25-frame smoothing and the v7 decoder; paired spell F1 against v10 on the eight
contexts and worst-case regret (seed ensembles of v9 were a wash; these members differ in objective / inputs).
Decode-only. AGGREGATE output only.

  python -I src/exp_ens.py                (ENS_SET=touch: v10 / v11 mixtures and two-seed averages)
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

EPS = 1e-4
C = bench.CACHE
MEMBERS = {"v10": os.path.join(C, "combo", "probs_a2_bfree_wm2.pkl"),
           "ET": os.path.join(C, "v10var", "probs_soft0_keep1_extra_treesTrue.pkl"),
           "soft12": os.path.join(C, "v10var", "probs_soft12_keep1_lgb.pkl"),
           "self": os.path.join(C, "selftrain", "probs_a1_c0.9.pkl"),
           "ctrl": os.path.join(C, "ctrlv10", "probs.pkl"),
           "keep2": os.path.join(C, "v10var", "probs_soft0_keep2_lgb.pkl"),
           "v10s1": os.path.join(C, "combo", "probs_a2_bfree_seed1_wm2.pkl"),
           "touch": os.path.join(C, "touch", "probs.pkl"),
           "touchs1": os.path.join(C, "touch", "probs_seed1.pkl")}
ENSEMBLES = (("v10",), ("v10", "ET"), ("v10", "soft12"), ("v10", "self"), ("v10", "ctrl"), ("v10", "ET", "soft12"),
             ("v10", "ET", "soft12", "self"), ("v10", "ET", "soft12", "self", "ctrl", "keep2"))
if os.environ.get("ENS_SET") == "touch":   # v10 / v11 (touch) mixtures and two-seed averages
    ENSEMBLES = (("v10",), ("touch",), ("v10", "touch"), ("v10", "v10s1"), ("touch", "touchs1"),
                 ("v10", "v10s1", "touch", "touchs1"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = {k: bench._load(p) for k, p in MEMBERS.items() if os.path.isfile(p)}
    ens = [e for e in ENSEMBLES if all(m in raw for m in e)]
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        phs = {"+".join(e): exp_smooth.smooth([e2.sig(np.mean([e2.logit(raw[m][src][i]) for m in e], 0))
                                               for i in range(len(labs))], 25) for e in ens}
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            res = {k: e2.score(src, conv, e2.decode_all(dec, pds, ph)) for k, ph in phs.items()}
            rows[ctx] = {k: v.mean() for k, v in res.items()}
            ref = res["v10"]
            print("%-24s v10 %.4f | %s" % (ctx, ref.mean(), " | ".join(
                "%s %+.4f (%d/%d)" % (k, v.mean() - ref.mean(), int((v > ref).sum()), len(v))
                for k, v in res.items() if k != "v10")), flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in rows[next(iter(rows))]}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
