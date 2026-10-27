"""Round 9: decoder-side touch use on v10 -- the switch-cost relief at touch-team changes (src/exp_tswitch.py) and the
anchor fusion (src/exp_tanchor.py), alone and together, their sensitivity (window W, relief lam, anchor beta), and the
same on test inputs the models never saw (sim V5: fabricated ball runs; V4: pre-smoothed tracks; v10 probabilities of
src/exp_stress.py, touches re-detected on those inputs). Decode-only; paired spell F1 against plain v10 per context;
worst-case regret over the eight V3 contexts. AGGREGATE output only.

  python -I src/exp_tcombo.py
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
import exp_fab  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import exp_stress  # noqa: E402
import exp_tanchor  # noqa: E402
import exp_tswitch  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
# name: (switch window W or None, lam, anchor beta)
VARIANTS = {"s8 W6": (6, 8.0, 0.0), "s8 W3": (3, 8.0, 0.0), "s8 W12": (12, 8.0, 0.0), "s6 W6": (6, 6.0, 0.0),
            "s10 W6": (6, 10.0, 0.0), "a0.5": (None, 0.0, 0.5), "s8 W6 + a0.25": (6, 8.0, 0.25),
            "s8 W6 + a0.5": (6, 8.0, 0.5)}
STRESS = ("s8 W6", "a0.5", "s8 W6 + a0.25")


def chans(blocks, w):
    old = exp_tswitch.W
    exp_tswitch.W = w
    try:
        return [exp_tswitch.gains(b) for b in blocks]
    finally:
        exp_tswitch.W = old


def run(src, conv, dec, phs, pds, names):
    blocks = bench.blocks(src)
    sp = exp_tswitch.spec(dec)
    anc = [exp_tanchor.anchors(b) for b in blocks]
    ch = {w: chans(blocks, w) for w in {VARIANTS[n][0] for n in names if VARIANTS[n][0]}}
    ref = e2.score(src, conv, e2.decode_all(dec, pds, phs))
    out = {}
    for n in names:
        w, lam, beta = VARIANTS[n]
        preds = []
        for i, (pd_, ph) in enumerate(zip(pds, phs)):
            p = e2.sig(e2.logit(ph) + beta * anc[i]) if beta else ph
            em = td.emissions(pd_, np.clip(p, EPS, 1 - EPS), 1.0, 1.0)
            c = -lam * ch[w][i] if w else np.zeros((len(em), 2))
            preds.append(sp.decode(em, c).astype(np.int8))
        out[n] = e2.score(src, conv, preds)
    return ref, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            ref, out = run(src, conv, dec, v10[src], pds, list(VARIANTS))
            rows[ctx] = dict({"v10": ref.mean()}, **{n: v.mean() for n, v in out.items()})
            print("V3 %-24s v10 %.4f | %s" % (ctx, ref.mean(), " | ".join(
                "%s %+.4f (%d/%d)" % (n, v.mean() - ref.mean(), int((v > ref).sum()), len(v)) for n, v in out.items())),
                  flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in rows[next(iter(rows))]}
    print("worst-case regret (V3): " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])),
          flush=True)
    for var in ("v5", "v4"):
        dead = exp_fab.cross_probs("v3", var)
        probs = {k: exp_smooth.smooth(p, 25) for k, p in exp_stress.v10_on(var).items()}
        for base, conv, dec in exp_stress.SETS:
            src = "%s@%s" % (base, var)
            labs = bench.labels(src, conv)
            for dname, pds in (("learned D", dead[base][1]),
                               ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
                ref, out = run(src, conv, dec, probs[base], pds, STRESS)
                print("%s %-8s %-6s %-9s v10 %.4f | %s" % (var.upper(), base, conv, dname, ref.mean(), " | ".join(
                    "%s %+.4f (%d/%d)" % (n, v.mean() - ref.mean(), int((v > ref).sum()), len(v))
                    for n, v in out.items())), flush=True)


if __name__ == "__main__":
    main()
