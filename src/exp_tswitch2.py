"""Round 9: refinements of v12's touch switch relief (src/exp_tswitch.py: where the touch sequence changes team, a change
of possession into that team within +-6 frames costs lam = 8 less). Variants:
  confirmed  only gains followed by another touch of the same team within 3 s (the official "control" cue: the new
             team keeps / passes the ball), lam 8 / 12
  asym       window [-3, +9] / [-9, +3] around the gain touch instead of +-6
Decode-only on v10's smoothed probabilities; paired spell F1 against v12 (lam 8, +-6) and v10 on the eight V3
contexts; worst-case regret. AGGREGATE output only.

  python -I src/exp_tswitch2.py
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
import exp_tswitch  # noqa: E402
import touch  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
# name: (lam, lo, hi, confirmed)
VARIANTS = {"v12": (8.0, -6, 6, False), "conf l8": (8.0, -6, 6, True), "conf l12": (12.0, -6, 6, True),
            "asym -3+9": (8.0, -3, 9, False), "asym -9+3": (8.0, -9, 3, False), "conf l12 -3+9": (12.0, -3, 9, True)}


def relief(b, lo, hi, confirmed):
    L, W = b["pitch"]
    t, k = touch.touches(b["P"][:, :, :2] * np.array([L, W], np.float32),
                         b["B"][:, :2] * np.array([L, W], np.float32), np.asarray(b["team"]))
    n = len(b["P"])
    out = np.zeros((n, 2))
    if len(t) < 2:
        return out
    g = np.flatnonzero(k[1:] != k[:-1]) + 1
    if confirmed:
        nxt = np.r_[g + 1]
        ok = nxt < len(t)
        g = g[ok]
        nxt = nxt[ok]
        g = g[(k[nxt] == k[g]) & (t[nxt] - t[g] <= 75)]
    for col, team in ((0, 1), (1, 0)):
        acc = np.zeros(n + 1)
        sel = t[g[k[g] == team]]
        np.add.at(acc, np.clip(sel + lo, 0, n), 1)
        np.add.at(acc, np.clip(sel + hi + 1, 0, n), -1)
        out[:, col] = np.cumsum(acc[:-1]) > 0
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        blocks = bench.blocks(src)
        sp = exp_tswitch.spec(dec)
        rel = {n: [relief(b, lo, hi, cf) for b in blocks] for n, (_, lo, hi, cf) in VARIANTS.items()}
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            ems = [td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0) for pd_, ph in zip(pds, v10[src])]
            res = {"v10": e2.score(src, conv, [sp.decode(em, np.zeros((len(em), 2))).astype(np.int8) for em in ems])}
            for n, (lam, _, _, _) in VARIANTS.items():
                res[n] = e2.score(src, conv, [sp.decode(em, -lam * r).astype(np.int8) for em, r in zip(ems, rel[n])])
            rows[ctx] = {n: v.mean() for n, v in res.items()}
            ref = res["v12"]
            print("%-24s v10 %.4f v12 %.4f | %s" % (ctx, res["v10"].mean(), ref.mean(), " | ".join(
                "%s %+.4f (%d/%d)" % (n, v.mean() - ref.mean(), int((v > ref).sum()), len(v))
                for n, v in res.items() if n not in ("v12", "v10"))), flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in rows[next(iter(rows))]}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
