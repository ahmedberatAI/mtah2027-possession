"""Round 9: tracking touches as decoder anchors. Round 8's anchor oracle (src/exp_anchor.py) found that team-attributed
PASS anchors of video-pilot quality (recall 0.6, team accuracy 0.94) could add +0.007..+0.020 to v7 through a capped
fusion; the touches of src/touch.py come from the tracking alone (team = possession team at the touch on 0.93 of
alive Metrica touches). Same fusion on the smoothed team logit of v10 / v11 (touch features already inside v11):
z = logit(p_s) + beta * clip(sum_a s_a 1[t_a - W0 <= t <= t_a + W1], -1, 1), s = +1 home / -1 away, then the v7
decoder. Decode-only; paired spell F1 against the model without anchors on the eight contexts; worst-case regret.
AGGREGATE output only.

  python -I src/exp_tanchor.py
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
import touch  # noqa: E402

EPS = 1e-4
W0, W1 = 12, 3
BETAS = (0.5, 1.0, 2.0)
MODELS = {"v10": os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"),
          "v11": os.path.join(bench.CACHE, "touch", "probs.pkl")}


def anchors(b):
    L, W = b["pitch"]
    t, k = touch.touches(b["P"][:, :, :2] * np.array([L, W], np.float32),
                         b["B"][:, :2] * np.array([L, W], np.float32), np.asarray(b["team"]))
    n = len(b["P"])
    acc = np.zeros(n + 1)
    s = np.where(k == 0, 1.0, -1.0)
    np.add.at(acc, np.clip(t - W0, 0, n), s)
    np.add.at(acc, np.clip(t + W1 + 1, 0, n), -s)
    return np.clip(np.cumsum(acc[:-1]), -1, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = {k: bench._load(p) for k, p in MODELS.items()}
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        ev = [anchors(b) for b in bench.blocks(src)]
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            rows[ctx] = {}
            cells = []
            for name, r in raw.items():
                phs = exp_smooth.smooth(r[src], 25)
                ref = e2.score(src, conv, e2.decode_all(dec, pds, phs))
                rows[ctx][name] = ref.mean()
                for beta in BETAS:
                    v = e2.score(src, conv, e2.decode_all(dec, pds, [e2.sig(e2.logit(p) + beta * a)
                                                                    for p, a in zip(phs, ev)]))
                    rows[ctx]["%s+a%g" % (name, beta)] = v.mean()
                    cells.append("%s b%g %+.4f (%d/%d)" % (name, beta, v.mean() - ref.mean(), int((v > ref).sum()),
                                                          len(v)))
            print("%-24s v10 %.4f v11 %.4f | %s" % (ctx, rows[ctx]["v10"], rows[ctx]["v11"], " | ".join(cells)),
                  flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in rows[next(iter(rows))]}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
