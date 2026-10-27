"""Round 9: touches as a time-varying switch cost instead of evidence. Where the touch sequence changes team (a touch by
team k right after one by the other team: an interception / a won duel / a reception of a loose ball) a change of
possession INTO team k within +-W frames costs lam less than pen_team (12); everything else is v10 / v11's decoder.
Unlike the anchor fusion (src/exp_tanchor.py) this adds no evidence for k: it only lets a short spell start where a
turnover is plausible. Decode-only on the cached smoothed probabilities; paired spell F1 against the model without it
on the eight contexts; worst-case regret. AGGREGATE output only.

  python -I src/exp_tswitch.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402
import exp_v7  # noqa: E402
import touch  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
W = 6
LAMS = (4.0, 8.0, 12.0)
MODELS = {"v10": os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"),
          "v11": os.path.join(bench.CACHE, "touch", "probs.pkl")}


def spec(dec):
    """v7's B2 spec with one channel per direction: 0 = entering A from H, 1 = entering H from A."""
    ramp = exp_v7.ramp(175, 48.0)
    macros = [dict(emis=0, out=0, L=len(ramp), min_exit=dec[1] - 1, exit_cost=ramp),
              dict(emis=1, out=1, L=len(ramp), min_exit=dec[1] - 1, exit_cost=ramp),
              dict(emis=2, out=2, L=dec[2])]
    pen = [[0, 12.0, dec[0]], [12.0, 0, dec[0]], [dec[0], dec[0], 0]]
    return decode2.Spec(macros, pen, [[-1, 0, -1], [1, -1, -1], [-1, -1, -1]])


def gains(b):
    """chan (n, 2): 1 within +-W frames of a touch that changes the touch team (column 0: into A, 1: into H)."""
    L, Wd = b["pitch"]
    t, k = touch.touches(b["P"][:, :, :2] * np.array([L, Wd], np.float32),
                         b["B"][:, :2] * np.array([L, Wd], np.float32), np.asarray(b["team"]))
    n = len(b["P"])
    out = np.zeros((n, 2))
    if len(t) > 1:
        g = np.flatnonzero(k[1:] != k[:-1]) + 1
        for col, team in ((0, 1), (1, 0)):
            acc = np.zeros(n + 1)
            sel = t[g[k[g] == team]]
            np.add.at(acc, np.clip(sel - W, 0, n), 1)
            np.add.at(acc, np.clip(sel + W + 1, 0, n), -1)
            out[:, col] = np.cumsum(acc[:-1]) > 0
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    raw = {k: bench._load(p) for k, p in MODELS.items()}
    rows = {}
    for src, conv, dec in exp_a2.SETS:
        _, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        chans = [gains(b) for b in bench.blocks(src)]
        sp = spec(dec)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ctx = "%s %s %s" % (src.split("@")[0], conv, dname)
            rows[ctx] = {}
            cells = []
            for name, r in raw.items():
                phs = exp_smooth.smooth(r[src], 25)
                ems = [td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0) for pd_, ph in zip(pds, phs)]
                ref = e2.score(src, conv, [sp.decode(em, np.zeros((len(em), 2))).astype(np.int8) for em in ems])
                if name == "v10":
                    chk = e2.score(src, conv, e2.decode_all(dec, pds, phs))
                    assert abs(chk.mean() - ref.mean()) < 1e-9, "lam = 0 must reproduce v10's decoder"
                rows[ctx][name] = ref.mean()
                for lam in LAMS:
                    v = e2.score(src, conv, [sp.decode(em, -lam * ch).astype(np.int8) for em, ch in zip(ems, chans)])
                    rows[ctx]["%s+s%g" % (name, lam)] = v.mean()
                    cells.append("%s lam %g %+.4f (%d/%d)" % (name, lam, v.mean() - ref.mean(), int((v > ref).sum()),
                                                             len(v)))
            print("%-24s v10 %.4f v11 %.4f | %s" % (ctx, rows[ctx]["v10"], rows[ctx]["v11"], " | ".join(cells)),
                  flush=True)
    regret = {n: max(max(r.values()) - r[n] for r in rows.values()) for n in rows[next(iter(rows))]}
    print("worst-case regret: " + " | ".join("%s %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
