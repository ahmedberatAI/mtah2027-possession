"""Round 9: a destination-aware B2 ramp. B2's soft minimum charges a short possession the same exit cost whether it
ends in a change of team or in a stoppage; v10's misses on Metrica-ctrl / DFL-trk with the true D are mostly short
true spells next to a stoppage, and the largest group ends in one ("before D": Metrica 86 of 168, DFL 102 of 219;
src/exp_errbins.py) -- a team wins the ball and play stops soon after. Here exits into D pay f x the ramp (f = 1 is
v7 / v10's B2; round 6 found exempting the spells AFTER a restart a loss). Decode-only on v10's smoothed probabilities;
paired spell F1 against v10 on the eight contexts; worst-case regret. AGGREGATE output only.

  python -I src/exp_b2d.py
"""
import os
import sys
import warnings

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_a2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_v7  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
FS = (1.0, 0.5, 0.25, 0.0)


@njit(cache=True)
def _run2(emis, emis_idx, offs, L, min_exit, exit_cost, exit_cost_d, to_d, adv, pen, init, back):
    """decode2._run without channels, with a second exit-cost array used when the next macro is a D macro."""
    n = emis.shape[0]
    M = L.shape[0]
    S = offs[M]
    NEG = -1e300
    delta = np.full(S, NEG)
    new = np.full(S, NEG)
    for s in range(S):
        if init[s] < 1e290:
            delta[s] = emis[0, emis_idx[s]] - init[s]
    ex = np.empty(M)
    exarg = np.zeros(M, np.int64)
    exd = np.empty(M)
    exdarg = np.zeros(M, np.int64)
    for t in range(1, n):
        for m in range(M):
            best = NEG
            arg = offs[m] + L[m] - 1
            bestd = NEG
            argd = arg
            for j in range(min_exit[m], L[m]):
                s = offs[m] + j
                v = delta[s] - exit_cost[s]
                if v > best:
                    best = v
                    arg = s
                v = delta[s] - exit_cost_d[s]
                if v > bestd:
                    bestd = v
                    argd = s
            ex[m] = best
            exarg[m] = arg
            exd[m] = bestd
            exdarg[m] = argd
        for m in range(M):
            s0 = offs[m]
            best = NEG
            arg = s0
            for m2 in range(M):
                if m2 == m:
                    continue
                c = pen[m2, m]
                if c > 1e290:
                    continue
                if to_d[m]:
                    v = exd[m2] - c
                    a2 = exdarg[m2]
                else:
                    v = ex[m2] - c
                    a2 = exarg[m2]
                if v > best:
                    best = v
                    arg = a2
            if L[m] == 1:
                v = delta[s0] - adv[s0]
                if v > best:
                    best = v
                    arg = s0
            new[s0] = best + emis[t, emis_idx[s0]]
            back[t, s0] = arg
            for j in range(1, L[m]):
                s = s0 + j
                a_ = delta[s - 1] - adv[s - 1]
                if j == L[m] - 1:
                    b_ = delta[s] - adv[s]
                    if b_ >= a_:
                        new[s] = b_ + emis[t, emis_idx[s]]
                        back[t, s] = s
                        continue
                new[s] = a_ + emis[t, emis_idx[s]]
                back[t, s] = s - 1
        for s in range(S):
            delta[s] = new[s]
    best = NEG
    arg = 0
    for s in range(S):
        if delta[s] > best:
            best = delta[s]
            arg = s
    path = np.empty(n, np.int64)
    s = arg
    for t in range(n - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


class SpecD:
    """B2 with exits into D charged f x the possession ramp."""

    def __init__(self, dec, f, d0=175, c=48.0):
        self.sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(d0, c))
        self.exit_d = self.sp.exit_cost * f
        self.to_d = np.array([False, False, True])

    def decode(self, emis):
        sp = self.sp
        emis = np.ascontiguousarray(emis, np.float64)
        n = len(emis)
        S = int(sp.offs[-1])
        back = np.empty((n, S), np.int16 if S < 32000 else np.int32)
        path = _run2(emis, sp.emis_idx, sp.offs, sp.L, sp.min_exit, sp.exit_cost, self.exit_d, self.to_d, sp.adv,
                     sp.pen, sp.init, back)
        return sp.out[path]


def decode_all(spec, pds, phs):
    return [spec.decode(td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0)).astype(np.int8)
            for pd_, ph in zip(pds, phs)]


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
            res = {f: e2.score(src, conv, decode_all(SpecD(dec, f), pds, v10[src])) for f in FS}
            ref0 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            assert abs(ref0.mean() - res[1.0].mean()) < 1e-9, "f = 1 must reproduce v10"
            rows[ctx] = {f: v.mean() for f, v in res.items()}
            ref = res[1.0]
            print("%-24s v10 %.4f | %s" % (ctx, ref.mean(), " | ".join(
                "f %g %+.4f (%d/%d)" % (f, v.mean() - ref.mean(), int((v > ref).sum()), len(v))
                for f, v in res.items() if f != 1.0)), flush=True)
    regret = {f: max(max(r.values()) - r[f] for r in rows.values()) for f in FS}
    print("worst-case regret: " + " | ".join("f %g %.4f" % kv for kv in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
