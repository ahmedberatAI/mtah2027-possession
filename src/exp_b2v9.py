"""Round 9: re-check the B2 soft-minimum parameters (D0, c; v7: 175, 48) for v9's smoothed team probabilities
(cache/bench/combo/probs_a2bf_w25.pkl) at pen 12, worst-case regret over DFL-trk (learned D), Metrica-ctrl and
SkillCorner phase / player-possession (learned / true D). AGGREGATE output only.

  python -I src/exp_b2v9.py
  B2_PROBS=probs_v10_w25.pkl B2_GRID=wide python -I src/exp_b2v9.py     (v10, a 4 x 5 grid incl. c = 0)
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_role_pen as rp  # noqa: E402
import exp_v7  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
GRID = [(d0, c) for d0 in (125, 175, 250) for c in (24.0, 48.0, 72.0)]
if os.environ.get("B2_GRID") == "wide":
    GRID = [(d0, c) for d0 in (75, 125, 175, 250) for c in (0.0, 24.0, 48.0, 72.0, 96.0)]
PROBS = os.environ.get("B2_PROBS", "probs_a2bf_w25.pkl")


def decode_all(d0, c, dec, pds, phs):
    sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(d0, c))
    return [sp.decode(td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0)).astype(np.int8)
            for pd_, ph in zip(pds, phs)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    alt = bench._load(os.path.join(bench.CACHE, "combo", PROBS))
    import exp_oracle
    res = {}
    for src in ("dfl@v3", "metrica@v3", "sc@v3"):
        _, pdl = exp_oracle.probs(src)
        bl = bench.blocks(src)
        dvars = [("learned D", pdl)]
        if src != "dfl@v3":
            dvars.append(("true D", [np.where(g == 2, 1 - EPS, EPS) for g in bench.labels(src, rp.CONVS[src][0])]))
        for dname, pds in dvars:
            for d0, c in GRID:
                preds = decode_all(d0, c, rp.DEC[src], pds, alt[src])
                for conv in rp.CONVS[src]:
                    r = bench.spell_by_group(bl, bench.labels(src, conv), preds)
                    res.setdefault("%s %s %s" % (src.split("@")[0], conv, dname), {})[d0, c] = float(
                        np.mean([r[k][0] for k in sorted(r)]))
            print("  %s %s done" % (src, dname), flush=True)
    sets = sorted(res)
    print("%-26s %s" % ("set", " ".join("%d/%-3d" % (d0, c) for d0, c in GRID)))
    for s in sets:
        print("%-26s %s" % (s, " ".join("%.4f " % res[s][k] for k in GRID)))
    regret = {k: max(max(res[s].values()) - res[s][k] for s in sets) for k in GRID}
    print("worst-case regret: " + " | ".join("D0 %d c %g %.4f" % (k[0], k[1], r)
                                             for k, r in sorted(regret.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
