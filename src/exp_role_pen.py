"""Round 7c follow-up: does the role model's frame gain turn into spellF1 once the team-switch penalty is re-chosen?
Same protocol as v7's pen_team decision (docs/RAPOR.md, "Açık sorular"): spellF1 of v7 and of the role model
(src/exp_role.py, cached probabilities) for pen_team in PENS with the v7 decoder otherwise unchanged, on DFL-trk
(learned D), Metrica-ctrl (learned / true D) and SkillCorner phase and player-possession labels (learned / true D);
worst-case regret = max over sets of (best of all (model, pen) on that set - this one). AGGREGATE output only.

  python -I src/exp_role_pen.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_role  # noqa: E402
import exp_v7  # noqa: E402
import train_dfl as td  # noqa: E402

PENS = (8, 12, 16, 20, 24, 32, 48)
EPS = 1e-4
DEC = {"dfl@v3": (32.0, 5, 25), "metrica@v3": (30.0, 8, 50), "sc@v3": (30.0, 8, 50)}
CONVS = {"dfl@v3": ("trk",), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp")}


def decode_all(pen, dec, pds, phs):
    sp = decode2.duration_spec(float(pen), dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
    return [sp.decode(td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0)).astype(np.int8)
            for pd_, ph in zip(pds, phs)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    res = {}          # (set name) -> {(model, pen): mean spellF1}
    for src in ("dfl@v3", "metrica@v3", "sc@v3"):
        phl, pdl = exp_oracle.probs(src)
        phr = exp_role.feat_probs(src, None)
        bl = bench.blocks(src)
        dvars = [("learned D", pdl)]
        if src != "dfl@v3":
            dvars.append(("true D", [np.where(g == 2, 1 - EPS, EPS) for g in bench.labels(src, CONVS[src][0])]))
        for dname, pds in dvars:
            for model, phs in (("v7", phl), ("roles", phr)):
                for pen in PENS:
                    preds = decode_all(pen, DEC[src], pds, phs)
                    for conv in CONVS[src]:
                        r = bench.spell_by_group(bl, bench.labels(src, conv), preds)
                        res.setdefault("%s %s %s" % (src.split("@")[0], conv, dname), {})[model, pen] = float(
                            np.mean([r[k][0] for k in sorted(r)]))
            print("  %s %s done" % (src, dname), flush=True)
    sets = sorted(res)
    print("%-26s %s" % ("set", " ".join("%s%-3d" % ("v" if m == "v7" else "r", p) for m in ("v7", "roles")
                                        for p in PENS)))
    for s in sets:
        print("%-26s %s" % (s, " ".join("%.3f" % res[s][m, p] for m in ("v7", "roles") for p in PENS)))
    for group, keep in (("all sets", sets), ("GT-like protocol sets", [s for s in sets if "pp" not in s])):
        regret = {(m, p): max(max(res[s].values()) - res[s][m, p] for s in keep) for m in ("v7", "roles")
                  for p in PENS}
        best = sorted(regret.items(), key=lambda kv: kv[1])[:6]
        print("worst-case regret over %s: %s" % (group, " | ".join("%s pen %d %.4f" % (m, p, r) for (m, p), r in best)))
        print("  v7 pen 12 regret %.4f" % regret["v7", 12])


if __name__ == "__main__":
    main()
