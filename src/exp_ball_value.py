"""How much would a better in-play ball track help? Paired DFL CV with in-play ball missing at
~0% (a0), half (ah) and the BAR-ATH-calibrated rate (V2, 6.5%). Same seeds, same dead-ball rates.
AGGREGATE output (DFL data is open)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import dfl, sim, simfix, train_dfl as td
sys.stdout.reconfigure(encoding="utf-8")
VAR = {"a0": dict(sim.V2, v2_alive_missing=1e-3, restart_missing=1e-3),
       "ah": dict(sim.V2, v2_alive_missing=0.0325, restart_missing=0.125)}
for v, params in VAR.items():
    d_out = os.path.join(td.ROOT, "cache", "dfl_" + v); os.makedirs(d_out, exist_ok=True)
    for mi, m in enumerate(dfl.MATCHES):
        out = os.path.join(d_out, m + "_feat.pkl")
        if os.path.isfile(out): continue
        d = dfl.parse(m); pitch = tuple(d["pitch"]); blocks = []
        for h in (1, 2):
            t = "h%d" % h
            Pn, Bn = sim.degrade(d[t+"_players"], d[t+"_ball"], d[t+"_labels"], pitch, seed=1000 * mi + h, params=params)
            XF, XH, XA, fk, tk = td.block_features(Pn, Bn, d[t+"_team"], d[t+"_gk"], pitch)
            blocks.append({"XF": XF, "XH": XH, "XA": XA, "lab": d[t+"_labels"], "seen": np.isfinite(Bn[:, 0])})
        simfix._dump({"blocks": blocks, "fk": fk, "tk": tk}, out)
    print("built", v, flush=True)
res = {}
for v in ("v2", "a0", "ah"):
    data = simfix.load_data(v)
    oof = {}
    for m in dfl.MATCHES:
        models = td.fit(*td.stack(data, [x for x in dfl.MATCHES if x != m]))
        oof[m] = [td.predict_proba(models, b["XF"], b["XH"], b["XA"]) for b in data[m]["blocks"]]
    fm = simfix.frame_metrics(data, oof)
    alive_seen = np.mean([np.mean(b["seen"][(b["lab"] < 2)]) for m in dfl.MATCHES for b in data[m]["blocks"]])
    fixed = simfix.spell_eval(data, oof, dict(simfix.FIXED_DEC, pen_team=20))
    best = simfix.tune_quiet(data, oof)
    res[v] = dict(alive_seen=alive_seen, team_acc=fm["team_acc"], alive_dead=fm["alive_dead_acc"],
                  spell_pen20=fixed["spellF1"], spell_best=best[1]["spellF1"], best=best[0])
    print("%-3s in-play ball seen %.3f | team acc %.4f | alive/dead acc %.4f | spellF1 pen20 %.4f | tuned %.4f %s" % (
        v, alive_seen, fm["team_acc"], fm["alive_dead_acc"], fixed["spellF1"], best[1]["spellF1"], best[0]), flush=True)
json.dump(res, open(os.path.join(td.ROOT, "reports", "exp_ball_value.json"), "w"), indent=1, default=float)
