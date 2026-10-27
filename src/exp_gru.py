"""Idea N25 (docs/RAPOR.md): BiGRU team model (src/seq_gru.py) vs v7's LightGBM team model, and their
logit-average ensemble. Meant for a GPU VM (colab/run_gru.sh); everything is rebuilt there from open data so
both models see identical inputs. Dead-ball evidence = TRUE D on every set (the GRU only changes the team side),
decoder = v7's B2 soft minimum. DFL: leave-one-match-out for both models; Metrica-ctrl and SkillCorner-phase:
all-DFL models. AGGREGATE output only; probabilities and the all-DFL GRU are saved under cache/gru.

  python src/exp_gru.py
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import seq_gru  # noqa: E402
import train_dfl as td  # noqa: E402

V = model_v7.VERSION
OUT = os.path.join(bench.CACHE, "gru")
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))
PENS = (8, 12, 20)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ens(a, b):
    return [1 / (1 + np.exp(-(lg(x) + lg(y)) / 2)) for x, y in zip(a, b)]


def scores(src, conv, dec, phl):
    bl, gts = bench.blocks(src), bench.labels(src, conv)
    pdl = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in gts]
    out = {}
    for pt in PENS:
        sp = decode2.duration_spec(pt, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
        r = bench.spell_by_group(bl, gts, [sp.decode(td.emissions(pd_, ph, 1.0, 1.0)).astype(np.int8)
                                           for pd_, ph in zip(pdl, phl)])
        out[pt] = np.array([r[k][0] for k in sorted(r)])
    return out


def team_acc(phl, labs):
    return float(np.mean([np.mean((ph[l < 2] > 0.5) == (l[l < 2] == 0)) for ph, l in zip(phl, labs)]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    feats = {s: bench.features(V, s) for s, _, _ in SETS}
    print("features ready %.0f s; device %s" % (time.time() - t0, "cuda" if torch.cuda.is_available() else "cpu"),
          flush=True)
    fd, ld, bl = feats["dfl@v3"], bench.labels("dfl@v3", "native"), bench.blocks("dfl@v3")
    probs = {"gru": {}, "lgbm": {}}
    path = os.path.join(OUT, "probs.pkl")
    if os.path.isfile(path):
        probs = bench._load(path)
    else:
        for kind in ("gru", "lgbm"):
            oof = [None] * len(bl)
            for m in dfl_matches(bl):
                tr = [i for i, b in enumerate(bl) if b["group"] != m]
                t = time.time()
                if kind == "gru":
                    mdl = seq_gru.fit([fd[i] for i in tr], [ld[i] for i in tr])
                    pred = lambda f: seq_gru.predict(mdl, f)
                else:
                    mdl = bench.fit_team([fd[i] for i in tr], [ld[i] for i in tr],
                                         params={"num_threads": os.cpu_count()})
                    pred = lambda f: bench.predict_team(mdl, f)
                for i, b in enumerate(bl):
                    if b["group"] == m:
                        oof[i] = pred(fd[i]).astype(np.float32)
                print("  %s fold %s %.0f s" % (kind, m, time.time() - t), flush=True)
            probs[kind]["dfl@v3"] = oof
            if kind == "gru":
                full = seq_gru.fit(fd, ld)
                torch.save({"state": full[0].state_dict(), "mu": full[1].mu, "sd": full[1].sd},
                           os.path.join(OUT, "gru_all_dfl.pt"))
                pred = lambda f: seq_gru.predict(full, f)
            else:
                full = bench.fit_team(fd, ld, params={"num_threads": os.cpu_count()})
                full.save_model(os.path.join(OUT, "lgbm_all_dfl.txt"))
                pred = lambda f: bench.predict_team(full, f)
            for s in ("metrica@v3", "sc@v3"):
                probs[kind][s] = [pred(f).astype(np.float32) for f in feats[s]]
        bench._dump(probs, path)
    probs["ens"] = {s: ens(probs["gru"][s], probs["lgbm"][s]) for s in probs["gru"]}
    for s, conv, dec in SETS:
        labs = bench.labels(s, conv)
        res = {k: scores(s, conv, dec, probs[k][s]) for k in ("lgbm", "gru", "ens")}
        ref = res["lgbm"][12]
        print("\n%s vs %s (true D, B2 decoder): team acc lgbm %.4f gru %.4f ens %.4f" % (
            s, conv, team_acc(probs["lgbm"][s], labs), team_acc(probs["gru"][s], labs), team_acc(probs["ens"][s], labs)))
        for k in ("lgbm", "gru", "ens"):
            cells = []
            for pt in PENS:
                d = res[k][pt] - ref
                cells.append("@%d %.4f (%+.4f, %d/%d)" % (pt, res[k][pt].mean(), d.mean(), int((d > 0).sum()), len(d)))
            print("  %-5s %s" % (k, " | ".join(cells)), flush=True)
    print("\ntotal %.0f s" % (time.time() - t0))


def dfl_matches(bl):
    out = []
    for b in bl:
        if b["group"] not in out:
            out.append(b["group"])
    return out


if __name__ == "__main__":
    main()
