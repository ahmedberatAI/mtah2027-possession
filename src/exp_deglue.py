"""Round 45 (docs/RAPOR.md, pre-registered): the rest of the Metrica blind spot. Metrica's sample ball sits on its owner, which flatters ball-based parts; round 37 scored only the specialist on
the de-glued input (bench v3g). v10 and the shared BiGRU are still scored on the glued ball, and v10 is still trained on it. Here:

  v10 raw on v3g     today's models (trained on glued Metrica), Metrica input de-glued
  BiGRU on v3g       the shared checkpoints on the de-glued input (local GPU)
  v10g               v10's recipe with its Metrica rows from v3g too: DFL held out, Metrica held out by game, SkillCorner from the all-data model

  python -I src/exp_deglue.py features          features and ball-free extras of metrica@v3g
  python -I src/exp_deglue.py bigru             the BiGRU members on metrica@v3g
  python -I src/exp_deglue.py v10 SEED          v10 on v3g and v10g, one LightGBM seed
  python -I src/exp_deglue.py eval              v19's dead ball; Metrica de-glued everywhere: today's blend, v10g, blend weights
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import model_v7  # noqa: E402
import short_specialist as ss  # noqa: E402

OUT = os.path.join(bench.CACHE, "deglue")
G = "metrica@v3g"
W_M = 2.0                   # v10's weight per Metrica game (the _wm2 runs)
SEED_OF = {0: 0, 1: 1, 2: 2}


def features():
    import exp_bfree
    bench.features(model_v7.VERSION, G)
    exp_bfree.extra(G)
    print("features ready", flush=True)


def bigru():
    import json
    from pathlib import Path
    import torch
    import seq_pff as seq
    path = os.path.join(OUT, "bigru_members_v3g.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder = Path(bench.ROOT) / "models" / "bigru_dfl_pff_v1"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    fs = bench.features(model_v7.VERSION, G, procs=2)
    members = []
    for item in manifest["models"]:
        ck = torch.load(folder / item["file"], map_location="cpu", weights_only=False)
        c = ck["config"]
        net = seq.Net(345, c["kind"], c["width"], c["drop"]).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        sc = seq.Scaler(ck["mean"], ck["scale"])
        members.append([seq.predict_raw(net, sc, seq.views(ff), len(ff["XF"])).astype(np.float32) for ff in fs])
        del net
        print("predicted", item["file"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(dict(members=members), path)
    return dict(members=members)


def _aug(src):
    import exp_bfree
    import exp_ctrl2 as e2
    ex = exp_bfree.extra(src)
    out = []
    for b, (h, a) in zip(bench.blocks(src), ex):
        f = bench._load(e2._feat_path(b))
        out.append(dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
    return out


def v10(seed):
    """seed: LGB_SEED of exp_a2. Writes raw probabilities {src: per half} for v10g and today's models on the v3g input."""
    import lightgbm as lgb
    seed = int(seed)
    path = os.path.join(OUT, "v10g_seed%d.pkl" % seed)
    if os.path.isfile(path):
        return
    params = dict(bench.LGB_TEAM, num_threads=bench.THREADS, seed=seed)

    def rows(feats, labs, w):
        Xt, yt, _ = bench.team_rows(feats, labs)
        return Xt, yt, np.full(len(yt), w, np.float32)

    def fit(parts):
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        w = np.concatenate([p[2] for p in parts])
        return lgb.train(params, lgb.Dataset(X, y, weight=w), bench.ROUNDS)

    fd, fm, fg = _aug("dfl@v3"), _aug("metrica@v3"), _aug(G)
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bg = bench.blocks("dfl@v3"), bench.blocks(G)
    res = {"dfl@v3": [None] * len(bd), G: [None] * len(bg)}
    today = [None] * len(bg)
    g_all = rows(fg, lm, W_M / 3.0)
    m_all = rows(fm, lm, W_M / 3.0)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = fit([rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), g_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  seed %d DFL fold %s done" % (seed, m), flush=True)
    d_all = rows(fd, ld, 1.0)
    for g in sorted({b["group"] for b in bg}):
        tr = [i for i, b in enumerate(bg) if b["group"] != g]
        mdl = fit([d_all, rows([fg[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        old = fit([d_all, rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)])
        for i, b in enumerate(bg):
            if b["group"] == g:
                res[G][i] = bench.predict_team(mdl, fg[i])
                today[i] = bench.predict_team(old, fg[i])
        print("  seed %d Metrica fold %s done" % (seed, g), flush=True)
    mdl = fit([d_all, g_all])
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in _aug("sc@v3")]
    os.makedirs(OUT, exist_ok=True)
    mdl.save_model(os.path.join(OUT, "v10g_all_seed%d.txt" % seed))
    bench._dump(dict(v10g=res, today=today), path)
    print("  seed %d done" % seed, flush=True)


def team_raws(src, seed, v10_kind, w_gru=0.5, w_spec=0.25):
    """Raw team probability per half of one v10 seed; Metrica's parts on the de-glued input where given."""
    k = seed
    if src.startswith("metrica"):
        members = bench._load(os.path.join(OUT, "bigru_members_v3g.pkl"))["members"]
    else:
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
    d = bench._load(os.path.join(OUT, "v10g_seed%d.pkl" % SEED_OF[k]))
    if v10_kind == "v10g":
        v = d["v10g"][G if src.startswith("metrica") else src]
    else:
        v = d["today"] if src.startswith("metrica") else es.v10_raw(k, src)
    import exp_kpu
    outs = exp_kpu.outputs("J1", src)[ss.SEEDS[k]]
    return [(1 - w_spec) * ((1 - w_gru) * np.asarray(a, np.float64) + w_gru * np.asarray(b, np.float64)) + w_spec * ss.pk(o, 0) for a, b, o in zip(v, members[k], outs)]


def decode_par(a_list, phl, pen_team=12.0, pen_dead=30.0, min_poss=8, min_dead=50):
    """sd.decode_scores on a thread pool (the decoder releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    import decode2
    import exp_v7
    sp = decode2.duration_spec(pen_team, pen_dead, min_poss, min_dead, exp_v7.ramp(175, 48.0))

    def one(i):
        q = np.clip(phl[i], sd.EPS, 1 - sd.EPS)
        return sp.decode(np.stack([a_list[i] / 2 + np.log(q), a_list[i] / 2 + np.log(1 - q), -a_list[i] / 2], 1)).astype(np.int8)
    with ThreadPoolExecutor(int(os.environ.get("DECODE_WORKERS", "6"))) as pool:
        return list(pool.map(one, range(len(a_list))))


def evaluate():
    import exp_blend_weight as bw
    import exp_deadmix as dm
    import exp_rules_open as ro
    names = ("today", "v10g", "today g.25", "today g.75", "v10g g.25", "v10g g.75", "today s.5", "v10g s.5", "v10g s.15")
    cfg = {"today": ("today", 0.5, 0.25), "v10g": ("v10g", 0.5, 0.25), "today g.25": ("today", 0.25, 0.25), "today g.75": ("today", 0.75, 0.25), "v10g g.25": ("v10g", 0.25, 0.25), "v10g g.75": ("v10g", 0.75, 0.25),
           "today s.5": ("today", 0.5, 0.5), "v10g s.5": ("v10g", 0.5, 0.5), "v10g s.15": ("v10g", 0.5, 0.15)}
    cells = {}
    for src, convs in dm.CONVS.items():
        fsrc = G if src.startswith("metrica") else src
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        a = ro.a25_evidence(src)
        for k in es.SEEDS:
            for name, (kind, wg, ws) in cfg.items():
                smo = exp_smooth.smooth(team_raws(src, k, kind, wg, ws), 25)
                for dname, ev2 in (("v19 D", a), ("true D", [np.where(g == 2, -12.0, 12.0) for g in labs[convs[0]]])):
                    preds = decode_par(ev2, smo)
                    for c in convs:
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (src, k), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="today")
    for cand in names[1:]:
        bw.decide(mean, "today", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"features": features, "bigru": bigru, "v10": v10, "eval": evaluate}[sys.argv[1]](*sys.argv[2:])
