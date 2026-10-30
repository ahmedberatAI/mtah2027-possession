"""Shared CPU test bench for model ideas (docs/RAPOR.md). AGGREGATE output only.

Sources (each a list of half-match "blocks" with degraded inputs, cached under cache/bench/inputs):
  dfl      7 open DFL matches, sim.V2 degradation, seeds as train_dfl / simfix (1000 * mi + h)
  metrica  3 Metrica sample games x 3 seeds, sim.V2 degradation, seeds as simfix (7000 + 100g + 10s + h)
  barath   BAR-ATH graded halves (tactical players + merged ball), never printed
Possession conventions for the labels:
  dfl      native (DFL BallPossession), tau2 / tau3 (short spells merged, metrica_sample.merge_short),
           trk (controlled possession from clean tracking: a spell after the other team's spell counts
           only if the new team passes within itself or keeps the ball ~1 s, see trk_ctrl)
  metrica  events, tau2, ctrl (metrica_sample conventions; ctrl ~ the challenge's GT reading)
Feature versions: "base" = mlfeat.compute; "base+mod:func+..." adds the dicts returned by
func(P, B, team, gk, pitch) -> (F, T_home, T_away) from src/<mod>.py (new ideas live in their own files).

  python -I src/bench.py inputs                 # degraded inputs (once)
  python -I src/bench.py conv                   # label conventions: spell statistics
  python -I src/bench.py run <feat> <train_conv> [tag]   # team model LOMO + Metrica + BAR-ATH report
"""
import importlib
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import atd  # noqa: E402
import dfl  # noqa: E402
import metrica_sample as ms  # noqa: E402
import skillcorner as scr  # noqa: E402
import mlfeat  # noqa: E402
import sim  # noqa: E402
import train_dfl as td  # noqa: E402

pm = td.pm
H, A, D, X = 0, 1, 2, 3
FPS = 25
CACHE = os.path.join(ROOT, "cache", "bench")
INP = os.path.join(CACHE, "inputs")
REPORTS = os.path.join(ROOT, "reports", "bench")
M_SEEDS = (0, 1, 2)
THREADS = int(os.environ.get("BENCH_THREADS", "8"))
PROCS = int(os.environ.get("BENCH_PROCS", "7"))       # worker processes for inputs / features / labels
LGB_TEAM = dict(td.LGB_TEAM, num_threads=THREADS)
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "400"))
STRIDE = int(os.environ.get("BENCH_STRIDE", "2"))
PEN_GRID = (2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 96)
DECODERS = {"dflcv": dict(pen_dead=32.0, min_poss=5, min_dead=25),   # DFL-CV tuned
            "v5": dict(pen_dead=30.0, min_poss=8, min_dead=50)}      # model_v5's (inherited from v2)
EPS = 1e-4
VARIANTS = {"v2": sim.V2, "v3": sim.V3, "v4": sim.V4}   # degradations; "dfl" / "metrica" alone mean @v2
VARIANTS.update({"v3b%d" % k: dict(sim.V3, ball_every=k) for k in (5, 10, 15, 25, 50, 250)})   # ball every k frames
VARIANTS.update({"v5": sim.V5, "v5c": sim.V5C, "v3c": sim.V3C})   # ATD-like fabricated ball runs (+ cleaned)
VARIANTS["v3s2"] = dict(sim.V3, seed_offset=50000)   # a second, independently degraded copy (training augmentation)
# exact positions (round 10, src/exp_clean.py): V3's masks (same random stream) without position noise / aerial parallax
VARIANTS["v3p0"] = dict(sim.V3, ball_noise=0.0, p_noise=0.0, cam=(0.5, -0.45, 1e6))
LGB_DEAD = dict(td.LGB_DEAD, num_threads=THREADS)


def split_src(src):
    """'dfl@v3' -> ('dfl', 'v3'); 'dfl' -> ('dfl', 'v2')."""
    base, _, var = src.partition("@")
    return base, var or "v2"


def src_name(base, var):
    return base if var == "v2" or base == "barath" else "%s@%s" % (base, var)


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(path + ".part", path)


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_booster(path):
    """A shipped LightGBM model file, read as text: a Windows checkout with core.autocrlf=true writes CRLF line
    endings, which LightGBM 4.7 cannot parse from a path (the process aborts); read this way they load the same model."""
    with open(path, encoding="utf-8") as fh:
        return lgb.Booster(model_str=fh.read())


# -- degraded inputs ---------------------------------------------------------------------------------

def _dfl_job(job):
    mi, m, var = job
    out = os.path.join(INP, "%s_%s.pkl" % (src_name("dfl", var), m))
    if os.path.isfile(out):
        return out
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = dfl.parse(m)
    pitch = tuple(float(v) for v in d["pitch"])
    blocks = []
    for h in (1, 2):
        t = "h%d" % h
        Pn, Bn = sim.degrade(d[t + "_players"], d[t + "_ball"], d[t + "_labels"], pitch,
                             seed=1000 * mi + h + VARIANTS[var].get("seed_offset", 0), params=VARIANTS[var])
        blocks.append({"key": "%s/%s/h%d" % (src_name("dfl", var), m, h), "group": m, "P": Pn[:, :, :2].astype(np.float32),
                       "B": Bn[:, :2].astype(np.float32), "team": d[t + "_team"], "gk": d[t + "_gk"],
                       "pitch": pitch, "lab": d[t + "_labels"]})
    _dump(blocks, out)
    return out


def _metrica_job(job):
    g, s, var = job
    out = os.path.join(INP, "%s_g%d_s%d.pkl" % (src_name("metrica", var), g, s))
    if os.path.isfile(out):
        return out
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = ms.parse(g)
    pitch = tuple(float(v) for v in d["pitch"])
    blocks = []
    for h in (1, 2):
        t = "h%d" % h
        Pn, Bn = sim.degrade(d[t + "_players"], d[t + "_ball"], d[t + "_labels"], pitch,
                             seed=7000 + 100 * g + 10 * s + h + VARIANTS[var].get("seed_offset", 0),
                             params=VARIANTS[var])
        blocks.append({"key": "%s/g%d/s%d/h%d" % (src_name("metrica", var), g, s, h), "group": "g%d" % g, "seed": s,
                       "P": Pn[:, :, :2].astype(np.float32), "B": Bn[:, :2].astype(np.float32),
                       "team": d[t + "_team"], "gk": d[t + "_gk"], "pitch": pitch, "lab": d[t + "_labels"]})
    _dump(blocks, out)
    return out


def _sc_job(job):
    mi, m, var = job
    out = os.path.join(INP, "%s_%s.pkl" % (src_name("sc", var), m))
    if os.path.isfile(out):
        return out
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = scr.parse(m)
    pitch = tuple(float(v) for v in d["pitch"])
    blocks = []
    for h in (1, 2):
        t = "h%d" % h
        Pn, Bn = sim.degrade(d[t + "_players"], d[t + "_ball"], d[t + "_labels"], pitch,
                             seed=20000 + 10 * mi + h + VARIANTS[var].get("seed_offset", 0), params=VARIANTS[var])
        blocks.append({"key": "%s/%s/h%d" % (src_name("sc", var), m, h), "group": m,
                       "P": Pn[:, :, :2].astype(np.float32), "B": Bn[:, :2].astype(np.float32),
                       "team": d[t + "_team"], "gk": d[t + "_gk"], "pitch": pitch, "lab": d[t + "_labels"]})
    _dump(blocks, out)
    return out


def build_sc_inputs(var="v3", procs=None):
    from multiprocessing import Pool
    os.makedirs(INP, exist_ok=True)
    with Pool(procs or PROCS) as pool:
        pool.map(_sc_job, [(mi, m, var) for mi, m in enumerate(scr.matches())])


def _barath_inputs():
    out = os.path.join(INP, "barath.pkl")
    if os.path.isfile(out):
        return out
    d = atd.load("tactical")
    f = _load(os.path.join(ROOT, "cache", "features_merged.pkl"))
    gk = np.zeros(d["players"].shape[1], bool)
    gk[f["gk_slots"]] = True
    blocks = []
    for h, a, b in atd.HALVES:
        blocks.append({"key": "barath/h%d" % h, "group": "barath", "a": a, "b": b,
                       "P": d["players"][a:b + 1, :, :2].astype(np.float32),
                       "B": (f["bxy"][a:b + 1] / np.array(atd.PITCH, np.float32)).astype(np.float32),
                       "team": d["team"], "gk": gk, "pitch": tuple(atd.PITCH), "lab": None})
    _dump(blocks, out)
    return out


def build_inputs(var="v2", procs=None):
    from multiprocessing import Pool
    os.makedirs(INP, exist_ok=True)
    with Pool(procs or PROCS) as pool:
        r1 = pool.map_async(_dfl_job, [(mi, m, var) for mi, m in enumerate(dfl.MATCHES)])
        r2 = pool.map_async(_metrica_job, [(g, s, var) for g in ms.GAMES for s in M_SEEDS])
        r1.get(), r2.get()
    if os.path.isfile(atd.paths()["meta"]):      # BAR-ATH data is local only (never on remote VMs)
        _barath_inputs()
    print("inputs", var, "ready in", INP)


_blocks = {}


def blocks(src):
    if src not in _blocks:
        base, var = split_src(src)
        name = src_name(base, var)
        if base == "dfl":
            if not os.path.isfile(os.path.join(INP, "%s_%s.pkl" % (name, dfl.MATCHES[-1]))):
                build_inputs(var)
            _blocks[src] = [b for m in dfl.MATCHES for b in _load(os.path.join(INP, "%s_%s.pkl" % (name, m)))]
        elif base == "metrica":
            if not os.path.isfile(os.path.join(INP, "%s_g%d_s%d.pkl" % (name, ms.GAMES[-1], M_SEEDS[-1]))):
                build_inputs(var)
            _blocks[src] = [b for g in ms.GAMES for s in M_SEEDS
                            for b in _load(os.path.join(INP, "%s_g%d_s%d.pkl" % (name, g, s)))]
        elif base == "sc":
            ms_ = scr.matches()
            if not os.path.isfile(os.path.join(INP, "%s_%s.pkl" % (name, ms_[-1]))):
                build_sc_inputs(var)
            _blocks[src] = [b for m in ms_ for b in _load(os.path.join(INP, "%s_%s.pkl" % (name, m)))]
        elif base == "barath":
            _blocks[src] = _load(os.path.join(INP, "barath.pkl"))
        else:
            raise ValueError(src)
    return _blocks[src]


# -- possession conventions --------------------------------------------------------------------------

def trk_ctrl(lab, P, B, team, pitch, r=1.5, z_max=2.0, keep=25):
    """Controlled possession from clean tracking: a spell that directly follows the other team's spell
    (no D between) is a change of possession only if the new team shows control inside it -- two of
    its players on the ball (a completed pass / hand-over) or one of them on the ball for >= keep
    frames. Otherwise the spell goes back to the team that had the ball (cf. merge_uncontrolled)."""
    L, W = pitch
    pxy = P[:, :, :2] * np.array([L, W], np.float32)
    bxy = B[:, :2] * np.array([L, W], np.float32)
    z = np.nan_to_num(B[:, 3]) if B.shape[1] > 3 else np.zeros(len(B))
    dd = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    dd = np.where(np.isfinite(dd), dd, np.inf)
    j = np.argmin(dd, 1)
    near = dd[np.arange(len(dd)), j]
    owner = np.where((near <= r) & (z <= z_max), j, -1)
    owner_team = np.where(owner >= 0, team[np.maximum(owner, 0)], -1)
    out = lab.copy()
    sp = ms.spells(lab)
    for i in range(1, len(sp)):
        a, b, k = sp[i]
        pa, pb, _ = sp[i - 1]
        if pb != a or out[pa] not in (H, A) or out[pa] == k:
            continue
        mine = owner_team[a:b] == k
        if len(set(owner[a:b][mine].tolist())) >= 2 or mine.sum() >= keep:
            continue
        out[a:b] = out[pa]
    return out


def _dfl_conv_job(m):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = dfl.parse(m)
    pitch = tuple(float(v) for v in d["pitch"])
    res = []
    for h in (1, 2):
        t = "h%d" % h
        lab = d[t + "_labels"]
        res.append({"native": lab, "tau2": ms.merge_short(lab, 2), "tau3": ms.merge_short(lab, 3),
                    "trk": trk_ctrl(lab, d[t + "_players"], d[t + "_ball"], d[t + "_team"], pitch)})
    return m, res


_labels = {}


def labels(src, conv):
    """Per-block label arrays of src under conv (same order as blocks(src))."""
    src = split_src(src)[0]
    key = (src, conv)
    if key in _labels:
        return _labels[key]
    if src == "dfl" and conv in ("evctrl", "phase"):   # round 12: from the synchronised DFL events (src/dfl_events.py)
        import dfl_events as de
        fn = de.evctrl_labels if conv == "evctrl" else de.phase_labels
        out = []
        for m in dfl.MATCHES:
            d, ev = dfl.parse(m), de.synced(m)
            out += [fn(d["h%d_labels" % (h + 1)], ev[h]) for h in (0, 1)]
        _labels[key] = out
        return out
    if src == "dfl":
        path = os.path.join(CACHE, "dfl_conv.pkl")
        if not os.path.isfile(path):
            from multiprocessing import Pool
            with Pool(PROCS) as pool:
                _dump(dict(pool.map(_dfl_conv_job, dfl.MATCHES)), path)
        allc = _load(path)
        conv = "native" if conv == "events" else conv
        _labels[key] = [allc[m][h][conv] for m in dfl.MATCHES for h in (0, 1)]
    elif src == "metrica":
        conv = "events" if conv == "native" else conv
        per_game = {g: ms.gt_halves(g, conv) for g in ms.GAMES}
        _labels[key] = [per_game[g][h] for g in ms.GAMES for s in M_SEEDS for h in (0, 1)]
    elif src == "sc":
        if conv in ("phase", "native"):              # SkillCorner phases of play
            _labels[key] = [b["lab"] for b in blocks("sc@v3")]
        elif conv == "pp":                           # team from the dynamic events' player possessions
            _labels[key] = [scr.parse(m)["h%d_labels_pp" % h] for m in scr.matches() for h in (1, 2)]
        elif conv == "ctrl":                         # pp put into the controlled convention with trk_ctrl's rule
            out = []
            for m in scr.matches():
                d = scr.parse(m)
                for h in (1, 2):
                    t = "h%d" % h
                    out.append(trk_ctrl(d[t + "_labels_pp"], d[t + "_players"], d[t + "_ball"], d[t + "_team"],
                                        tuple(float(v) for v in d["pitch"])))
            _labels[key] = out
        else:
            raise ValueError(conv)
    else:
        raise ValueError(src)
    return _labels[key]


def conv_stats():
    rows = []
    for src, convs in (("dfl", ("native", "tau2", "tau3", "trk")), ("metrica", ("events", "tau2", "ctrl"))):
        bl = blocks(src)
        for c in convs:
            labs = labels(src, c)
            if src == "metrica":   # one copy per game, not per seed
                labs = [l for b, l in zip(bl, labs) if b["seed"] == 0]
            s = ms.spell_stats(labs)
            n_match = len(labs) // 2
            rows.append("%-8s %-7s spells/match %6.1f  per alive min %.2f  median %5.2f s  mean %5.2f s  <2s %.3f"
                        "  <1s %.3f  dead segs/match %.0f" % (src, c, s["n_spells"] / n_match,
                                                               s["n_spells"] / s["alive_min"], s["median_s"],
                                                               s["mean_s"], s["lt2s"], s["lt1s"],
                                                               s["n_dead_segs"] / n_match))
    print("\n".join(rows))
    return rows


# -- features ----------------------------------------------------------------------------------------

def _feat_fns(version):
    parts = version.split("+")
    assert parts[0] == "base", version
    fns = []
    for p in parts[1:]:
        mod, fn = p.split(":")
        fns.append(getattr(importlib.import_module(mod), fn))
    return fns


def _block_feats(job):
    version, b, out = job
    if os.path.isfile(out):
        return out
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    F, Th, Ta = mlfeat.compute(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
    for fn in _feat_fns(version):
        f2, th2, ta2 = fn(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        F.update(f2)
        Th.update(th2)
        Ta.update(ta2)
    fk, tk = sorted(F), sorted(Th)
    _dump({"XF": np.stack([F[k] for k in fk], 1).astype(np.float32),
           "XH": np.stack([Th[k] for k in tk], 1).astype(np.float32),
           "XA": np.stack([Ta[k] for k in tk], 1).astype(np.float32), "fk": fk, "tk": tk}, out)
    return out


def _safe(version):
    return version.replace(":", "-").replace("+", "__")


def features(version, src, procs=None):
    bl = blocks(src)
    outs = [os.path.join(CACHE, "feat", _safe(version), b["key"].replace("/", "_") + ".pkl") for b in bl]
    todo = [(version, b, o) for b, o in zip(bl, outs) if not os.path.isfile(o)]
    if todo:
        from multiprocessing import Pool
        with Pool(min(procs or PROCS, len(todo))) as pool:
            pool.map(_block_feats, todo)
    return [_load(o) for o in outs]


# -- models ------------------------------------------------------------------------------------------

def turnover_weight(lab, w, half=50):
    """1 everywhere, w within +-half frames of an in-play change of team (H <-> A, no D between)."""
    out = np.ones(len(lab), np.float32)
    t = np.flatnonzero((lab[1:] < 2) & (lab[:-1] < 2) & (lab[1:] != lab[:-1])) + 1
    if len(t):
        cover = np.zeros(len(lab) + 1, np.int32)
        np.add.at(cover, np.maximum(t - half, 0), 1)
        np.add.at(cover, np.minimum(t + half, len(lab)), -1)
        out[np.cumsum(cover[:-1]) > 0] = w
    return out


def team_rows(feats, labs, stride=STRIDE, turnover_w=None):
    XF, XH, XA, y, wt = [], [], [], [], []
    for f, lab in zip(feats, labs):
        sl = slice(0, None, stride)
        XF.append(f["XF"][sl]); XH.append(f["XH"][sl]); XA.append(f["XA"][sl]); y.append(lab[sl])
        wt.append(turnover_weight(lab, turnover_w)[sl] if turnover_w else np.ones(len(lab[sl]), np.float32))
    XF, XH, XA, y, wt = (np.concatenate(XF), np.concatenate(XH), np.concatenate(XA), np.concatenate(y),
                         np.concatenate(wt))
    alive = (y == H) | (y == A)
    Xt = np.concatenate([np.hstack([XF[alive], XH[alive]]), np.hstack([XF[alive], XA[alive]])])
    yt = np.concatenate([y[alive] == H, y[alive] == A]).astype(int)
    return Xt, yt, np.concatenate([wt[alive], wt[alive]])


def fit_team(feats, labs, params=None, rounds=ROUNDS):
    """params may carry "turnover_w" (sample weight near in-play team changes, idea N9); the rest goes to
    LightGBM."""
    params = dict(params or {})
    tw = params.pop("turnover_w", None)
    Xt, yt, wt = team_rows(feats, labs, turnover_w=tw)
    return lgb.train(dict(LGB_TEAM, **params), lgb.Dataset(Xt, yt, weight=wt if tw else None), rounds)


def predict_team(model, f):
    p1 = model.predict(np.hstack([f["XF"], f["XH"]]), num_threads=THREADS)
    p2 = 1 - model.predict(np.hstack([f["XF"], f["XA"]]), num_threads=THREADS)
    return 0.5 * (p1 + p2)


def fit_dead(feats, labs, rounds=ROUNDS, stride=STRIDE):
    XF = np.concatenate([f["XF"][::stride] for f in feats])
    y = np.concatenate([l[::stride] for l in labs])
    g = y < 3
    return lgb.train(LGB_DEAD, lgb.Dataset(XF[g], (y[g] == D).astype(int)), rounds)


def dead_models(version, var):
    """Dead-ball probabilities for (features, degradation): DFL out of fold, Metrica and BAR-ATH from the
    all-DFL model. V2 + base reuses the earlier sim.V2 models (cache/dfl_v2, cache/simfix/metrica_v2_v2)."""
    path = os.path.join(CACHE, "dead", "%s_%s.pkl" % (_safe(version), var))
    if os.path.isfile(path):
        return _load(path)
    sd, sm = src_name("dfl", var), src_name("metrica", var)
    if version == "base" and var == "v2":
        oof = _load(os.path.join(ROOT, "cache", "dfl_v2", "oof.pkl"))
        per = {s: _load(os.path.join(ROOT, "cache", "simfix", "metrica_v2_v2", "probs_seed%d.pkl" % s))
               for s in M_SEEDS}
        res = {"dfl": [oof[m][h][0] for m in dfl.MATCHES for h in (0, 1)],
               "metrica": [per[s][g][h][0] for g in ms.GAMES for s in M_SEEDS for h in (0, 1)]}
        model = _load(os.path.join(ROOT, "cache", "dfl_v2", "models_all.pkl"))[0]
    else:
        fd, fm = features(version, sd), features(version, sm)
        labs = labels("dfl", "native")
        bl = blocks(sd)
        oof = [None] * len(bl)
        for m in dfl.MATCHES:
            tr = [i for i, b in enumerate(bl) if b["group"] != m]
            mdl = fit_dead([fd[i] for i in tr], [labs[i] for i in tr])
            for i, b in enumerate(bl):
                if b["group"] == m:
                    oof[i] = mdl.predict(fd[i]["XF"], num_threads=THREADS)
        model = fit_dead(fd, labs)
        res = {"dfl": oof, "metrica": [model.predict(f["XF"], num_threads=THREADS) for f in fm]}
    res["barath"] = [model.predict(f["XF"], num_threads=THREADS) for f in features(version, "barath")]
    _dump(res, path)
    return res


def dead_ext(version, var, src):
    """Dead-ball probabilities of the all-DFL dead model (features `version`, sim `var`) on another source
    (e.g. "sc@v3"); the model is trained once and kept as cache/bench/dead/<version>_<var>_model.txt."""
    out = os.path.join(CACHE, "dead", "%s_%s_%s.pkl" % (_safe(version), var, src.replace("@", "-")))
    if os.path.isfile(out):
        return _load(out)
    mpath = os.path.join(CACHE, "dead", "%s_%s_model.txt" % (_safe(version), var))
    if os.path.isfile(mpath):
        model = lgb.Booster(model_file=mpath)
    else:
        model = fit_dead(features(version, src_name("dfl", var)), labels("dfl", "native"))
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        model.save_model(mpath)
    res = [model.predict(f["XF"], num_threads=THREADS) for f in features(version, src)]
    _dump(res, out)
    return res


def dead_probs(src, version="base"):
    base, var = split_src(src)
    return dead_models(version, var)[base]


def barath_learned_dead(version, var, ph_path, pens=(4, 8, 20)):
    """BAR-ATH decoded with the LEARNED dead-ball model (train_dfl.apply style: + broadcast validity / live
    cues, pre-kickoff D) and the given team probabilities; public window + full match, aggregates."""
    import evaluate as ev
    from make_submission import merged_features
    f = merged_features()
    pdb = dead_models(version, var)["barath"]
    probs = _load(ph_path)
    lines = []
    for pt in pens:
        labels_ = np.full(f["n"], 3, np.int8)
        for b, pd_ in zip(blocks("barath"), pdb):
            a, e = b["a"], b["b"]
            S = td.emissions(pd_, probs[a][1], 1.0, 1.0)
            have = f["b_have"][a:e + 1]
            ab = np.zeros(e - a + 1)
            ab[have & ~f["b_valid"][a:e + 1]] -= 1.5
            ab[have & f["b_live"][a:e + 1]] += 0.4
            S[:, 0] += ab / 2
            S[:, 1] += ab / 2
            S[:, 2] -= ab / 2
            if a == atd.HALVES[1][1]:
                S[:atd.KICKOFF_2ND - a, 2] += 1e6
            labels_[a:e + 1] = td.decode_block(S, pt, 32, 5, 25)
        sc, dg = ev.score(labels_), ev.diagnostics(labels_)
        lines.append("  BAR-ATH learned-dead pen_team %2d | public spellF1 %.4f macroF1 %.4f adF1 %.4f team_acc %.4f "
                     "n_pred %d/%d | full n_poss %d dead share %.3f D segs %d" % (
                         pt, sc["spellF1"], sc["macroF1"], sc["alive_dead_f1"], sc["team_acc"], sc["n_pred"],
                         sc["n_gt"], dg["n_poss"], dg["dead_share"], dg["n_dead_segs"]))
    gt = ev.public_gt()
    pdf = np.full(f["n"], np.nan)
    for b, pd_ in zip(blocks("barath"), pdb):
        pdf[b["a"]:b["b"] + 1] = pd_
    m = (gt < 3) & np.isfinite(pdf)
    alive_gt, alive_pred = gt[m] < 2, pdf[m] < 0.5
    f1 = 2 * np.sum(alive_gt & alive_pred) / max(1, alive_gt.sum() + alive_pred.sum())
    lines.append("  BAR-ATH learned-dead raw (p<0.5 alive) public alive F1 %.4f, predicted dead share full %.3f" % (
        f1, float(np.mean(np.concatenate(pdb) > 0.5))))
    return lines


# -- scoring -----------------------------------------------------------------------------------------

def decode(pd_, ph, pen_team, dec, temp_t=1.0):
    S = td.emissions(pd_, ph, 1.0, temp_t)
    return td.decode_block(S, pen_team, dec["pen_dead"], dec["min_poss"], dec["min_dead"]).astype(np.int8)


def spell_by_group(bl, gts, preds):
    """Official spell metric per group (match / game x seed), halves concatenated with X between."""
    groups = {}
    for b, g, p in zip(bl, gts, preds):
        k = (b["group"], b.get("seed", 0))
        groups.setdefault(k, ([], []))
        groups[k][0].append(g)
        groups[k][1].append(p)
    return {k: td.spell_score(v[0], v[1]) for k, v in groups.items()}


def curve(src, pdl, phl, conv, dec="dflcv", pens=PEN_GRID, temp_t=1.0, preds_fn=None):
    """spellF1 (mean over groups) per pen_team; preds_fn(pd_, ph, pen_team, block) overrides decoding."""
    bl = blocks(src)
    gts = labels(src, conv)
    out = {}
    for pt in pens:
        if preds_fn is None:
            preds = [decode(pd_, ph, pt, DECODERS[dec], temp_t) for pd_, ph in zip(pdl, phl)]
        else:
            preds = [preds_fn(pd_, ph, pt, b) for pd_, ph, b in zip(pdl, phl, bl)]
        r = spell_by_group(bl, gts, preds)
        arr = np.array(list(r.values()))
        out[pt] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()), int(arr[:, 2].sum()), int(arr[:, 3].sum()))
    return out


def team_acc(phl, labs):
    acc = []
    for ph, lab in zip(phl, labs):
        al = (lab == H) | (lab == A)
        acc.append(np.mean((ph[al] > 0.5) == (lab[al] == H)))
    return float(np.mean(acc))


def fmt_curve(c, keys=(4, 8, 12, 20, 32)):
    best = max(c, key=lambda k: c[k][0])
    return "best %2d %.4f | " % (best, c[best][0]) + " ".join("%d:%.4f" % (k, c[k][0]) for k in keys if k in c)


# -- BAR-ATH -----------------------------------------------------------------------------------------

def barath_report(model, version, tag, pens=(8, 12, 20)):
    import evaluate as ev
    import model_v5
    import proxy
    from make_submission import merged_features
    feats = features(version, "barath")
    probs = {}
    for b, f in zip(blocks("barath"), feats):
        probs[b["a"]] = (None, predict_team(model, f))
    path = os.path.join(CACHE, "barath_probs_%s.pkl" % tag)
    _dump(probs, path)
    f = merged_features()
    lines = []
    for pt in pens:
        model_v5._probs.clear()
        lh, _ = model_v5.team_logp(f, path)
        lab = model_v5.predict(f, {"pen_team": float(pt)})
        model_v5._probs.clear()
        sc, dg, px = ev.score(lab), ev.diagnostics(lab), proxy.proxies(lab, f)
        g = lab[atd.graded_mask(len(lab))]
        alive_min = np.sum(g < 2) / FPS / 60
        lines.append("  BAR-ATH v5 pen_team %2d | public spellF1 %.4f macroF1 %.4f team_acc %.4f n_pred %d/%d | full "
                     "n_poss %d (%.2f/alive min) median %.1f s | tag_team %.4f setpiece %.3f" % (
                         pt, sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["n_pred"], sc["n_gt"], dg["n_poss"],
                         dg["n_poss"] / alive_min, dg["poss_median_s"], px["tag_team_agree"], px["setpiece_ok"]))
    return lines, path


# -- one idea, end to end ----------------------------------------------------------------------------

def run(version="base", train_conv="native", tag=None, params=None, rounds=ROUNDS, barath=True,
        eval_dfl=("native", "trk"), eval_metrica=("ctrl", "events"), var="v2", learned_dead=None):
    """Team model on `version` features trained on `train_conv` labels: DFL leave-one-match-out OOF,
    all-DFL model on Metrica and BAR-ATH. Prints and returns aggregate scores."""
    tag = tag or "%s_%s%s" % (_safe(version), train_conv, "" if var == "v2" else "_" + var)
    learned_dead = (var != "v2") if learned_dead is None else learned_dead
    t0 = time.time()
    sd, sm = src_name("dfl", var), src_name("metrica", var)
    fd = features(version, sd)
    fm = features(version, sm)
    labs = labels("dfl", train_conv)
    bl = blocks(sd)
    oof_path = os.path.join(CACHE, "oof", tag + ".pkl")
    if os.path.isfile(oof_path):
        ph_oof = _load(oof_path)
    else:
        ph_oof = [None] * len(bl)
        for m in dfl.MATCHES:
            tr = [i for i, b in enumerate(bl) if b["group"] != m]
            te = [i for i, b in enumerate(bl) if b["group"] == m]
            model = fit_team([fd[i] for i in tr], [labs[i] for i in tr], params, rounds)
            for i in te:
                ph_oof[i] = predict_team(model, fd[i])
        _dump(ph_oof, oof_path)
    all_path = os.path.join(CACHE, "models", tag + ".txt")
    if os.path.isfile(all_path):
        model = lgb.Booster(model_file=all_path)
    else:
        model = fit_team(fd, labs, params, rounds)
        os.makedirs(os.path.dirname(all_path), exist_ok=True)
        model.save_model(all_path)
    ph_m = [predict_team(model, f) for f in fm]
    pd_d, pd_m = dead_probs(sd, version), dead_probs(sm, version)
    lines = ["== %s  (features %s, sim %s, trained on DFL %s labels, %d rounds, %.0f s)" % (
        tag, version, var, train_conv, rounds, time.time() - t0)]
    res = {"tag": tag, "version": version, "train_conv": train_conv, "dfl": {}, "metrica": {}}
    for c in eval_dfl:
        cv_ = curve(sd, pd_d, ph_oof, c, "dflcv")
        acc = team_acc(ph_oof, labels("dfl", c))
        res["dfl"][c] = {"curve": cv_, "team_acc": acc}
        lines.append("  DFL-CV  vs %-6s team acc %.4f | spellF1 %s" % (c, acc, fmt_curve(cv_)))
    for c in eval_metrica:
        for dec in ("v5", "dflcv"):
            cm = curve(sm, pd_m, ph_m, c, dec)
            res["metrica"]["%s|%s" % (c, dec)] = {"curve": cm}
            lines.append("  Metrica vs %-6s dec %-5s | spellF1 %s" % (c, dec, fmt_curve(cm)))
        res["metrica"][c + "|team_acc"] = team_acc(ph_m, labels("metrica", c))
    lines.append("  Metrica team acc: " + " ".join("%s %.4f" % (c, res["metrica"][c + "|team_acc"])
                                                 for c in eval_metrica))
    if barath:
        bl_lines, ph_path = barath_report(model, version, tag)
        lines += bl_lines
        if learned_dead:
            lines += barath_learned_dead(version, var, ph_path)
    print("\n".join(lines), flush=True)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, tag + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORTS, tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "inputs":
        build_inputs(args[0] if args else "v2")
    elif cmd == "conv":
        conv_stats()
    elif cmd == "prep":
        # prep <features> <var>: inputs, features and dead-ball models, so parallel runs can share them
        var = args[1] if len(args) > 1 else "v2"
        for src in (src_name("dfl", var), src_name("metrica", var), "barath"):
            features(args[0], src)
        dead_models(args[0], var)
        print("prepared", args[0], var)
    elif cmd == "run":
        # run <features> <train_conv> [var] [tag]
        run(args[0], args[1], args[3] if len(args) > 3 else None, var=args[2] if len(args) > 2 else "v2")
