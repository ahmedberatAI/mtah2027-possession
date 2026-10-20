"""Ball-dropout fix (sim.ball_dropout_v2) end to end: rebuild the degraded DFL training data, rerun the
DFL CV, retrain the team / dead models, and redo the BAR-ATH and Metrica checks. Writes only to
cache/dfl_<variant>, cache/simfix and reports/simfix; cache/dfl, cache/metrica and model_v5.py are
read, never written.

  .venv/Scripts/python.exe -I src/simfix.py build v2 v2t     # degraded DFL features -> cache/dfl_<v>
  .venv/Scripts/python.exe -I src/simfix.py cv v2             # train_dfl.cv() on them; keeps the fold models
  .venv/Scripts/python.exe -I src/simfix.py folds v1          # fold models on the v1 features (no writes to cache/dfl)
  .venv/Scripts/python.exe -I src/simfix.py cross             # every fold-model set on every test degradation
  .venv/Scripts/python.exe -I src/simfix.py models v2         # all-DFL models -> cache/dfl_<v>/models_all.pkl
  .venv/Scripts/python.exe -I src/simfix.py barath            # v5 with each model set (aggregates only)
  .venv/Scripts/python.exe -I src/simfix.py metrica_build     # degraded Metrica (v1 / v2) x model sets
  .venv/Scripts/python.exe -I src/simfix.py metrica_sweep v2 v2   # pen_team sweep, model set x Metrica degradation

Variants: v1 = sim.DEFAULT (the current cache/dfl), v2 = sim.V2 (BAR-ATH merged-ball rates),
v2t = sim.V2_TACTICAL (tactical-ball rates = the v1 targets 12% / 70%, without the spill).
Seeds are the ones train_dfl.build / metrica_calibration.build use, and v2 keeps v1's random stream for
the players, so v1 and v2 data differ only in the ball mask.
"""
import itertools
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
import dfl  # noqa: E402
import sim  # noqa: E402
import train_dfl as td  # noqa: E402

VARIANTS = {"v1": None, "v2": sim.V2, "v2t": sim.V2_TACTICAL}
SIMFIX = os.path.join(ROOT, "cache", "simfix")
REPORTS = os.path.join(ROOT, "reports", "simfix")
FIXED_DEC = dict(temp_d=1.0, temp_t=1.0, pen_team=4, pen_dead=32, min_poss=5, min_dead=25)  # DFL-CV best (v1)
EPS = 1e-4


def cache_dir(v):
    return os.path.join(ROOT, "cache", "dfl") if v == "v1" else os.path.join(ROOT, "cache", "dfl_" + v)


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(path + ".part", path)


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# -- degraded DFL features ---------------------------------------------------------------------------

def _build_match(job):
    v, mi, m = job
    out = os.path.join(cache_dir(v), "{}_feat.pkl".format(m))
    if os.path.isfile(out):
        return v, m, 0.0, None
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t = time.time()
    d = dfl.parse(m)
    pitch = tuple(d["pitch"])
    blocks = []
    for h in (1, 2):
        tag = "h%d" % h
        P, B, lab = d[tag + "_players"], d[tag + "_ball"], d[tag + "_labels"]
        Pn, Bn = sim.degrade(P, B, lab, pitch, seed=1000 * mi + h, params=VARIANTS[v])
        XF, XH, XA, fk, tk = td.block_features(Pn, Bn, d[tag + "_team"], d[tag + "_gk"], pitch)
        blocks.append({"XF": XF, "XH": XH, "XA": XA, "lab": lab, "seen": np.isfinite(Bn[:, 0])})
    _dump({"blocks": blocks, "fk": fk, "tk": tk}, out)
    return v, m, time.time() - t, float(np.mean(np.concatenate([b["seen"] for b in blocks])))


def build(variants, procs=7):
    from multiprocessing import Pool
    jobs = [(v, mi, m) for v in variants if v != "v1" for mi, m in enumerate(dfl.MATCHES)]
    for v in variants:
        os.makedirs(cache_dir(v), exist_ok=True)
    with Pool(min(procs, len(jobs))) as pool:
        for v, m, sec, seen in pool.imap_unordered(_build_match, jobs):
            print("built", v, m, "in %.0fs" % sec, "" if seen is None else "ball seen %.1f%%" % (100 * seen),
                  flush=True)


def load_data(v):
    old = td.CACHE
    td.CACHE = cache_dir(v)
    try:
        return td.load_all()
    finally:
        td.CACHE = old


# -- CV -----------------------------------------------------------------------------------------------

def cv(v):
    """train_dfl.cv() on cache/dfl_<v> (writes oof.pkl / best_decoder.json there) + the fold models."""
    assert v != "v1", "cache/dfl stays untouched; use `folds v1`"
    td.CACHE = cache_dir(v)
    folds = []
    fit = td.fit

    def fit_keep(*a):
        folds.append(fit(*a))
        return folds[-1]
    td.fit = fit_keep
    try:
        td.cv()
    finally:
        td.fit = fit
    _dump(dict(zip(dfl.MATCHES, folds)), os.path.join(SIMFIX, "fold_models_%s.pkl" % v))


def folds(v):
    """Fold models (leave one match out) without train_dfl.cv's writes, e.g. on the v1 features."""
    data = load_data(v)
    out = {}
    for m in dfl.MATCHES:
        t = time.time()
        out[m] = td.fit(*td.stack(data, [x for x in dfl.MATCHES if x != m]))
        print("fold", v, m, "%.0fs" % (time.time() - t), flush=True)
    _dump(out, os.path.join(SIMFIX, "fold_models_%s.pkl" % v))


def team_quality(ph, y, seen):
    """Team probability on alive frames: accuracy, log-loss, Brier, confidence (median |logit|, share
    max p > 0.9), mean confidence minus accuracy; overall and for ball seen / missing."""
    out = {}
    alive = (y == 0) | (y == 1)
    for name, sel in (("all", alive), ("seen", alive & seen), ("miss", alive & ~seen)):
        if not sel.any():
            continue
        p = np.clip(ph[sel], EPS, 1 - EPS)
        home = y[sel] == 0
        pt = np.where(home, p, 1 - p)
        conf = np.maximum(p, 1 - p)
        acc = np.mean((p > 0.5) == home)
        out[name] = {"n": int(sel.sum()), "acc": float(acc), "logloss": float(-np.mean(np.log(pt))),
                     "brier": float(np.mean((p - home) ** 2)),
                     "med_abs_logit": float(np.median(np.abs(np.log(p / (1 - p))))),
                     "share_p90": float(np.mean(conf > 0.9)), "conf_minus_acc": float(conf.mean() - acc)}
    return out


def frame_metrics(data, oof):
    """Mean over matches of train_dfl.cv's per-match frame metrics (themselves means over halves) and
    of the pooled team-probability quality."""
    accs, tq = [], []
    for m in dfl.MATCHES:
        a, ys, phs, seens = [], [], [], []
        for b, (pd_, ph) in zip(data[m]["blocks"], oof[m]):
            y = b["lab"]
            g = y < 3
            al = (y == 0) | (y == 1)
            pred = np.where(pd_ > 0.5, 2, np.where(ph > 0.5, 0, 1))
            a.append((np.mean(pred[g] == y[g]), np.mean((pd_[g] > 0.5) == (y[g] == 2)),
                      np.mean((ph > 0.5)[al] == (y[al] == 0))))
            ys.append(y); phs.append(ph); seens.append(b["seen"])
        accs.append(np.mean(a, 0))
        tq.append(team_quality(np.concatenate(phs), np.concatenate(ys), np.concatenate(seens)))
    accs = np.array(accs)
    team = {k: {s: float(np.mean([t[k][s] for t in tq])) for s in tq[0][k]} for k in tq[0]}
    return {"frame_acc": float(accs[:, 0].mean()), "alive_dead_acc": float(accs[:, 1].mean()),
            "team_acc": float(accs[:, 2].mean()), "per_match": accs.tolist(), "team": team}


def spell_eval(data, oof, dec):
    res = []
    for m in dfl.MATCHES:
        gts, preds = [], []
        for b, (pd_, ph) in zip(data[m]["blocks"], oof[m]):
            S = td.emissions(pd_, ph, dec["temp_d"], dec["temp_t"])
            preds.append(td.decode_block(S, dec["pen_team"], dec["pen_dead"], dec["min_poss"], dec["min_dead"]))
            gts.append(b["lab"])
        res.append(td.spell_score(gts, preds))
    res = np.array(res)
    return {"spellF1": float(res[:, 0].mean()), "spellF1_min": float(res[:, 0].min()),
            "macroF1": float(res[:, 1].mean()), "n_gt": int(res[:, 2].sum()), "n_pred": int(res[:, 3].sum()),
            "per_match": res[:, 0].tolist()}


def tune_quiet(data, oof):
    """train_dfl.tune's grid without printing / writing; best config and its scores."""
    best = None
    for temp_d, temp_t, pt, pdd, mp, md in itertools.product([1.0], [0.5, 1.0], [4, 8, 16, 32],
                                                             [8, 16, 32, 64], [5], [25]):
        dec = dict(temp_d=temp_d, temp_t=temp_t, pen_team=pt, pen_dead=pdd, min_poss=mp, min_dead=md)
        r = spell_eval(data, oof, dec)
        if best is None or r["spellF1"] > best[1]["spellF1"]:
            best = (dec, r)
    return best


def _cross_job(job):
    """One (fold-model set R, test degradation T) pair: OOF probabilities + metrics."""
    R, T = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out_path = os.path.join(SIMFIX, "cross", "oof_%s_on_%s.pkl" % (R, T))
    data = load_data(T)
    if os.path.isfile(out_path):
        oof = _load(out_path)
    else:
        fm = _load(os.path.join(SIMFIX, "fold_models_%s.pkl" % R))
        oof = {m: [td.predict_proba(fm[m], b["XF"], b["XH"], b["XA"]) for b in data[m]["blocks"]]
               for m in dfl.MATCHES}
        _dump(oof, out_path)
    best_dec, best = tune_quiet(data, oof)
    return R, T, {"frame": frame_metrics(data, oof), "fixed": spell_eval(data, oof, FIXED_DEC),
                  "best_dec": best_dec, "best": best}


def cross(models=("v1", "v2", "v2t"), tests=("v1", "v2", "v2t"), procs=5):
    from multiprocessing import Pool
    os.makedirs(os.path.join(SIMFIX, "cross"), exist_ok=True)
    jobs = [(R, T) for R in models for T in tests
            if os.path.isfile(os.path.join(SIMFIX, "fold_models_%s.pkl" % R))
            and os.path.isfile(os.path.join(cache_dir(T), "{}_feat.pkl".format(dfl.MATCHES[-1])))]
    res = {}
    with Pool(min(procs, len(jobs))) as pool:
        for R, T, r in pool.imap_unordered(_cross_job, jobs):
            res["%s|%s" % (R, T)] = r
            print("cross models %-3s on test %-3s done" % (R, T), flush=True)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "dfl_cross.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print_cross(res)


def print_cross(res=None):
    if res is None:
        with open(os.path.join(REPORTS, "dfl_cross.json"), encoding="utf-8") as fh:
            res = json.load(fh)
    lines = ["| fold models (train sim) | test sim | frame acc | alive/dead acc | team acc | team acc, ball seen | "
             "team acc, ball missing | team log-loss | conf - acc | median abs logit | spellF1 @4/32 | "
             "best spellF1 | best decoder (temp_t, pen_team, pen_dead) | n_pred / n_gt @best |",
             "|---|---|" + "---|" * 12]
    for key in sorted(res):
        R, T = key.split("|")
        r = res[key]
        f, t = r["frame"], r["frame"]["team"]
        d = r["best_dec"]
        lines.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %+.4f | %.2f | %.4f | %.4f | %.1f, %d, %d | %d / %d |" % (
            R, T, f["frame_acc"], f["alive_dead_acc"], f["team_acc"], t["seen"]["acc"],
            t["miss"]["acc"] if "miss" in t else float("nan"), t["all"]["logloss"], t["all"]["conf_minus_acc"],
            t["all"]["med_abs_logit"], r["fixed"]["spellF1"], r["best"]["spellF1"], d["temp_t"], d["pen_team"],
            d["pen_dead"], r["best"]["n_pred"], r["best"]["n_gt"]))
    print("\n".join(lines))
    return lines


# -- all-DFL models and BAR-ATH (aggregates only) --------------------------------------------------

def models(v):
    assert v != "v1", "cache/dfl/models_all.pkl is not replaced here"
    path = os.path.join(cache_dir(v), "models_all.pkl")
    if not os.path.isfile(path):
        t = time.time()
        _dump(td.fit(*td.stack(load_data(v), dfl.MATCHES)), path)
        print("models", v, "%.0fs" % (time.time() - t))
    return path


def model_path(v):
    return os.path.join(cache_dir(v), "models_all.pkl")


def barath(variants=("v1", "v2", "v2t"), pens=(8, 12, 20)):
    """model_v5 with the team probabilities of each model set: public-window score, full-match
    diagnostics and proxies, and the team model's confidence (aggregates only)."""
    import evaluate as ev
    import model_v5
    import proxy
    from make_submission import merged_features
    f = merged_features()
    _, blocks = td.barath_blocks()
    gt = ev.public_gt()
    g_alive = gt < 2
    lines = []
    out = {}
    for v in variants:
        mods = _load(model_path(v))
        probs = {blk["a"]: td.predict_proba(mods, blk["XF"], blk["XH"], blk["XA"]) for blk in blocks}
        if v == "v1":
            ref = _load(os.path.join(ROOT, "cache", "dfl", "barath_probs.pkl"))
            same = all(np.array_equal(probs[a][i], ref[a][i]) for a in ref for i in (0, 1))
            lines.append("v1 probabilities identical to cache/dfl/barath_probs.pkl: %s" % same)
            path = os.path.join(SIMFIX, "barath_probs_v1.pkl")
        else:
            path = os.path.join(cache_dir(v), "barath_probs.pkl")
        _dump(probs, path)
        model_v5._probs.clear()
        lh, la = model_v5.team_logp(f, path)
        ph_full = np.exp(lh)
        # raw team model on the public GT window (alive frames) and its confidence on v5-alive frames
        tq = team_quality(ph_full, np.where(g_alive, gt, 3), f["seen"])
        lab8 = model_v5.predict(f, {"pen_team": 8.0})
        alive8 = ev.atd.graded_mask(len(lab8)) & (lab8 < 2)
        lg = np.abs(lh - la)
        conf = {k: (float(np.median(lg[alive8 & m])), float(np.mean(lg[alive8 & m] > np.log(9))))
                for k, m in (("all", np.ones_like(alive8)), ("seen", f["seen"]), ("miss", ~f["seen"]))}
        out[v] = {"public_team": tq, "conf_v5_alive": conf, "pens": {}}
        lines.append("models %s | public window, raw team model on GT-alive frames: acc %.4f (ball seen %.4f, "
                     "missing %.4f, n_miss %d) log-loss %.4f conf-acc %+.4f | v5-alive frames, full match: "
                     "median |logit p_home| %.2f (seen %.2f, missing %.2f), share p>0.9 %.3f (seen %.3f, missing %.3f)" % (
                         v, tq["all"]["acc"], tq["seen"]["acc"], tq["miss"]["acc"], tq["miss"]["n"],
                         tq["all"]["logloss"], tq["all"]["conf_minus_acc"], conf["all"][0], conf["seen"][0],
                         conf["miss"][0], conf["all"][1], conf["seen"][1], conf["miss"][1]))
        print(lines[-1], flush=True)
        runs = [(a, b + 1) for _, a, b in ev.atd.HALVES]
        for pt in pens:
            lab = model_v5.predict(f, {"pen_team": float(pt)})
            sc, dg, px = ev.score(lab), ev.diagnostics(lab), proxy.proxies(lab, f)
            g = lab[ev.atd.graded_mask(len(lab))]
            alive_min = np.sum(g < 2) / ev.atd.FPS / 60
            lens = np.array([b - a for a, b, _ in ev.pm.possession_spells(lab, runs)]) / ev.atd.FPS
            out[v]["pens"][pt] = {"score": {k: sc[k] for k in ("spellF1", "macroF1", "n_pred", "n_gt", "team_acc",
                                                               "alive_dead_f1")},
                                  "diag": dg, "proxies": px}
            lines.append("  pen_team %2d | public: spellF1 %.4f macroF1 %.4f team_acc %.4f adF1 %.4f n_pred %d/%d | "
                         "full: n_poss %d (%.2f per alive min) median %.2f s <2s %.3f dead share %.3f home share %.3f | "
                         "proxies: tag_team %.4f setpiece %.3f restart %.3f ballout %.3f" % (
                             pt, sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["alive_dead_f1"], sc["n_pred"],
                             sc["n_gt"], dg["n_poss"], dg["n_poss"] / alive_min, dg["poss_median_s"],
                             float(np.mean(lens < 2)), dg["dead_share"], dg["home_share"], px["tag_team_agree"],
                             px["setpiece_ok"], px["restart_any"], px["ballout_cov"]))
            print(lines[-1], flush=True)
        # agreement of the v5 labels with the v1-model v5 labels (same decoder), full match
        out[v]["labels8"] = lab8
    if "v1" in out:
        base = out["v1"]["labels8"]
        gm = ev.atd.graded_mask(len(base))
        for v in out:
            lines.append("v5 pen_team 8 frame agreement with the v1 model set, full match: %s %.4f" % (
                v, np.mean(out[v]["labels8"][gm] == base[gm])))
    for v in out:
        out[v].pop("labels8")
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "barath_v5.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORTS, "barath_v5.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n".join(lines[-len(out):]))


# -- Metrica sample games ---------------------------------------------------------------------------

def _metrica_job(job):
    """One seed: degrade the 3 games under a Metrica degradation, features, probabilities of each model set."""
    import metrica_sample as ms
    regime, s, model_sets = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mods = {v: _load(model_path(v)) for v in model_sets}
    cols = {}
    for v in model_sets:
        ref = _load(os.path.join(cache_dir(v), "{}_feat.pkl".format(dfl.MATCHES[0])))
        cols[v] = (ref["fk"], ref["tk"])
    res = {v: {} for v in model_sets}
    t = time.time()
    for g in ms.GAMES:
        d = ms.parse(g)
        pitch = tuple(float(x) for x in d["pitch"])
        for v in model_sets:
            res[v][g] = []
        for h in (1, 2):
            tag = "h%d" % h
            P, B, lab = d[tag + "_players"], d[tag + "_ball"], d[tag + "_labels"]
            Pn, Bn = sim.degrade(P, B, lab, pitch, seed=7000 + 100 * g + 10 * s + h, params=VARIANTS[regime])
            XF, XH, XA, fk, tk = td.block_features(Pn, Bn, d[tag + "_team"], d[tag + "_gk"], pitch)
            for v in model_sets:
                if (fk, tk) != cols[v]:
                    raise ValueError("feature columns differ from the training features")
                pd_, ph = td.predict_proba(mods[v], XF, XH, XA)
                res[v][g].append((pd_.astype(np.float32), ph.astype(np.float32), lab, np.isfinite(Bn[:, 0])))
    for v in model_sets:
        _dump(res[v], os.path.join(SIMFIX, "metrica_%s_%s" % (v, regime), "probs_seed%d.pkl" % s))
    return regime, s, time.time() - t


def metrica_build(regimes=("v1", "v2"), model_sets=("v1", "v2", "v2t"), procs=6):
    import metrica_calibration as mc
    from multiprocessing import Pool
    model_sets = [v for v in model_sets if os.path.isfile(model_path(v))]
    jobs = [(r, s, model_sets) for r in regimes for s in mc.SEEDS]
    with Pool(min(procs, len(jobs))) as pool:
        for r, s, sec in pool.imap_unordered(_metrica_job, jobs):
            print("metrica degradation %s seed %d built in %.0fs" % (r, s, sec), flush=True)
    # determinism check: v1 models on v1 degradation = cache/metrica (metrica_calibration.build)
    if "v1" in regimes and "v1" in model_sets:
        for s in mc.SEEDS:
            a = _load(os.path.join(SIMFIX, "metrica_v1_v1", "probs_seed%d.pkl" % s))
            b = _load(os.path.join(ROOT, "cache", "metrica", "probs_seed%d.pkl" % s))
            same = all(np.array_equal(x, y) for g in a for ha, hb in zip(a[g], b[g]) for x, y in zip(ha, hb))
            print("seed %d: v1 x v1 identical to cache/metrica: %s" % (s, same))


def sweep_dfl_from(oof_path):
    """metrica_calibration.sweep_dfl on another OOF file."""
    import metrica_calibration as mc
    oof = _load(oof_path)
    labs = {}
    for m in dfl.MATCHES:
        d = dfl.parse(m)
        labs[m] = [d["h1_labels"], d["h2_labels"]]
    rows = []
    for dname, dec in mc.DECODERS.items():
        for mode in mc.MODES:
            for pt in mc.PEN_TEAM:
                for m in dfl.MATCHES:
                    preds = [mc.decode(pd_, ph, pt, dec, np.where(lab == 3, 0, lab) if mode == "oracle" else None)
                             for (pd_, ph), lab in zip(oof[m], labs[m])]
                    r = td.spell_score(labs[m], preds)
                    rows.append((dname, mode, "dfl", pt, 0, m) + tuple(float(x) for x in r))
    return rows


def oof_path(v):
    return os.path.join(ROOT, "cache", "dfl", "oof.pkl") if v == "v1" else os.path.join(cache_dir(v), "oof.pkl")


def metrica_sweep(model_set, regime):
    """metrica_calibration.sweep() for one (model set, Metrica degradation), redirected to
    cache/simfix/metrica_<models>_<regime> and reports/simfix/metrica_<models>_<regime>/."""
    import metrica_calibration as mc
    tag = "metrica_%s_%s" % (model_set, regime)
    mc.CACHE = os.path.join(SIMFIX, tag)
    mc.REPORTS = os.path.join(REPORTS, tag)
    mc.sweep_dfl = lambda: sweep_dfl_from(oof_path(model_set))
    mc.sweep()


def metrica_summary(combos=(("v1", "v1"), ("v1", "v2"), ("v2", "v1"), ("v2", "v2"), ("v2t", "v2"))):
    """Best pen_team per convention (v5 decoder, learned / oracle dead-ball evidence) for each combo,
    the regret table and the paired difference of 20 vs 8, from the redirected sweeps."""
    import metrica_calibration as mc
    lines = []
    for ms_, rg in combos:
        tag = "metrica_%s_%s" % (ms_, rg)
        rep = os.path.join(REPORTS, tag, "metrica_sweep.json") if (ms_, rg) != ("v1", "v1") else \
            os.path.join(ROOT, "reports", "metrica_sweep.json")
        if not os.path.isfile(rep):
            continue
        with open(rep, encoding="utf-8") as fh:
            res = json.load(fh)
        q = res["quality"]
        lines.append("models %s, Metrica degradation %s: alive/dead acc %.3f team acc %.3f ball missing alive %.3f "
                     "dead %.3f median |logit| %.2f p>0.9 %.3f" % ((ms_, rg) + tuple(q)))
        for dec in ("v5",):
            for mode in mc.MODES:
                cells = []
                for c in list(mc.CONVS):
                    summ = res["metrica"]["%s|%s|%s" % (dec, mode, c)]
                    means = {int(k): v[0] for k, v in summ.items()}
                    b = max(means, key=means.get)
                    band = [k for k in sorted(means) if means[k] >= means[b] - 0.005]
                    cells.append("%s %d [%d-%d] %.4f (8: %.4f, 20: %.4f)" % (c, b, band[0], band[-1], means[b],
                                                                         means[8], means[20]))
                summ = res["dfl"]["%s|%s" % (dec, mode)]
                means = {int(k): v[0] for k, v in summ.items()}
                b = max(means, key=means.get)
                cells.append("DFL %d %.4f" % (b, means[b]))
                lines.append("  decoder %s %-6s | %s" % (dec, mode, " | ".join(cells)))
    print("\n".join(lines))
    return lines


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "build":
        build(args or ["v2", "v2t"])
    elif cmd == "cv":
        cv(args[0])
    elif cmd == "folds":
        folds(args[0])
    elif cmd == "cross":
        cross()
    elif cmd == "print_cross":
        print_cross()
    elif cmd == "models":
        for v in args:
            print(models(v))
    elif cmd == "barath":
        barath(tuple(args) if args else ("v1", "v2", "v2t"))
    elif cmd == "metrica_build":
        metrica_build()
    elif cmd == "metrica_sweep":
        metrica_sweep(args[0], args[1])
    elif cmd == "metrica_summary":
        metrica_summary()
