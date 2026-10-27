"""Calibrate the decoder's team-switch penalty (pen_team) on Metrica's public sample games, whose
event-derived possession convention is much closer to our ground truth than open DFL's.

Per game and half: clean Metrica tracking (metrica_sample.py) -> sim.degrade (3 seeds) -> mlfeat
features -> the DFL-trained LightGBM models (cache/dfl/models_all.pkl) -> emissions ->
min-duration Viterbi (train_dfl.decode_block) -> official spell metric per match
(train_dfl.spell_score), against the event labels under several possession conventions
(metrica_sample.CONVENTIONS). The same sweep runs on the DFL out-of-fold probabilities
(cache/dfl/oof.pkl) for comparison.

  .venv/Scripts/python.exe -I src/metrica_calibration.py build     # degrade + features + probs (cached)
  .venv/Scripts/python.exe -I src/metrica_calibration.py sweep     # pen_team curves -> reports/
  .venv/Scripts/python.exe -I src/metrica_calibration.py tables    # markdown tables for the report
  .venv/Scripts/python.exe -I src/metrica_calibration.py paired 8  # paired differences vs pen_team 8
  .venv/Scripts/python.exe -I src/metrica_calibration.py regret    # regret across conventions
  .venv/Scripts/python.exe -I src/metrica_calibration.py windows   # public window vs 5-min windows
  .venv/Scripts/python.exe -I src/metrica_calibration.py barath    # v5 per pen_team, aggregates only
  .venv/Scripts/python.exe -I src/metrica_calibration.py plot 8    # charts, 8 = marked pen_team
"""
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
import metrica_sample as ms  # noqa: E402
import sim  # noqa: E402
import train_dfl as td  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "metrica")
REPORTS = os.path.join(ROOT, "reports")
SEEDS = (0, 1, 2)
PEN_TEAM = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
CONVS = ms.CONVENTIONS
CONV_NAME = {"events": "event labels", "tau1": "short spells < 1 s merged", "tau2": "short spells < 2 s merged",
             "tau3": "short spells < 3 s merged", "ctrl": "event-controlled"}
# decoder settings around pen_team: model_v5 inherits pen_dead/min_poss/min_dead from model_v2;
# train_dfl's DFL CV picked the second set
DECODERS = {"v5": dict(pen_dead=30.0, min_poss=8, min_dead=50),
            "dflcv": dict(pen_dead=32.0, min_poss=5, min_dead=25)}
MODES = ("model", "oracle")      # dead-ball evidence: the DFL dead model, or the true D frames
EPS = 1e-4


def _models():
    with open(os.path.join(ROOT, "cache", "dfl", "models_all.pkl"), "rb") as fh:
        return pickle.load(fh)


def _check_columns(fk, tk, models):
    with open(os.path.join(ROOT, "cache", "dfl", "{}_feat.pkl".format(dfl.MATCHES[0])), "rb") as fh:
        ref = pickle.load(fh)
    if fk != ref["fk"] or tk != ref["tk"]:
        raise ValueError("feature columns differ from the ones the DFL models were trained on")
    dead, team = models
    if dead.num_feature() != len(fk) or team.num_feature() != len(fk) + len(tk):
        raise ValueError("model feature count mismatch")


def build(refresh=False):
    """Per seed: {game: [(pdead, phome, labels, ball_seen), ...per half]} in cache/metrica."""
    os.makedirs(CACHE, exist_ok=True)
    models = None
    for s in SEEDS:
        out = os.path.join(CACHE, "probs_seed%d.pkl" % s)
        if os.path.isfile(out) and not refresh:
            continue
        models = models or _models()
        res = {}
        for g in ms.GAMES:
            t = time.time()
            d = ms.parse(g)
            pitch = tuple(float(v) for v in d["pitch"])
            res[g] = []
            for h in (1, 2):
                tag = "h%d" % h
                P, B, lab = d[tag + "_players"], d[tag + "_ball"], d[tag + "_labels"]
                Pn, Bn = sim.degrade(P, B, lab, pitch, seed=7000 + 100 * g + 10 * s + h)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    XF, XH, XA, fk, tk = td.block_features(Pn, Bn, d[tag + "_team"], d[tag + "_gk"], pitch)
                _check_columns(fk, tk, models)
                pd_, ph = td.predict_proba(models, XF, XH, XA)
                res[g].append((pd_.astype(np.float32), ph.astype(np.float32), lab,
                               np.isfinite(Bn[:, 0])))
            print("seed %d game %d built in %.0fs" % (s, g, time.time() - t), flush=True)
        with open(out, "wb") as fh:
            pickle.dump(res, fh, protocol=4)


def load_probs():
    out = {}
    for s in SEEDS:
        with open(os.path.join(CACHE, "probs_seed%d.pkl" % s), "rb") as fh:
            out[s] = pickle.load(fh)
    return out


def frame_quality(probs):
    rows = []
    for s, res in probs.items():
        for g, halves in res.items():
            for pd_, ph, lab, seen in halves:
                alive = lab < 2
                p = np.clip(ph[alive], EPS, 1 - EPS)
                lg = np.abs(np.log(p / (1 - p)))
                rows.append((np.mean((pd_ > 0.5) == (lab == 2)),
                             np.mean((ph[alive] > 0.5) == (lab[alive] == 0)),
                             1 - seen[alive].mean(), 1 - seen[lab == 2].mean(),
                             np.median(lg), np.mean(lg > np.log(9))))
    return np.array(rows).mean(0)


def decode(pd_, ph, pen_team, dec, oracle_lab=None):
    if oracle_lab is not None:
        pd_ = np.where(oracle_lab == 2, 1 - EPS, EPS)
    S = td.emissions(pd_, ph, 1.0, 1.0)
    return td.decode_block(S, pen_team, dec["pen_dead"], dec["min_poss"], dec["min_dead"]).astype(np.int8)


def sweep_metrica(probs):
    """rows: (decoder, dead_mode, convention, pen_team, seed, game, spellF1, macroF1, n_gt, n_pred)."""
    gts = {g: {c: ms.gt_halves(g, c) for c in CONVS} for g in ms.GAMES}
    for res in probs.values():
        for g, halves in res.items():
            if not all(np.array_equal(lab, gl) for (_, _, lab, _), gl in zip(halves, gts[g]["events"])):
                raise ValueError("cached probabilities were built on different labels; rerun build")
    rows = []
    for dname, dec in DECODERS.items():
        for mode in MODES:
            for pt in PEN_TEAM:
                for s, res in probs.items():
                    for g, halves in res.items():
                        preds = [decode(pd_, ph, pt, dec, lab if mode == "oracle" else None)
                                 for pd_, ph, lab, _ in halves]
                        for c in CONVS:
                            r = td.spell_score(gts[g][c], preds)
                            rows.append((dname, mode, c, pt, s, g) + tuple(float(v) for v in r))
            print("metrica sweep", dname, mode, "done", flush=True)
    return rows


def sweep_dfl():
    """Same sweep on the DFL leave-one-match-out probabilities, DFL's own labels."""
    with open(os.path.join(ROOT, "cache", "dfl", "oof.pkl"), "rb") as fh:
        oof = pickle.load(fh)
    labs = {}
    for m in dfl.MATCHES:
        d = dfl.parse(m)
        labs[m] = [d["h1_labels"], d["h2_labels"]]
    rows = []
    for dname, dec in DECODERS.items():
        for mode in MODES:
            for pt in PEN_TEAM:
                for m in dfl.MATCHES:
                    preds = [decode(pd_, ph, pt, dec, np.where(lab == 3, 0, lab) if mode == "oracle" else None)
                             for (pd_, ph), lab in zip(oof[m], labs[m])]
                    r = td.spell_score(labs[m], preds)
                    rows.append((dname, mode, "dfl", pt, 0, m) + tuple(float(v) for v in r))
            print("dfl sweep", dname, mode, "done", flush=True)
    return rows


def summarise(rows, key_filter):
    """Per pen_team: (spellF1 mean over matches, min over matches, macroF1, n_gt, n_pred, per-match)."""
    out = {}
    for pt in PEN_TEAM:
        sel = [r for r in rows if key_filter(r) and r[3] == pt]
        if not sel:
            continue
        games = sorted(set(r[5] for r in sel))
        per_game = [np.mean([r[6] for r in sel if r[5] == g]) for g in games]
        out[pt] = (float(np.mean(per_game)), float(np.min(per_game)), float(np.mean([r[7] for r in sel])),
                   float(np.mean([r[8] for r in sel])), float(np.mean([r[9] for r in sel])),
                   [float(x) for x in per_game])
    return out


def _curve(summ):
    return "  ".join("%d:%.4f(%d/%d)" % (pt, v[0], v[4], v[3]) for pt, v in summ.items())


def sweep():
    probs = load_probs()
    q = frame_quality(probs)
    lines = ["Metrica degraded: alive/dead acc %.3f  team acc (alive) %.3f  ball missing alive %.3f dead %.3f  "
             "median |logit p_home| (alive) %.2f  share p>0.9 %.3f" % tuple(q)]
    mrows = sweep_metrica(probs)
    drows = sweep_dfl()
    res = {"quality": q.tolist(), "metrica": {}, "dfl": {}}
    for dname in DECODERS:
        for mode in MODES:
            for c in CONVS:
                summ = summarise(mrows, lambda r: r[0] == dname and r[1] == mode and r[2] == c)
                res["metrica"]["%s|%s|%s" % (dname, mode, c)] = summ
                best = max(summ, key=lambda k: summ[k][0])
                lines.append("METRICA decoder %-5s dead %-6s %-6s | best pen_team %2d spellF1 %.4f | %s" % (
                    dname, mode, c, best, summ[best][0], _curve(summ)))
            summ = summarise(drows, lambda r: r[0] == dname and r[1] == mode)
            res["dfl"]["%s|%s" % (dname, mode)] = summ
            best = max(summ, key=lambda k: summ[k][0])
            lines.append("DFL     decoder %-5s dead %-6s        | best pen_team %2d spellF1 %.4f | %s" % (
                dname, mode, best, summ[best][0], _curve(summ)))
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "metrica_sweep.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORTS, "metrica_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(CACHE, "sweep_rows.pkl"), "wb") as fh:
        pickle.dump({"metrica": mrows, "dfl": drows}, fh, protocol=4)
    print("\n".join(lines))


def _load_sweep():
    with open(os.path.join(REPORTS, "metrica_sweep.json"), encoding="utf-8") as fh:
        return json.load(fh)


def plot(recommended=None, decoder="v5", mode="model", convs=("events", "ctrl", "tau2")):
    """spellF1 and predicted/true spell count vs pen_team (from reports/metrica_sweep.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = _load_sweep()
    ink, ink2, muted, grid, axis, surf = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots, fixed order
    series = [("Metrica, " + CONV_NAME[c], res["metrica"]["%s|%s|%s" % (decoder, mode, c)]) for c in convs]
    series.append(("open DFL (own BallPossession labels)", res["dfl"]["%s|%s" % (decoder, mode)]))
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surf)
    panels = [("spellF1 (mean over matches)", lambda v: v[0]),
              ("predicted / true possession spells", lambda v: v[4] / v[3])]
    for ax, (ylabel, fn) in zip(axes, panels):
        ax.set_facecolor(surf)
        for (name, summ), col in zip(series, colors):
            x, y = zip(*sorted((int(k), fn(v)) for k, v in summ.items()))
            ax.plot(x, y, color=col, lw=2, solid_joinstyle="round", solid_capstyle="round", label=name, zorder=3)
            if ylabel.startswith("spellF1"):
                i = int(np.argmax(y))
                ax.scatter([x[i]], [y[i]], s=64, color=col, edgecolors=surf, linewidths=2, zorder=4)
                ax.annotate("%d" % x[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 8),
                            ha="center", color=ink2, fontsize=8)
        if ylabel.startswith("predicted"):
            ax.axhline(1.0, color=muted, lw=1, zorder=2)
        if recommended:
            ax.axvline(recommended, color=muted, lw=1, zorder=2)
            ax.annotate("recommended %g" % recommended, (recommended, 0), xycoords=("data", "axes fraction"),
                        xytext=(4, 4), textcoords="offset points", color=ink2, fontsize=8)
        ticks = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(p) for p in ticks])
        ax.minorticks_off()
        ax.set_xlabel("pen_team (k_team = 1)", color=ink2)
        ax.set_ylabel(ylabel, color=ink2)
        ax.grid(True, color=grid, lw=1, ls="-", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(axis)
        ax.tick_params(colors=muted, labelcolor=ink2)
    axes[0].legend(frameon=False, labelcolor=ink, loc="lower left", fontsize=8)
    fig.suptitle("Team-switch penalty sweep: decoder %s, %s dead-ball evidence, degraded tracking (3 seeds)" % (
        decoder, "learned" if mode == "model" else "oracle"), color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(REPORTS, "metrica_pen_team_%s_%s.png" % (decoder, mode))
    fig.savefig(out, dpi=150, facecolor=surf)
    print("wrote", out)


def tables():
    """Markdown tables for docs/RAPOR.md Ek B (spell statistics + sweep summaries)."""
    out = ["| set | convention | spells/match | per alive min | median s | mean s | < 2 s | < 1 s | "
           "D segs/match | D share |", "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = "| %s | %s | %.0f | %.2f | %.2f | %.2f | %.3f | %.3f | %.0f | %.3f |"
    for c in CONVS:
        allh = []
        for g in ms.GAMES:
            halves = ms.gt_halves(g, c)
            allh += halves
            if c == "events":
                s = ms.spell_stats(halves)
                out.append(fmt % ("Metrica game %d" % g, CONV_NAME[c], s["n_spells"], s["n_spells"] / s["alive_min"],
                                  s["median_s"], s["mean_s"], s["lt2s"], s["lt1s"], s["n_dead_segs"],
                                  s["dead_share"]))
        s = ms.spell_stats(allh)
        k = len(ms.GAMES)
        out.append(fmt % ("**Metrica, 3 games**", CONV_NAME[c], s["n_spells"] / k, s["n_spells"] / s["alive_min"],
                          s["median_s"], s["mean_s"], s["lt2s"], s["lt1s"], s["n_dead_segs"] / k, s["dead_share"]))
    allh = []
    for m in dfl.MATCHES:
        d = dfl.parse(m)
        allh += [d["h1_labels"], d["h2_labels"]]
    s = ms.spell_stats(allh)
    k = len(dfl.MATCHES)
    out.append(fmt % ("open DFL, 7 matches", "BallPossession", s["n_spells"] / k, s["n_spells"] / s["alive_min"],
                      s["median_s"], s["mean_s"], s["lt2s"], s["lt1s"], s["n_dead_segs"] / k, s["dead_share"]))
    print("\n".join(out) + "\n")

    res = _load_sweep()
    for dec in DECODERS:
        for mode in MODES:
            cols = [(c, res["metrica"]["%s|%s|%s" % (dec, mode, c)]) for c in CONVS]
            cols.append(("DFL", res["dfl"]["%s|%s" % (dec, mode)]))
            hdr = "| pen_team | " + " | ".join(c for c, _ in cols) + " | pred / true spells (events) |"
            rows = [hdr, "|" + "---|" * (len(cols) + 2)]
            best = {c: max(v.values(), key=lambda r: r[0])[0] for c, v in cols}
            for pt in PEN_TEAM:
                cells = [("**%.4f**" if v[str(pt)][0] == best[c] else "%.4f") % v[str(pt)][0] for c, v in cols]
                v0 = cols[0][1][str(pt)]
                rows.append("| %d | %s | %.0f / %.0f |" % (pt, " | ".join(cells), v0[4], v0[3]))
            print("decoder %s, %s dead-ball evidence\n" % (dec, mode))
            print("\n".join(rows) + "\n")
            # near-optimal band (within 0.005 of the best mean) and per-match argmax
            for c, v in cols:
                means = {int(k): r[0] for k, r in v.items()}
                b = max(means, key=means.get)
                band = [k for k in sorted(means) if means[k] >= means[b] - 0.005]
                games = np.array([v[str(pt)][5] for pt in PEN_TEAM])   # (pen, match)
                per_game = [PEN_TEAM[i] for i in games.argmax(0)]
                print("  %-6s best %2d (%.4f)  within 0.005: %s  per-match argmax %s" % (
                    c, b, means[b], band, per_game))
            print()


def paired(ref=8, decoder="v5", mode="model"):
    """spellF1(pen_team) - spellF1(ref), paired over the 9 (seed, game) runs: mean, sd, wins."""
    with open(os.path.join(CACHE, "sweep_rows.pkl"), "rb") as fh:
        rows = pickle.load(fh)["metrica"]
    for c in CONVS:
        sel = {(r[3], r[4], r[5]): r[6] for r in rows if r[0] == decoder and r[1] == mode and r[2] == c}
        runs = sorted({(s, g) for _, s, g in sel})
        cells = []
        for pt in PEN_TEAM:
            dlt = np.array([sel[(pt, s, g)] - sel[(ref, s, g)] for s, g in runs])
            cells.append("%d:%+.4f(sd %.4f, %d/%d)" % (pt, dlt.mean(), dlt.std(ddof=1), np.sum(dlt > 0), len(dlt)))
        print("%-6s vs pen_team %d: %s" % (c, ref, "  ".join(cells)))


def regret(decoder="v5", priors=None):
    """Regret of each pen_team (best spellF1 of a convention minus its spellF1), per convention, and
    its expectation under a few priors over which convention the hidden ground truth follows."""
    res = _load_sweep()
    # "public-window" weights each convention by the share of its Metrica 5-min windows that are at
    # least as calm as the public window (rate and short-spell share, see windows())
    pw = {c: v[1] for c, v in _window_stats()[1].items()}
    priors = priors or {"ctrl-centred": {"ctrl": .5, "events": .15, "tau1": .1, "tau2": .15, "tau3": .1},
                        "uniform": {c: 1.0 / len(CONVS) for c in CONVS},
                        "public-window": {c: pw[c] / sum(pw.values()) for c in CONVS},
                        "events-only": {c: float(c == "events") for c in CONVS}}
    for mode in MODES:
        F = np.array([[res["metrica"]["%s|%s|%s" % (decoder, mode, c)][str(p)][0] for p in PEN_TEAM]
                      for c in CONVS])
        R = F.max(1, keepdims=True) - F
        print("decoder %s, %s dead-ball evidence | pen_team: %s" % (decoder, mode, list(PEN_TEAM)))
        for c, r in zip(CONVS, R):
            print("  regret %-6s %s" % (c, " ".join("%.4f" % x for x in r)))
        for nm, w in priors.items():
            er = np.array([w[c] for c in CONVS]) @ R
            print("  E[regret] %-18s %s  -> best %d" % (nm, " ".join("%.4f" % x for x in er),
                                                      PEN_TEAM[int(np.argmin(er))]))
        mx = R.max(0)
        print("  max regret (all)       %s  -> best %d" % (" ".join("%.4f" % x for x in mx),
                                                         PEN_TEAM[int(np.argmin(mx))]))


def _window_stats(win_s=300, step_s=30):
    """Public-window aggregates (from the public ground truth) and, per convention, the 5-minute
    windows of the Metrica labels as rows (spells per alive min, median s, share < 2 s) plus the share
    of windows at least as calm as the public one (rate and short-spell share both <= public)."""
    import evaluate as ev
    gt = ev.public_gt()
    sp = ev.pm.possession_spells(gt, ev.pm.valid_runs(gt))
    L = np.array([b - a for a, b, _ in sp]) / ms.FPS
    alive = np.sum(gt < 2) / ms.FPS
    pub = {"n": len(L), "alive_s": alive, "rate": len(L) / (alive / 60), "median": float(np.median(L)),
           "mean": float(L.mean()), "lt2": float(np.mean(L < 2))}
    W, step = win_s * ms.FPS, step_s * ms.FPS
    out = {}
    for c in CONVS:
        rows = []
        for g in ms.GAMES:
            for lab in ms.gt_halves(g, c):
                for s in range(0, len(lab) - W + 1, step):
                    w = lab[s:s + W]
                    sp = ms.spells(w)
                    if not sp:
                        continue
                    x = np.array([b - a for a, b, _ in sp]) / ms.FPS
                    rows.append((len(x) / (np.sum(w < 2) / ms.FPS / 60), np.median(x), np.mean(x < 2)))
        R = np.array(rows)
        out[c] = (R, float(np.mean((R[:, 0] <= pub["rate"]) & (R[:, 2] <= pub["lt2"]))))
    return pub, out


def windows():
    """Where does the public 5-minute window sit among 5-minute windows of the Metrica labels?"""
    pub, out = _window_stats()
    print("public window: %d spells, %.1f s alive, %.2f per alive min, median %.2f s, mean %.2f s, <2 s %.3f" % (
        pub["n"], pub["alive_s"], pub["rate"], pub["median"], pub["mean"], pub["lt2"]))
    for c, (R, joint) in out.items():
        print("%-6s: %d windows | percentile of the public value: rate %.0f  median %.0f  <2s %.0f | share of "
              "windows with rate <= and <2s share <= public: %.3f | window medians: rate %.2f median %.2f "
              "<2s %.3f" % (c, len(R), 100 * np.mean(R[:, 0] <= pub["rate"]), 100 * np.mean(R[:, 1] <= pub["median"]),
                            100 * np.mean(R[:, 2] <= pub["lt2"]), joint, *np.median(R, 0)))


def barath():
    """model_v5 on BAR-ATH for each pen_team. Prints aggregate statistics only (competition data):
    full-match possession count / durations and the 17-possession public-window score."""
    import evaluate as ev
    import model_v5
    from make_submission import merged_features
    f = merged_features()
    lines = []
    # confidence of the DFL team model on BAR-ATH (compare with frame_quality on degraded Metrica)
    lh, la = model_v5.team_logp(f)
    lab8 = ev.to_codes(model_v5.predict(f, {"pen_team": 8.0}))
    alive8 = ev.atd.graded_mask(len(lab8)) & (lab8 < 2)
    lg = np.abs(lh - la)[alive8]
    lines.append("BAR-ATH team model on v5-alive frames: median |logit p_home| %.2f  share p>0.9 %.3f" % (
        np.median(lg), np.mean(lg > np.log(9))))
    print(lines[-1], flush=True)
    runs = [(a, b + 1) for _, a, b in ev.atd.HALVES]
    for pt in PEN_TEAM:
        codes = ev.to_codes(model_v5.predict(f, {"pen_team": float(pt)}))
        dg = ev.diagnostics(codes)
        sc = ev.score(codes)
        g = codes[ev.atd.graded_mask(len(codes))]
        alive_min = np.sum((g == 0) | (g == 1)) / ev.atd.FPS / 60
        lens = np.array([b - a for a, b, _ in ev.pm.possession_spells(codes, runs)]) / ev.atd.FPS
        lines.append("pen_team %2d | full match: n_poss %d (%.2f per alive min)  median %.2f s  mean %.2f s  "
                     "<2s %.3f  dead share %.3f | public window: spellF1 %.4f  macroF1 %.4f  n_pred %d/%d" % (
                         pt, dg["n_poss"], dg["n_poss"] / alive_min, dg["poss_median_s"], dg["poss_mean_s"],
                         float(np.mean(lens < 2)), dg["dead_share"], sc["spellF1"], sc["macroF1"],
                         sc["n_pred"], sc["n_gt"]))
        print(lines[-1], flush=True)
    with open(os.path.join(REPORTS, "metrica_barath_v5_pen_team.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "sweep":
        sweep()
    elif cmd == "tables":
        tables()
    elif cmd == "paired":
        paired(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif cmd == "regret":
        regret()
    elif cmd == "windows":
        windows()
    elif cmd == "barath":
        barath()
    elif cmd == "plot":
        rec = float(sys.argv[2]) if len(sys.argv) > 2 else None
        for dec in DECODERS:
            for mode in MODES:
                plot(rec, dec, mode)
