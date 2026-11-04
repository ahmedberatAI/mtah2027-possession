"""Round 43 (docs/RAPOR.md, pre-registered): the dead-ball rules' constants from open data only. The rules of model_v2 / model_v3 were written on 20 October while
looking at BAR-ATH's public window (the speed look-ahead was even scanned with its live / dead F1); the official rules say that window is not a tuning set. Here the same
functional form gets every constant from the open sets: thresholds (speed look-ahead, ball margin, unseen time, stillness, fast ball) by provider-balanced log-loss on DFL, de-glued
Metrica and SkillCorner, weights by logistic regression of true alive on the terms, the view flag's two weights on SkillCorner alone (its flag is real; a transplanted one
switches at true boundaries). BAR-ATH plays no part.

  python -I src/exp_rules_open.py fit      thresholds and weights; writes cache/bench/rules_open.json
  python -I src/exp_rules_open.py eval     held-out weights by match: old against open-data rules, alone and in the 0.5 mix, the decision rule
  python -I src/exp_rules_open.py arb K    all candidates on one open provider (dfl, metrica, sc); decide pools them
  python -I src/exp_rules_open.py pff      PFF's 64 matches as the independent test; st: SoccerTrack v2
  python -I src/exp_rules_open.py final K  round 44's candidates and stress on one open provider; fdecide pools; fpff, fst the independent tests
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import dead_timing as dt  # noqa: E402
import exp_blend  # noqa: E402
import exp_deadmix as dm  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402

OUT = os.path.join(bench.CACHE, "rules_open.json")
SOURCES = {"dfl": ("dfl@v3", "dfl@v3", "evctrl"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}
GRID = dict(shift=(0, 12, 25, 37, 50, 75), out_margin=(0.0, 0.5, 1.0, 2.0), missing_after=(12, 25, 50), still_disp=(0.3, 0.6, 1.0), ball_fast=(3.0, 5.0, 7.0))
START = dict(shift=25, out_margin=1.0, missing_after=25, still_disp=1.0, ball_fast=5.0)
STRIDE = 10
_f = {}


def data():
    """provider -> list of (rule fields, truth, group, valid, live)."""
    if _f:
        return _f
    for k, (fsrc, lsrc, conv) in SOURCES.items():
        blocks = bench.blocks(fsrc)
        labs = bench.labels(lsrc, conv)
        cues = dm.sd.view_cues() if k == "sc" else [(None, None)] * len(blocks)
        _f[k] = []
        for b, g, (v, n) in zip(blocks, labs, cues):
            r = exp_blend.rule_f(b)
            keep = {x: r[x] for x in ("n", "seen", "bxy", "bspeed", "ball_out", "since_seen", "spd_all_s25", "spd_all_s100")}
            _f[k].append((keep, np.asarray(g), b["group"], v, (v & (n >= 4)) if v is not None else None))
        print("  %s: %d halves" % (k, len(_f[k])), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    return _f


def bounds():
    """Tempo clip bounds: the pooled 2nd and 98th percentiles of the open sets."""
    d = data()
    v1 = np.concatenate([f["spd_all_s25"][::STRIDE] for k in d for f, *_ in d[k]])
    v3 = np.concatenate([f["spd_all_s100"][::STRIDE] for k in d for f, *_ in d[k]])
    q1, q3 = np.nanquantile(v1, (0.02, 0.5, 0.98)), np.nanquantile(v3, (0.02, 0.5, 0.98))
    return dict(v1=q1.tolist(), v3=q3.tolist())


def terms(f, t, b):
    """The rule form's terms per frame: intercept, tempo over +-1 s and +-3 s (clipped, looked up t['shift'] ahead), ball out, ball unseen, ball still, ball fast."""
    n = f["n"]
    s = int(t["shift"])
    v1 = np.clip(np.nan_to_num(np.roll(f["spd_all_s25"], -s), nan=b["v1"][1]), b["v1"][0], b["v1"][2])
    v3 = np.clip(np.nan_to_num(np.roll(f["spd_all_s100"], -s), nan=b["v3"][1]), b["v3"][0], b["v3"][2])
    bo = np.nan_to_num(f["ball_out"], nan=-1) > t["out_margin"]
    seen = f["seen"]
    miss = (~seen) & (f["since_seen"] > t["missing_after"])
    lo, hi = np.clip(np.arange(n) - 12, 0, n - 1), np.clip(np.arange(n) + 12, 0, n - 1)
    ok = seen & seen[lo] & seen[hi]
    disp = np.where(ok, np.linalg.norm(f["bxy"][hi] - f["bxy"][lo], axis=1), np.nan)
    still = ok & (disp < t["still_disp"])
    inside = np.nan_to_num(f["ball_out"], nan=1) <= 0
    fast = seen & inside & (np.nan_to_num(f["bspeed"]) > t["ball_fast"])
    return np.stack([np.ones(n), v1, v3, bo, miss, still, fast], 1).astype(np.float64)


def logistic(X, y, w, offset=None, it=25, ridge=1e-3):
    """Weighted logistic regression by Newton steps."""
    beta = np.zeros(X.shape[1])
    off = 0.0 if offset is None else offset
    w = w / w.mean()
    for _ in range(it):
        z = np.clip(X @ beta + off, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (w * (y - p)) - ridge * len(y) * beta
        h = (X * (w * np.maximum(p * (1 - p), 1e-6))[:, None]).T @ X + ridge * len(y) * np.eye(X.shape[1])
        d = np.linalg.solve(h, g)
        beta += d
        if np.abs(d).max() < 1e-5:
            break
    z = np.clip(X @ beta + off, -30, 30)
    p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
    return beta, float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / len(y))


def design(t, b, skip=()):
    """(X, y, weights) over all providers, each the same total weight, skipping the given groups."""
    d = data()
    Xs, ys, ws = [], [], []
    for k in d:
        xk, yk = [], []
        for f, g, grp, _, _ in d[k]:
            if grp in skip:
                continue
            m = g[::STRIDE] < 3
            xk.append(terms(f, t, b)[::STRIDE][m])
            yk.append((g[::STRIDE][m] < 2).astype(float))
        x, y = np.concatenate(xk), np.concatenate(yk)
        Xs.append(x)
        ys.append(y)
        ws.append(np.full(len(y), 1.0 / len(y)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def fit_view(t, b, beta, skip=()):
    """The view flag's two weights on SkillCorner alone, with the tracking terms as a fixed offset."""
    d = data()["sc"]
    X, y, off = [], [], []
    for f, g, grp, v, l in d:
        if grp in skip:
            continue
        m = g[::STRIDE] < 3
        X.append(np.stack([(~v)[::STRIDE][m], l[::STRIDE][m]], 1).astype(np.float64))
        y.append((g[::STRIDE][m] < 2).astype(float))
        off.append((terms(f, t, b) @ beta)[::STRIDE][m])
    X, y, off = np.concatenate(X), np.concatenate(y), np.concatenate(off)
    gamma, _ = logistic(X, y, np.ones(len(y)), offset=off)
    return gamma


def fit():
    b = bounds()
    print("tempo bounds", {k: np.round(v, 3).tolist() for k, v in b.items()}, flush=True)
    t = dict(START)
    for name, vals in GRID.items():
        best = None
        for v in vals:
            t2 = dict(t, **{name: v})
            X, y, w = design(t2, b)
            _, ll = logistic(X, y, w)
            print("  %-14s %-5s log-loss %.5f" % (name, v, ll), flush=True)
            if best is None or ll < best[0] - 1e-5:
                best = (ll, v)
        t[name] = best[1]
    X, y, w = design(t, b)
    beta, ll = logistic(X, y, w)
    gamma = fit_view(t, b, beta)
    names = ("intercept", "tempo +-1 s", "tempo +-3 s", "ball out", "ball unseen", "ball still", "ball fast")
    print("thresholds", t)
    print("weights", {n: round(float(x), 3) for n, x in zip(names, beta)})
    print("view weights: invalid %.3f, live %.3f; log-loss %.5f" % (gamma[0], gamma[1], ll), flush=True)
    with open(OUT, "w") as fh:
        json.dump(dict(thresholds=t, bounds=b, beta=beta.tolist(), gamma=gamma.tolist()), fh, indent=1)


def evidence(k, conf, draw=0):
    """Open-data rule evidence per half of provider k, weights held out by match (SkillCorner by five folds); without and with the view flag."""
    d = data()[k]
    t, b = conf["thresholds"], conf["bounds"]
    groups = sorted({x[2] for x in d}, key=str)
    folds = [groups[i::5] for i in range(5)] if k == "sc" else [[g] for g in groups]
    track, view = [None] * len(d), [None] * len(d)
    for fold in folds:
        X, y, w = design(t, b, skip=fold)
        beta, _ = logistic(X, y, w)
        gamma = fit_view(t, b, beta, skip=fold if k == "sc" else ())
        for i, (f, g, grp, v, l) in enumerate(d):
            if grp in fold:
                track[i] = terms(f, t, b) @ beta
                if v is None:
                    v2, l2 = dm.transplant(g, np.random.default_rng([dm.SEED, draw, i]))
                else:
                    v2, l2 = v, l
                view[i] = gamma[0] * (~v2) + gamma[1] * l2
        print("    %s fold %s done" % (k, fold), flush=True)
    return track, view


def evaluate():
    import exp_blend_weight as bw
    with open(OUT) as fh:
        conf = json.load(fh)
    names = ("v17 recipe", "v18", "open rules+DT", "open mix.5", "open mix1", "open mix.25")
    cells = {}
    for k, (fsrc, lsrc, conv0) in SOURCES.items():
        src = lsrc
        convs = dm.CONVS[src]
        track, view = evidence(k, conf)
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", fsrc)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        a_rb = dm.rule_view(fsrc, blocks, labs[convs[0]], 0)
        a_new = [x + v for x, v in zip(track, view)]
        smos = dm.team(src)
        for s in es.SEEDS:
            ref = sd.decode_scores(a_lrn, smos[s])
            res = {"v17 recipe": [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_rb, smos[s]), ref)], "v18": sd.decode_scores(dm.mix_evidence(a_rb, a_lrn), smos[s]),
                   "open rules+DT": [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_new, smos[s]), ref)], "open mix.5": sd.decode_scores(dm.mix_evidence(a_new, a_lrn), smos[s]),
                   "open mix1": sd.decode_scores(dm.mix_evidence(a_new, a_lrn, 1.0), smos[s]), "open mix.25": sd.decode_scores(dm.mix_evidence(a_new, a_lrn, 0.25), smos[s])}
            for name, preds in res.items():
                for c in convs:
                    cells.setdefault((src, c, dm.TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            if s == es.SEEDS[0]:
                truth = labs[convs[0]]
                print("  %s: dead share true %.3f | %s" % (src, np.mean(np.concatenate(truth) == 2), " | ".join("%s %.3f F1 %.3f" % (n, np.mean(np.concatenate(res[n]) == 2), dm._dead_f1(res[n], truth)) for n in names)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="v18")
    for cand in names[2:]:
        bw.decide(mean, "v18", cand)
    bw.decide(mean, "v17 recipe", "open mix.5")


NAMES = ("v17 recipe", "v18", "O+DT", "O.25", "O.5", "OE.25", "OE.5", "M+V", "LM+V", "OL.25M")


def arms(a_rb, a_new, a_view, a_lrn, a_m, smo):
    """All candidates on one source and one team probability. a_new: open rules + their view terms; a_view: the open view terms alone."""
    ref = sd.decode_scores(a_lrn, smo)
    ev = {"v18": dm.mix_evidence(a_rb, a_lrn), "O.25": dm.mix_evidence(a_new, a_lrn, 0.25), "O.5": dm.mix_evidence(a_new, a_lrn, 0.5),
          "OE.25": [r + 0.25 * l + 0.25 * m for r, l, m in zip(a_new, a_lrn, a_m)], "OE.5": [r + 0.5 * l + 0.5 * m for r, l, m in zip(a_new, a_lrn, a_m)],
          "M+V": [m + v for m, v in zip(a_m, a_view)], "LM+V": [0.5 * l + 0.5 * m + v for l, m, v in zip(a_lrn, a_m, a_view)], "OL.25M": [r + 0.25 * m for r, m in zip(a_new, a_m)]}
    res = {k: sd.decode_scores(v, smo) for k, v in ev.items()}
    res["v17 recipe"] = [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_rb, smo), ref)]
    res["O+DT"] = [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_new, smo), ref)]
    return res


def arb_open(k):
    """Round 43's candidates on one open provider, weights and the multi-source model held out by match."""
    import pickle
    with open(OUT) as fh:
        conf = json.load(fh)
    fsrc, src, _ = SOURCES[k]
    convs = dm.CONVS[src]
    track, view = evidence(k, conf)
    blocks = bench.blocks(fsrc)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", fsrc)
    a_lrn = [exp_blend.logit_alive(x) for x in base]
    a_m = [exp_blend.logit_alive(x) for x in dm.balanced_probs("DMSb", fsrc)]
    a_rb = dm.rule_view(fsrc, blocks, labs[convs[0]], 0)
    a_new = [x + v for x, v in zip(track, view)]
    smos = dm.team(src)
    cells = {}
    for s in es.SEEDS:
        res = arms(a_rb, a_new, view, a_lrn, a_m, smos[s])
        for name, preds in res.items():
            for c in convs:
                cells.setdefault((src, c, dm.TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        if s == es.SEEDS[0]:
            truth = labs[convs[0]]
            print("  %s: dead share true %.3f | %s" % (src, np.mean(np.concatenate(truth) == 2), " | ".join("%s %.3f F1 %.3f" % (n, np.mean(np.concatenate(res[n]) == 2), dm._dead_f1(res[n], truth)) for n in NAMES)), flush=True)
        print("  seed %d done" % s, flush=True)
    er.summarize(cells, list(NAMES), ref="v18")
    with open(os.path.join(bench.ROOT, "tools", "runs", "r43_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)


def arb_decide():
    """Pool the providers' cells and apply the decision rule among the candidates free of constants from the public window."""
    import exp_blend_weight as bw
    import pickle
    cells = {}
    for k in SOURCES:
        with open(os.path.join(bench.ROOT, "tools", "runs", "r43_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    mean, _ = er.summarize(cells, list(NAMES), ref="v18")
    for cand in NAMES[2:]:
        bw.decide(mean, "v18", cand)
    for cur in ("O.5", "O+DT"):
        for cand in NAMES[2:]:
            if cand != cur:
                bw.decide(mean, cur, cand)


def terms_view(f_like, conf, valid, live):
    t, b = conf["thresholds"], conf["bounds"]
    beta, gamma = np.asarray(conf["beta"]), np.asarray(conf["gamma"])
    return terms(f_like, t, b) @ beta, gamma[0] * (~valid) + gamma[1] * live


def arb_pff():
    """PFF's 64 matches as the independent test (as exp_deadmix.pff_test): the rule weights of the whole open fit, the v7 team LightGBM, transplanted flag."""
    import gradient
    import lightgbm as lgb
    import model_v2
    import model_v7
    import pathlib
    with open(OUT) as fh:
        conf = json.load(fh)
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    l_model = bench.load_booster(os.path.join(bench.ROOT, "models", "dead_base_v3.txt"))
    m_model = bench.load_booster(os.path.join(bench.CACHE, "dead", "dead_multi_v1.txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    import exp_smooth
    cells = {}
    n = 0
    F1 = {k: [0, 0, 0] for k in NAMES}
    for match in gradient.matches():
        blocks, labs, a_rb, a_new, a_view, a_l, a_m, smo = [], [], [], [], [], [], [], []
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                t0 = "h%d" % int(per)
                b = dict(P=d[t0 + "_players"], B=d[t0 + "_ball"], team=d["team"], gk=d["gk"], pitch=tuple(d["pitch"]), group=match)
                lab = np.asarray(d[t0 + "_labels"])
                idx = [f["fk"].index(k) for k in base_names]
                x = f["XF"][:, idx]
                rf = exp_blend.rule_f(b)
                v, l = dm.transplant(lab, np.random.default_rng([dm.SEED, n]))
                n += 1
                tr, vw = terms_view(rf, conf, v, l)
                blocks.append(b)
                labs.append(lab)
                a_rb.append(model_v2.aliveness(rf, dm.P) - dm.P["w_binvalid"] * (~v) + dm.P["w_blive"] * l)
                a_new.append(tr + vw)
                a_view.append(vw)
                a_l.append(exp_blend.logit_alive(l_model.predict(x, num_threads=bench.THREADS)))
                a_m.append(exp_blend.logit_alive(m_model.predict(x, num_threads=bench.THREADS)))
                smo.append(bench.predict_team(team_model, f))
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        res = arms(a_rb, a_new, a_view, a_l, a_m, smo)
        for name, preds in res.items():
            cells.setdefault(("pff", "ctrl", dm.TAG, name), []).append(er.by_match(blocks, labs, preds))
            for p, g in zip(preds, labs):
                F1[name][0] += int(np.sum((p == 2) & (g == 2)))
                F1[name][1] += int(np.sum((p == 2) & (g < 2)))
                F1[name][2] += int(np.sum((p != 2) & (g == 2)))
        print("  %s done" % match, flush=True)
    print("dead F1: " + " | ".join("%s %.3f (share %.3f)" % (k, 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1), 0.0) for k, v in F1.items()))
    merged = {}
    for k, v in cells.items():
        d2 = {}
        for x in v:
            d2.update(x)
        merged[k] = [d2]
    er.summarize(merged, list(NAMES), ref="v18")


def arb_st():
    """SoccerTrack v2: the rule weights of the whole open fit; its ball is rebuilt from the events, so the ball terms are unreliable there."""
    import exp_smooth
    import exp_st
    import short_specialist as ss
    with open(OUT) as fh:
        conf = json.load(fh)
    src = exp_st.SRC
    blocks = bench.blocks(src)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in exp_st.CONVS}
    truth = labs["evctrl"]
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % src.replace("@", "-")))
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", src)]
    a_m = [exp_blend.logit_alive(x) for x in dm.balanced_probs("DMSb", src)]
    a_rb = dm.rule_view(src, blocks, truth, 0)
    a_new, a_view = [], []
    for i, (b, g) in enumerate(zip(blocks, truth)):
        v, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        tr, vw = terms_view(exp_blend.rule_f(b), conf, v, l)
        a_new.append(tr + vw)
        a_view.append(vw)
    cells = {}
    for k in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
        smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
        res = arms(a_rb, a_new, a_view, a_lrn, a_m, smo)
        for name, preds in res.items():
            for c in exp_st.CONVS:
                cells.setdefault((src, c, dm.TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  seed %d done" % k, flush=True)
    er.summarize(cells, list(NAMES), ref="v18")


NAMES2 = ("v18", "M+V", "A25", "A50", "A33", "M+V pen20", "M+V pen45", "M+V min25", "M+V min75")


def evidence2(a_rb, track, view, a_lrn, a_m):
    """Round 44's candidates, free of constants from the public window. M+V: the multi-source model + the view terms. A: averages of calibrated log-odds + the view terms: A25 = 0.25 open rules + 0.75 model, A50 = half and half, A33 = open rules, DFL model and multi-source model in equal parts."""
    return {"v18": dm.mix_evidence(a_rb, a_lrn), "M+V": [m + v for m, v in zip(a_m, view)], "A25": [0.25 * t + 0.75 * m + v for t, m, v in zip(track, a_m, view)],
            "A50": [0.5 * t + 0.5 * m + v for t, m, v in zip(track, a_m, view)], "A33": [(t + l + m) / 3 + v for t, l, m, v in zip(track, a_lrn, a_m, view)]}


def arms2(a_rb, track, view, a_lrn, a_m, smo):
    ev = evidence2(a_rb, track, view, a_lrn, a_m)
    res = {k: sd.decode_scores(v, smo) for k, v in ev.items()}
    res["M+V pen20"] = sd.decode_scores(ev["M+V"], smo, pen_dead=20.0)
    res["M+V pen45"] = sd.decode_scores(ev["M+V"], smo, pen_dead=45.0)
    res["M+V min25"] = sd.decode_scores(ev["M+V"], smo, min_dead=25)
    res["M+V min75"] = sd.decode_scores(ev["M+V"], smo, min_dead=75)
    return res


def final_open(k):
    import pickle
    with open(OUT) as fh:
        conf = json.load(fh)
    fsrc, src, _ = SOURCES[k]
    convs = dm.CONVS[src]
    track, view = evidence(k, conf)
    blocks = bench.blocks(fsrc)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", fsrc)
    a_lrn = [exp_blend.logit_alive(x) for x in base]
    a_m = [exp_blend.logit_alive(x) for x in dm.balanced_probs("DMSb", fsrc)]
    a_rb = dm.rule_view(fsrc, blocks, labs[convs[0]], 0)
    smos = dm.team(src)
    cells = {}
    for s0 in es.SEEDS:
        res = arms2(a_rb, track, view, a_lrn, a_m, smos[s0])
        for name, preds in res.items():
            for c in convs:
                cells.setdefault((src, c, dm.TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        if s0 == es.SEEDS[0]:
            truth = labs[convs[0]]
            print("  %s: dead share true %.3f | %s" % (src, np.mean(np.concatenate(truth) == 2), " | ".join("%s %.3f F1 %.3f" % (n, np.mean(np.concatenate(res[n]) == 2), dm._dead_f1(res[n], truth)) for n in NAMES2[:5])), flush=True)
        print("  seed %d done" % s0, flush=True)
    er.summarize(cells, list(NAMES2), ref="v18")
    with open(os.path.join(bench.ROOT, "tools", "runs", "r44_cells_%s.pkl" % k), "wb") as fh:
        pickle.dump(cells, fh)
    # stress: rules better (kappa x the truth on both rule forms), learned channels worse (slow noise sigma, drawn apart)
    truth = labs[convs[0]]
    smo = smos[es.SEEDS[0]]
    out = {}
    names = NAMES2[:5]
    print("stress %s: kappa sigma | %s" % (src, " ".join(names)), flush=True)
    keep = list(range(20)) if k == "sc" else list(range(len(blocks)))
    b2 = [blocks[i] for i in keep]
    g2 = [truth[i] for i in keep]
    for kappa, sigma in ((0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (0, 4), (1, 4), (2, 4)):
        sg = [np.where(truth[i] == 2, -1.0, 1.0) for i in keep]
        l2 = [a_lrn[i] + (dm._ar1(len(a_lrn[i]), np.random.default_rng([dm.SEED, 31, i]), sigma) if sigma else 0.0) for i in keep]
        m2 = [a_m[i] + (dm._ar1(len(a_m[i]), np.random.default_rng([dm.SEED, 32, i]), sigma) if sigma else 0.0) for i in keep]
        rb2 = [a_rb[i] + kappa * x for i, x in zip(keep, sg)]
        t2 = [track[i] + kappa * x for i, x in zip(keep, sg)]
        v2 = [view[i] for i in keep]
        ev = evidence2(rb2, t2, v2, l2, m2)
        sc = {n: float(np.mean(list(er.by_match(b2, g2, sd.decode_scores(ev[n], [smo[i] for i in keep])).values()))) for n in names}
        out["%d %d" % (kappa, sigma)] = sc
        print("  %d %d | %s" % (kappa, sigma, " ".join("%.4f" % sc[n] for n in names)), flush=True)
    with open(os.path.join(bench.ROOT, "tools", "runs", "r44_stress_%s.json" % k), "w") as fh:
        json.dump(out, fh)


def final_decide():
    import exp_blend_weight as bw
    import pickle
    cells = {}
    for k in SOURCES:
        with open(os.path.join(bench.ROOT, "tools", "runs", "r44_cells_%s.pkl" % k), "rb") as fh:
            cells.update(pickle.load(fh))
    mean, _ = er.summarize(cells, list(NAMES2), ref="v18")
    for cand in NAMES2[1:5]:
        bw.decide(mean, "v18", cand)
    for cand in NAMES2[2:]:
        bw.decide(mean, "M+V", cand)
    names = NAMES2[:5]
    reg = {n: {} for n in names}
    for k in SOURCES:
        with open(os.path.join(bench.ROOT, "tools", "runs", "r44_stress_%s.json" % k)) as fh:
            d = json.load(fh)
        for sc_name, v in d.items():
            best = max(v[n] for n in names)
            for n in names:
                reg[n][(k, sc_name)] = best - v[n]
    print("max regret over stress: " + " | ".join("%s %.4f" % (n, max(reg[n].values())) for n in names))
    for lim in (0, 2):
        sel = lambda key: int(key[1].split()[1]) <= lim
        print("max regret, sigma <= %d: " % lim + " | ".join("%s %.4f" % (n, max(v for key, v in reg[n].items() if sel(key))) for n in names), flush=True)


def final_pff():
    import exp_smooth
    import gradient
    import lightgbm as lgb
    import model_v2
    import model_v7
    import pathlib
    with open(OUT) as fh:
        conf = json.load(fh)
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    l_model = bench.load_booster(os.path.join(bench.ROOT, "models", "dead_base_v3.txt"))
    m_model = bench.load_booster(os.path.join(bench.CACHE, "dead", "dead_multi_v1.txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    cells = {}
    n = 0
    for match in gradient.matches():
        blocks, labs, a_rb, tr, vw, a_l, a_m, smo = [], [], [], [], [], [], [], []
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                t0 = "h%d" % int(per)
                b = dict(P=d[t0 + "_players"], B=d[t0 + "_ball"], team=d["team"], gk=d["gk"], pitch=tuple(d["pitch"]), group=match)
                lab = np.asarray(d[t0 + "_labels"])
                idx = [f["fk"].index(k) for k in base_names]
                x = f["XF"][:, idx]
                rf = exp_blend.rule_f(b)
                v, l = dm.transplant(lab, np.random.default_rng([dm.SEED, n]))
                n += 1
                t1, v1 = terms_view(rf, conf, v, l)
                blocks.append(b)
                labs.append(lab)
                a_rb.append(model_v2.aliveness(rf, dm.P) - dm.P["w_binvalid"] * (~v) + dm.P["w_blive"] * l)
                tr.append(t1)
                vw.append(v1)
                a_l.append(exp_blend.logit_alive(l_model.predict(x, num_threads=bench.THREADS)))
                a_m.append(exp_blend.logit_alive(m_model.predict(x, num_threads=bench.THREADS)))
                smo.append(bench.predict_team(team_model, f))
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        res = arms2(a_rb, tr, vw, a_l, a_m, smo)
        for name, preds in res.items():
            cells.setdefault(("pff", "ctrl", dm.TAG, name), []).append(er.by_match(blocks, labs, preds))
        print("  %s done" % match, flush=True)
    merged = {}
    for k, v in cells.items():
        d2 = {}
        for x in v:
            d2.update(x)
        merged[k] = [d2]
    er.summarize(merged, list(NAMES2), ref="v18")


def final_st():
    import exp_smooth
    import exp_st
    import short_specialist as ss
    with open(OUT) as fh:
        conf = json.load(fh)
    src = exp_st.SRC
    blocks = bench.blocks(src)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in exp_st.CONVS}
    truth = labs["evctrl"]
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % src.replace("@", "-")))
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", src)]
    a_m = [exp_blend.logit_alive(x) for x in dm.balanced_probs("DMSb", src)]
    a_rb = dm.rule_view(src, blocks, truth, 0)
    tr, vw = [], []
    for i, (b, g) in enumerate(zip(blocks, truth)):
        v, l = dm.transplant(g, np.random.default_rng([dm.SEED, 0, i]))
        t1, v1 = terms_view(exp_blend.rule_f(b), conf, v, l)
        tr.append(t1)
        vw.append(v1)
    cells = {}
    for k in es.SEEDS:
        raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
        smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
        res = arms2(a_rb, tr, vw, a_lrn, a_m, smo)
        for name, preds in res.items():
            for c in exp_st.CONVS:
                cells.setdefault((src, c, dm.TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  seed %d done" % k, flush=True)
    er.summarize(cells, list(NAMES2), ref="v18")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"fit": fit, "eval": evaluate, "arb": arb_open, "decide": arb_decide, "pff": arb_pff, "st": arb_st, "final": final_open, "fdecide": final_decide, "fpff": final_pff, "fst": final_st}[sys.argv[1]](*sys.argv[2:])
