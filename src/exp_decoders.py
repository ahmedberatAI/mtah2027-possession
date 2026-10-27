"""Decoder ideas (docs/RAPOR.md B1, B2, B4, B5, B6, G1, G2) on cached bench probabilities. AGGREGATE output
only (scores, counts, shares); nothing per frame is printed.

Data: DFL leave-one-match-out team probabilities (cache/bench/oof/base_native[_v3].pkl) + DFL dead OOF
(bench.dead_probs), Metrica = the all-DFL team model (cache/bench/models/base_native[_v3].txt) + all-DFL dead
model. Targets: DFL vs trk (controlled possession) and native labels, Metrica vs ctrl and events labels.
Decoder bases: DFL "dflcv" (pen_dead 32, min 5 / 25), Metrica "v5" (pen_dead 30, min 8 / 50).

  python -I src/exp_decoders.py prep [var]          # cache probabilities, labels, context cues
  python -I src/exp_decoders.py check [var]         # decode2.std_spec == decode.viterbi_min_dur
  python -I src/exp_decoders.py <idea> [var]        # base | b4 | b5 | b1stats | b1 | b2 | b6 | g12
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
import bench  # noqa: E402
import decode2  # noqa: E402
import train_dfl as td  # noqa: E402

pm = td.pm
H, A, D, X = 0, 1, 2, 3
FPS = 25
OUT = os.path.join(ROOT, "cache", "bench", "dec")
REP = os.path.join(ROOT, "reports", "bench")
PENS = (4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96)
BASE = {"dfl": dict(pen_dead=32.0, min_poss=5, min_dead=25), "metrica": dict(pen_dead=30.0, min_poss=8, min_dead=50)}
TARGETS = {"dfl": ("trk", "native"), "metrica": ("ctrl", "events")}
PROCS = int(os.environ.get("DEC_PROCS", "3"))


def _tag(var):
    return "base_native" if var == "v2" else "base_native_" + var


# -- data -------------------------------------------------------------------------------------------

def context(b):
    """Per-frame cues of one degraded block: ball seen, nearest-player distance, ball position carried
    forward / backward with its age (metres; pitch corner origin)."""
    L, W = b["pitch"]
    bxy = b["B"][:, :2].astype(np.float64) * np.array([L, W])
    pxy = b["P"][:, :, :2].astype(np.float64) * np.array([L, W])
    seen = np.isfinite(bxy[:, 0])
    n = len(seen)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf).min(1)
    dmin = np.where(seen & np.isfinite(d), d, np.nan)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    lastxy = np.where((last >= 0)[:, None], bxy[np.maximum(last, 0)], np.nan)
    nextxy = np.where((nxt < n)[:, None], bxy[np.minimum(nxt, n - 1)], np.nan)
    return {"seen": seen, "dmin": dmin.astype(np.float32), "last_age": np.where(last >= 0, idx - last, 10 ** 6),
            "next_age": np.where(nxt < n, nxt - idx, 10 ** 6), "lastxy": lastxy.astype(np.float32),
            "nextxy": nextxy.astype(np.float32), "pitch": (float(L), float(W))}


def prep(var="v2"):
    import lightgbm as lgb
    out = {}
    for base in ("dfl", "metrica"):
        src = bench.src_name(base, var)
        bl = bench.blocks(src)
        pdl = bench.dead_probs(src)
        if base == "dfl":
            phl = bench._load(os.path.join(ROOT, "cache", "bench", "oof", _tag(var) + ".pkl"))
        else:
            model = lgb.Booster(model_file=os.path.join(ROOT, "cache", "bench", "models", _tag(var) + ".txt"))
            phl = [bench.predict_team(model, f) for f in bench.features("base", src)]
        out[base] = {"pd": [np.asarray(x, np.float64) for x in pdl], "ph": [np.asarray(x, np.float64) for x in phl],
                     "gt": {c: bench.labels(src, c) for c in TARGETS[base]},
                     "groups": [(b["group"], b.get("seed", 0)) for b in bl], "ctx": [context(b) for b in bl]}
    bench._dump(out, os.path.join(OUT, "data_%s.pkl" % var))
    print("prepared", var)


_D = {}


def data(var):
    if var not in _D:
        _D[var] = bench._load(os.path.join(OUT, "data_%s.pkl" % var))
    return _D[var]


# -- scoring ----------------------------------------------------------------------------------------

def fast_score(gts, preds):
    """td.spell_score without the transition / bootstrap metrics: (spellF1, macroF1, n_gt, n_pred)."""
    sep = np.full(10, 3, np.int8)
    g = np.concatenate([np.concatenate([a, sep]) for a in gts]).astype(np.int8)
    p = np.concatenate([np.concatenate([a, sep]) for a in preds]).astype(np.int8)
    runs = pm.valid_runs(g)
    sp = pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)
    return sp["mean_f1"], pm.macro_f1(pm.scored_confusion(pm.confusion(g, p))), sp["n_gt"], sp["n_pred"]


def score_preds(d, preds, conv):
    groups = {}
    for k, g, p in zip(d["groups"], d["gt"][conv], preds):
        groups.setdefault(k, ([], []))
        groups[k][0].append(g)
        groups[k][1].append(p)
    arr = np.array([fast_score(*v) for v in groups.values()])
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), int(arr[:, 2].sum()), int(arr[:, 3].sum())


# -- decoders ---------------------------------------------------------------------------------------

def emissions(pd_, ph, temp_t=1.0):
    return td.emissions(pd_, ph, 1.0, temp_t)


def chan_team(ctx, cfg):
    """B5: the H <-> A cost is pen_team * (1 + w_unseen [ball unseen] + w_far [seen, > far_m from every player]
    - w_near [seen within 1.5 m of a player]); returned as the additive part, in units of pen_team."""
    c = np.zeros(len(ctx["seen"]))
    if cfg.get("w_unseen"):
        c += cfg["w_unseen"] * ~ctx["seen"]
    if cfg.get("w_far"):
        far = np.nan_to_num(ctx["dmin"], nan=0.0) > cfg.get("far_m", 3.0)
        c += cfg["w_far"] * far
    if cfg.get("w_near"):
        near = np.nan_to_num(ctx["dmin"], nan=99.0) <= 1.5
        c -= cfg["w_near"] * near
    return c


def restart_ctx(ctx, max_age=50):
    """Context of a restart at frame t from the ball nearest in time (seen within max_age frames before or
    after): 0 side line (< 3 m), 1 goal line (< 6 m, not side), 2 inside, 3 unknown."""
    L, W = ctx["pitch"]
    use_next = ctx["next_age"] <= np.minimum(ctx["last_age"], max_age)
    xy = np.where(use_next[:, None], ctx["nextxy"], ctx["lastxy"])
    ok = use_next | (ctx["last_age"] <= max_age)
    x, y = xy[:, 0], xy[:, 1]
    side = np.minimum(y, W - y) < 3.0
    goal = (np.minimum(x, L - x) < 6.0) & ~side
    cat = np.where(side, 0, np.where(goal, 1, 2))
    return np.where(ok, cat, 3).astype(np.int8)


def _has_b5(cfg):
    return any(cfg.get(k) for k in ("w_unseen", "w_far", "w_near"))


def build_spec(cfg, base):
    p = dict(BASE[base])
    p.update({k: v for k, v in cfg.items() if k in ("pen_dead", "min_poss", "min_dead")})
    kind = cfg.get("kind", "std")
    pt = cfg["pen_team"]
    if kind in ("std", "b4", "b5", "post"):
        return decode2.std_spec(pt, p["pen_dead"], p["min_poss"], p["min_dead"], cfg.get("pen_in"), cfg.get("pen_out"),
                                ch_team=0 if kind == "b5" else -1)
    if kind == "b1":
        po = cfg.get("pen_out", p["pen_dead"])
        return decode2.restart_spec(pt, cfg.get("pen_in", p["pen_dead"]), po + cfg.get("c_same", 0.0),
                                    po + cfg.get("c_switch", 0.0), p["min_poss"], p["min_dead"],
                                    ch_restart=bool(cfg.get("restart_tab")), ch_team=-1)
    if kind == "b2":
        D0 = int(cfg["D0"])
        j = np.arange(D0)
        ramp = np.clip(1.0 - (j + 1) / float(D0), 0, None) ** cfg.get("power", 1.0)
        dead_exit = None
        if cfg.get("D0_dead"):
            Dd = int(cfg["D0_dead"])
            jd = np.arange(Dd)
            dead_exit = cfg["c_dead"] * np.clip(1.0 - (jd + 1) / float(Dd), 0, None)
        return decode2.duration_spec(pt, p["pen_dead"], p["min_poss"], p["min_dead"], cfg["c_short"] * ramp,
                                     dead_exit=dead_exit, ch_team=0 if _has_b5(cfg) else -1)
    raise ValueError(kind)


def decode_block(cfg, base, pd_, ph, ctx, spec):
    S = emissions(pd_, ph, cfg.get("temp_t", 1.0))
    kind = cfg.get("kind", "std")
    chan = None
    if kind == "b5" or (kind == "b2" and _has_b5(cfg)):
        chan = (cfg["pen_team"] * chan_team(ctx, cfg))[:, None]
    elif kind == "b1" and cfg.get("restart_tab"):
        tab = cfg["restart_tab"]          # {category: (cost same, cost switch)}
        cat = restart_ctx(ctx)
        chan = np.zeros((len(S), 2))
        for c, (cs, cw) in tab.items():
            m = cat == int(c)
            chan[m, 0] = cs
            chan[m, 1] = cw
    lab = spec.decode(S, chan)
    if kind == "post" or cfg.get("post"):
        lab = postprocess(lab, cfg.get("post", cfg))
    if cfg.get("shift_json"):
        lab = shift_boundaries(lab, {tuple(int(x) for x in k.split(",")): v for k, v in cfg["shift_json"].items()})
    return lab.astype(np.int8)


def segments(lab):
    br = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], br, [len(lab)]))
    return [(int(a), int(b), int(lab[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def postprocess(lab, cfg):
    """G1: an H/A spell shorter than g1_s between two spells of the other team (no D between) joins them.
    G2: a D segment shorter than g2_s between two spells of the same team joins them."""
    out = lab.copy()
    for key, kind in (("g1_s", "g1"), ("g2_s", "g2")):
        tau = cfg.get(key)
        if not tau:
            continue
        segs = segments(out)
        for i in range(1, len(segs) - 1):
            a, b, k = segs[i]
            ka, kb = segs[i - 1][2], segs[i + 1][2]
            if b - a >= tau * FPS or ka != kb or ka == D:
                continue
            if (kind == "g1" and k != D and k != ka) or (kind == "g2" and k == D):
                out[a:b] = ka
    return out


def shift_boundaries(lab, shift):
    """B6: move every boundary of type (from, to) by shift[(from, to)] frames (positive = later)."""
    segs = segments(lab)
    if len(segs) < 2:
        return lab
    starts = np.array([s[0] for s in segs[1:]])
    new = starts.copy()
    for i in range(len(starts)):
        k0, k1 = segs[i][2], segs[i + 1][2]
        new[i] = starts[i] + int(shift.get((k0, k1), 0))
    lo = np.concatenate(([1], new[:-1] + 1))
    hi = np.concatenate((new[1:] - 1, [len(lab) - 1]))
    new = np.clip(new, lo, hi)
    out = np.empty_like(lab)
    edges = np.concatenate(([0], new, [len(lab)]))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        out[a:b] = segs[i][2]
    return out


# -- evaluation -------------------------------------------------------------------------------------

def _task(job):
    var, cfg, pen = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dd = data(var)
    res = {}
    for base in ("dfl", "metrica"):
        c = dict(cfg, pen_team=pen)
        spec = build_spec(c, base)
        d = dd[base]
        preds = [decode_block(c, base, pd_, ph, ctx, spec) for pd_, ph, ctx in zip(d["pd"], d["ph"], d["ctx"])]
        for conv in TARGETS[base]:
            res["%s|%s" % (base, conv)] = score_preds(d, preds, conv)
    return json.dumps(cfg, sort_keys=True), pen, res


def evaluate(var, cfgs, pens=PENS, procs=PROCS):
    """{cfg_json: {target: {pen: (spellF1, macroF1, n_gt, n_pred)}}}"""
    jobs = [(var, c, p) for c in cfgs for p in pens]
    out = {}
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            it = pool.imap_unordered(_task, jobs)
            rows = list(it)
    else:
        rows = [_task(j) for j in jobs]
    for key, pen, res in rows:
        for tgt, v in res.items():
            out.setdefault(key, {}).setdefault(tgt, {})[pen] = v
    return out


def summarise(out, name, var, ref=None):
    """Lines: per cfg and target the best pen_team / spellF1, and values at the fixed pens 32 and 48."""
    lines = []
    tg = ("metrica|ctrl", "dfl|trk", "dfl|native", "metrica|events")
    lines.append("%-62s | %s" % ("variant (%s, sim %s)" % (name, var), " | ".join(
        "%-26s" % t for t in tg)))
    best_rows = []
    for key, r in out.items():
        cells = []
        for t in tg:
            c = r[t]
            b = max(c, key=lambda k: c[k][0])
            cells.append("best %2d %.4f @32 %.4f @48 %.4f" % (b, c[b][0], c.get(32, (np.nan,))[0],
                                                              c.get(48, (np.nan,))[0]))
        lines.append("%-62s | %s" % (key[:62], " | ".join(cells)))
        best_rows.append((key, {t: max(r[t][k][0] for k in r[t]) for t in tg}))
    if ref is not None:
        lines.append("delta of best vs reference (%s):" % ref[0])
        for key, b in best_rows:
            lines.append("  %-60s %s" % (key[:60], "  ".join("%s %+.4f" % (t, b[t] - ref[1][t]) for t in tg)))
    return lines, best_rows


def write(lines, name, var, out=None):
    os.makedirs(REP, exist_ok=True)
    path = os.path.join(REP, "decoders_%s%s.txt" % (name, "" if var == "v2" else "_" + var))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if out is not None:
        with open(path.replace(".txt", ".json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=0)
    print("\n".join(lines), flush=True)


def reference(var):
    path = os.path.join(REP, "decoders_base%s.json" % ("" if var == "v2" else "_" + var))
    out = json.load(open(path, encoding="utf-8"))
    key = json.dumps({"kind": "std"}, sort_keys=True)
    r = out[key]
    return ("std", {t: max(v[0] for v in r[t].values()) for t in r})


# -- ideas ------------------------------------------------------------------------------------------

def check(var):
    from decode import viterbi_min_dur
    d = data(var)
    for base in ("dfl", "metrica"):
        p = BASE[base]
        spec = decode2.std_spec(20.0, p["pen_dead"], p["min_poss"], p["min_dead"])
        same = 0
        for pd_, ph in zip(d[base]["pd"][:4], d[base]["ph"][:4]):
            S = emissions(pd_, ph)
            pen = np.array([[0, 20, p["pen_dead"]], [20, 0, p["pen_dead"]], [p["pen_dead"], p["pen_dead"], 0]], float)
            a = viterbi_min_dur(S, pen, np.array([p["min_poss"], p["min_poss"], p["min_dead"]]))
            b = spec.decode(S)
            same += int(np.array_equal(a, b))
        print(base, "std_spec identical to decode.viterbi_min_dur on %d/4 blocks" % same)
        # fast_score == td.spell_score
        gts, preds = d[base]["gt"][TARGETS[base][0]][:2], [decode2.std_spec(20.0, p["pen_dead"], p["min_poss"],
                                                                             p["min_dead"]).decode(
            emissions(pd_, ph)).astype(np.int8) for pd_, ph in zip(d[base]["pd"][:2], d[base]["ph"][:2])]
        print(base, "fast_score", np.round(fast_score(gts, preds), 6), "td.spell_score",
              np.round(td.spell_score(gts, preds), 6))


def idea_base(var):
    cfgs = [{"kind": "std"}, {"kind": "std", "temp_t": 0.5}, {"kind": "std", "temp_t": 1.5}]
    out = evaluate(var, cfgs)
    lines, _ = summarise(out, "base", var)
    write(lines, "base", var, out)


def idea_b4(var):
    cfgs = []
    for pin in (16, 24, 32, 48, 64):
        for pout in (16, 24, 32, 48, 64):
            if pin == pout == 32:
                continue
            cfgs.append({"kind": "b4", "pen_in": pin, "pen_out": pout})
    out = evaluate(var, cfgs, pens=(16, 24, 32, 48, 64))
    lines, _ = summarise(out, "b4", var, reference(var))
    write(lines, "b4", var, out)


def idea_b5(var):
    cfgs = []
    for wu in (0.0, 0.25, 0.5, 1.0, 2.0):
        for wf in (0.0, 0.25, 0.5, 1.0):
            if wu == wf == 0:
                continue
            cfgs.append({"kind": "b5", "w_unseen": wu, "w_far": wf})
    cfgs += [{"kind": "b5", "w_near": w} for w in (0.25, 0.5, 0.75)]
    cfgs += [{"kind": "b5", "w_near": 0.5, "w_unseen": w} for w in (0.5, 1.0)]
    out = evaluate(var, cfgs)
    lines, _ = summarise(out, "b5", var, reference(var))
    write(lines, "b5", var, out)


def b1stats(var):
    """How often does the team that restarts after a stoppage equal the team that had the ball before it?
    Overall and by restart context (ball nearest in time to the restart; degraded data = what the decoder
    sees). Labels: DFL trk / native, Metrica ctrl / events."""
    dd = data(var)
    lines = ["restart team == pre-stoppage team (internal D segments between two alive spells)"]
    tabs = {}
    for base in ("dfl", "metrica"):
        d = dd[base]
        for conv in TARGETS[base]:
            cnt = np.zeros((4, 2), int)
            for lab, ctx, grp in zip(d["gt"][conv], d["ctx"], d["groups"]):
                if base == "metrica" and grp[1] != 0:
                    continue
                cat = restart_ctx(ctx)
                segs = segments(lab)
                for i in range(1, len(segs) - 1):
                    a, b, k = segs[i]
                    if k != D:
                        continue
                    k0, k1 = segs[i - 1][2], segs[i + 1][2]
                    if k0 not in (H, A) or k1 not in (H, A):
                        continue
                    cnt[cat[b], int(k0 == k1)] += 1
            tot = cnt.sum()
            names = ("side line", "goal line", "inside", "no ball")
            lines.append("%-8s %-7s n %4d  same %.3f | " % (base, conv, tot, cnt[:, 1].sum() / max(1, tot)) + " | ".join(
                "%s n %d same %.3f" % (nm, cnt[c].sum(), cnt[c, 1] / max(1, cnt[c].sum())) for c, nm in enumerate(names)))
            tabs["%s|%s" % (base, conv)] = cnt.tolist()
    write(lines, "b1stats", var, tabs)
    return tabs


LAMS = tuple(float(x) for x in os.environ.get("B1_LAMS", "1,3,6").split(","))


def idea_b1(var):
    tabs = b1stats(var)
    cfgs = []
    # global prior: restarting team = previous team costs c_same, the other c_switch (log-odds scaled by lam)
    for p_same_src in ("dfl|trk", "metrica|ctrl"):
        cnt = np.array(tabs[p_same_src])
        ps = cnt[:, 1].sum() / cnt.sum()
        for lam in LAMS:
            cfgs.append({"kind": "b1", "c_same": -lam * np.log(2 * ps), "c_switch": -lam * np.log(2 * (1 - ps)),
                         "src": p_same_src, "lam": lam})
            # context table: per restart category
            tab = {}
            for c in range(4):
                n = cnt[c].sum()
                pc = (cnt[c, 1] + 1) / (n + 2)
                tab[str(c)] = (-lam * np.log(2 * pc), -lam * np.log(2 * (1 - pc)))
            cfgs.append({"kind": "b1", "restart_tab": tab, "src": p_same_src, "lam": lam})
    for c in cfgs:
        for k in ("c_same", "c_switch"):
            if k in c:
                c[k] = round(float(c[k]), 3)
        if "restart_tab" in c:
            c["restart_tab"] = {k: (round(float(v[0]), 3), round(float(v[1]), 3)) for k, v in c["restart_tab"].items()}
    cfgs.append({"kind": "b1"})   # no prior: must equal std
    out = evaluate(var, cfgs)
    lines, _ = summarise(out, "b1", var, reference(var))
    write(lines, "b1", var, out)


def idea_b2(var):
    cfgs = []
    for D0 in (25, 50, 75, 125):
        for c in (4.0, 8.0, 16.0, 32.0):
            cfgs.append({"kind": "b2", "D0": D0, "c_short": c})
    cfgs += [{"kind": "b2", "D0": 75, "c_short": 16.0, "power": 2.0}, {"kind": "b2", "D0": 125, "c_short": 32.0,
                                                                          "power": 2.0}]
    out = evaluate(var, cfgs)
    lines, _ = summarise(out, "b2", var, reference(var))
    write(lines, "b2", var, out)


def idea_b2b(var):
    """Longer soft-minimum ramps, and B2 + B5 combined."""
    cfgs = []
    for D0 in (125, 175, 250):
        for c in (24.0, 32.0, 48.0, 64.0):
            for pw in (1.0, 2.0):
                cfgs.append({"kind": "b2", "D0": D0, "c_short": c, "power": pw})
    for D0, c in ((125, 32.0), (175, 48.0)):
        for wf, wu in ((1.0, 0.25), (1.0, 0.5), (2.0, 0.5)):
            cfgs.append({"kind": "b2", "D0": D0, "c_short": c, "w_far": wf, "w_unseen": wu})
    cfgs += [{"kind": "b5", "w_far": 2.0, "w_unseen": 0.5}, {"kind": "b5", "w_far": 4.0, "w_unseen": 0.5},
             {"kind": "b5", "w_far": 2.0}, {"kind": "b5", "w_far": 4.0}]
    out = evaluate(var, cfgs)
    lines, _ = summarise(out, "b2b", var, reference(var))
    write(lines, "b2b", var, out)


def idea_b2c(var):
    """Finer soft-minimum grid around the V2 / V3 optimum."""
    cfgs = []
    for D0 in (150, 175, 250, 375):
        for c in (32.0, 48.0, 64.0, 96.0):
            for pw in (1.0, 2.0):
                cfgs.append({"kind": "b2", "D0": D0, "c_short": c, "power": pw})
    out = evaluate(var, cfgs, pens=(2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48))
    lines, _ = summarise(out, "b2c", var, reference(var))
    write(lines, "b2c", var, out)


def barath(var="v2"):
    """BAR-ATH: model_v5's scores (v3 rule alive/dead + bench team probabilities) decoded with the standard
    decoder and with the B2 soft minimum; public window + full match, aggregates only."""
    import atd
    import evaluate as ev
    import model_v5
    import proxy
    from make_submission import merged_features
    f = merged_features()
    lines = []
    for tag in ("base_native", "base_native_v3"):
        path = os.path.join(ROOT, "cache", "bench", "barath_probs_%s.pkl" % tag)
        if not os.path.isfile(path):
            continue
        model_v5._probs.clear()
        model_v5.team_logp(f, path)
        p = dict(model_v5.DEFAULT)
        S = model_v5.scores(f, p)
        lines.append("team probabilities %s (model_v5 scores; decoder base pen_dead %g, min %d / %d)" % (
            tag, p["pen_dead"], p["min_poss"], p["min_dead"]))
        for name, cfg in (("std", {"kind": "std"}), ("b2 D0 175 c 48", {"kind": "b2", "D0": 175, "c_short": 48.0}),
                          ("b2 D0 250 c 64 p2", {"kind": "b2", "D0": 250, "c_short": 64.0, "power": 2.0})):
            for pt in (4, 8, 12, 16, 20, 32):
                c = dict(cfg, pen_team=float(pt), pen_dead=p["pen_dead"], min_poss=p["min_poss"], min_dead=p["min_dead"])
                spec = build_spec(c, "metrica")
                lab = np.full(f["n"], 3, np.int8)
                for _, a, b in atd.HALVES:
                    lab[a:b + 1] = spec.decode(S[a:b + 1])
                same = ""
                if name == "std":
                    same = " (== model_v5.predict: %s)" % np.array_equal(lab, model_v5.predict(f, {"pen_team": float(pt)}))
                sc, dg, px = ev.score(lab), ev.diagnostics(lab), proxy.proxies(lab, f)
                g = lab[atd.graded_mask(len(lab))]
                alive_min = np.sum(g < 2) / FPS / 60
                lines.append("  %-18s pen_team %2d | public spellF1 %.4f macroF1 %.4f team_acc %.4f n_pred %d/%d | full "
                             "n_poss %d (%.2f/alive min) median %.1f s <2s %d | tag_team %.4f setpiece %.3f%s" % (
                                 name, pt, sc["spellF1"], sc["macroF1"], sc["team_acc"], sc["n_pred"], sc["n_gt"],
                                 dg["n_poss"], dg["n_poss"] / alive_min, dg["poss_median_s"], dg["poss_lt2s"],
                                 px["tag_team_agree"], px["setpiece_ok"], same))
        model_v5._probs.clear()
    write(lines, "barath", "v2")


def idea_b2d(var):
    """B2 possession soft minimum (175 / 48) plus a soft minimum for dead-ball segments."""
    cfgs = [{"kind": "b2", "D0": 175, "c_short": 48.0}]
    for Dd in (50, 100, 150):
        for cd in (8.0, 16.0, 32.0):
            cfgs.append({"kind": "b2", "D0": 175, "c_short": 48.0, "D0_dead": Dd, "c_dead": cd})
    out = evaluate(var, cfgs, pens=(4, 8, 12, 16, 20, 24, 32))
    lines, _ = summarise(out, "b2d", var, reference(var))
    write(lines, "b2d", var, out)


def lag_stats(var, pens=None):
    """Median latency (pred - GT frame) of matched transitions per (from, to) type at each base's best pen."""
    dd = data(var)
    ref = json.load(open(os.path.join(REP, "decoders_base%s.json" % ("" if var == "v2" else "_" + var)),
                         encoding="utf-8"))[json.dumps({"kind": "std"}, sort_keys=True)]
    lines = ["matched transitions (typed 'to', tol 25 frames): latency pred - GT, frames"]
    shifts = {}
    for base in ("dfl", "metrica"):
        d = dd[base]
        for conv in TARGETS[base]:
            c = ref["%s|%s" % (base, conv)]
            pen = int(max(c, key=lambda k: c[k][0]))
            spec = build_spec({"pen_team": pen}, base)
            lat = {}
            for pd_, ph, ctx, gt in zip(d["pd"], d["ph"], d["ctx"], d["gt"][conv]):
                lab = decode_block({"pen_team": pen}, base, pd_, ph, ctx, spec)
                runs = pm.valid_runs(gt)
                gtt = pm.transitions(gt, runs)
                prt = pm.transitions(pm.fill_abstentions(lab, runs), runs)
                pairs, _, _ = pm.match_transitions(gtt, prt, 25, "pair")
                for g, p_ in pairs:
                    lat.setdefault((g[1], g[2]), []).append(p_[0] - g[0])
            sh = {}
            cells = []
            for key in sorted(lat):
                v = np.array(lat[key])
                cells.append("%s->%s n %d med %+d mean %+.1f" % ("HADX"[key[0]], "HADX"[key[1]], len(v), np.median(v),
                                                                v.mean()))
                sh[key] = -int(np.median(v))
            shifts["%s|%s" % (base, conv)] = (pen, sh)
            lines.append("%-8s %-7s pen %d | %s" % (base, conv, pen, " | ".join(cells)))
    return lines, shifts


def idea_b6(var):
    lines, shifts = lag_stats(var)
    cfgs = []
    for src in ("dfl|trk", "metrica|ctrl"):
        pen, sh = shifts[src]
        for scale in (0.5, 1.0):
            s = {"%d,%d" % k: int(round(v * scale)) for k, v in sh.items() if v != 0}
            cfgs.append({"kind": "std", "shift_src": src, "scale": scale, "shift_json": s})
    out = evaluate(var, cfgs)
    l2, _ = summarise(out, "b6", var, reference(var))
    write(lines + l2, "b6", var)


def idea_g12(var):
    cfgs = []
    for g1 in (0, 1, 2, 3, 4):
        for g2 in (0, 1, 2, 3):
            if g1 == g2 == 0:
                continue
            cfgs.append({"kind": "post", "g1_s": g1, "g2_s": g2})
    out = evaluate(var, cfgs)
    lines, _ = summarise(out, "g12", var, reference(var))
    write(lines, "g12", var, out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd = sys.argv[1]
    var = sys.argv[2] if len(sys.argv) > 2 else "v2"
    t0 = time.time()
    {"prep": prep, "check": check, "base": idea_base, "b4": idea_b4, "b5": idea_b5, "b1stats": b1stats,
     "b1": idea_b1, "b2": idea_b2, "b2b": idea_b2b, "b2c": idea_b2c, "b2d": idea_b2d, "barath": barath, "b6": idea_b6, "g12": idea_g12}[cmd](var)
    print("[%s %s done in %.0f s]" % (cmd, var, time.time() - t0))
