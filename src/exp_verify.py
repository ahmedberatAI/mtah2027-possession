"""Round 15, capability 2: a short-possession verifier trained across all external sources. AGGREGATE output only.

Candidates come from v10's own output (src/exp_short_cands.py): "ins" = runs >= 12 frames inside a decoded possession
where the +-25 probability favours the other team (flips the decoder suppressed), "del" = decoded possessions < 4 s.
Each gets deployable features (src/exp_short_angles.py: ball visibility / speed / flight, nearest-player shares and
distances, touches, distinct holders, longest hold, contest, depth, touchline distance, team-centroid reaction, v10
margins; time since the last DECODED restart; enclosing / neighbouring decoded spell lengths) and, in the "+G" variant,
the margin of the shared BiGRU (equal mean of its three members; Metrica and SkillCorner only, DFL left missing because
the BiGRU trained on all DFL). Label: positive when >= 50% of the candidate frames are the candidate team's and a true
spell of that team overlaps it with IoU >= 0.3, negative when < 20% are.
The verifier is a LightGBM pooled over DFL (trk labels), Metrica (ctrl) and SkillCorner (phase); every match is
predicted by a model that never saw it (leave-one-match-out over 30 matches), separately for each v10 seed, dead
channel and candidate type. Decisions: insert an "ins" candidate when p >= t_ins (both remaining parts of the enclosing
possession >= the decoder minimum, or empty); hand a "del" possession to the other team when p < t_del and that team
holds a neighbour (between two of its spells, or next to a stoppage). Judged with src/exp_refine.py's protocol.

  python -I src/exp_verify.py build [src]                 candidates (cache/bench/short15/)
  python -I src/exp_verify.py train SEED DEAD(0|1) VSEED   leave-one-match-out probabilities
  python -I src/exp_verify.py trainpool SEED VSEED          same, both dead channels pooled (as shipped)
  python -I src/exp_verify.py eval VSEED|pVSEED            decision variants, table and regret
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_short_cands as ec  # noqa: E402
import exp_smooth  # noqa: E402

SOURCES = (("dfl@v3", ("trk", "evctrl"), (32.0, 5, 25), "trk"), ("metrica@v3", ("ctrl",), (30.0, 8, 50), "ctrl"),
           ("sc@v3", ("phase", "pp"), (30.0, 8, 50), "phase"))
BASE = [f for f in ea.FEATS if f not in ("since_restart", "m_gru")] + [
    "since_dead", "enc_len", "prev_len", "next_len", "prev_dead", "next_dead"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0)
ROUNDS = 300
VARIANTS = {"V": (False, True, True, 0.5, 0.5), "V+G": (True, True, True, 0.5, 0.5),
            "V+G ins": (True, True, False, 0.5, 0.5), "V+G del": (True, False, True, 0.5, 0.5),
            "V+G strict": (True, True, True, 0.65, 0.35),
            # robustness set (added after the first run): neighbouring thresholds, strict without the BiGRU margin,
            # strict halves
            "V strict": (False, True, True, 0.65, 0.35), "V+G 0.6/0.4": (True, True, True, 0.6, 0.4),
            "V+G 0.7/0.3": (True, True, True, 0.7, 0.3), "V+G strict ins": (True, True, False, 0.65, 0.35),
            "V+G strict del": (True, False, True, 0.65, 0.35)}
CACHE = os.path.join(bench.CACHE, "short15")
DEADS = ("learned D", "true D")


def gru_mean(src):
    if src.startswith("dfl"):
        return None
    m = es.bigru_members(src)
    return [np.mean([x[i] for x in m], axis=0) for i in range(len(m[0]))]


BASE2 = BASE + ["adv_k", "adv_o", "goalside_o", "goalside_k", "press_k", "crowd", "gap_in", "gap_out", "reach_k",
                "reach_o", "land_k", "land_d", "strong_k", "m_max", "enc_start_dead", "enc_end_dead", "until_dead",
                "flips_enc"]


def feats2(c, ms, mr, pr, s, t, k, a, e):
    """Round 16 cues from the round-15 literature: the transition reaction right after the candidate starts (Bauer &
    Anzer: counter-press signals appear within 1-3 s), the reachability of a missing ball and the landing point of a
    moving one ("no teleport", PathCRF), the shape of the evidence, and the stoppage context."""
    n = len(pr)
    o = 1 - k
    pxy, team, gk, bxy, miss = c["pxy"], c["team"], c["gk"], c["bxy"], c["miss"]
    sgn = 1.0 if c["plus"][k] else -1.0                 # team k attacks towards +x when sgn > 0
    f = {}
    w = slice(s, min(s + 50, n))
    for key, tm, sg in (("adv_k", k, sgn), ("adv_o", o, -sgn)):
        v = c["cen"][tm][w] * sg
        f[key] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    vis = np.flatnonzero(~miss[s:min(s + 25, n)]) + s
    if len(vis):
        rel = (pxy[vis, :, 0] - bxy[vis, None, 0]) * sgn            # + = nearer team o's goal than the ball
        with np.errstate(invalid="ignore"):
            f["goalside_o"] = float(np.mean(np.sum((rel > 0) & ((team == o) & ~gk)[None, :], 1)))
            f["goalside_k"] = float(np.mean(np.sum((rel < 0) & ((team == k) & ~gk)[None, :], 1)))
            d = np.linalg.norm(pxy[vis] - bxy[vis, None, :], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        f["press_k"] = float(np.mean(np.sum((d <= 5) & (team == o)[None, :], 1)))
        f["crowd"] = float(np.mean(np.sum(d <= 10, 1)))
    else:
        f["goalside_o"] = f["goalside_k"] = f["press_k"] = f["crowd"] = np.nan
    lo = max(s - 100, 0)
    back = np.flatnonzero(~miss[lo:s + 1])
    lv = int(back[-1]) + lo if len(back) else -1
    ahead = np.flatnonzero(~miss[s:min(s + 100, n)])
    f["gap_in"] = (s - lv) / ea.FPS if lv >= 0 else 4.0
    f["gap_out"] = int(ahead[0]) / ea.FPS if len(ahead) else 4.0
    f["reach_k"] = f["reach_o"] = f["land_k"] = f["land_d"] = np.nan
    if lv >= 0:
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(pxy[s] - bxy[lv], axis=1)
        d = np.where(np.isfinite(d), d, np.inf)
        f["reach_k"] = float(min(np.min(np.where(team == k, d, np.inf)), 40.0))
        f["reach_o"] = float(min(np.min(np.where(team == o, d, np.inf)), 40.0))
        if lv >= 4 and not miss[lv - 4]:
            v = (bxy[lv] - bxy[lv - 4]) / (4 / ea.FPS)
            if np.linalg.norm(v) > 8:
                at = min(lv + 12, n - 1)
                with np.errstate(invalid="ignore"):
                    d = np.linalg.norm(pxy[at] - (bxy[lv] + v * ((at - lv) / ea.FPS)), axis=1)
                d = np.where(np.isfinite(d), d, np.inf)
                j = int(np.argmin(d))
                if np.isfinite(d[j]):
                    f["land_k"] = float(team[j] == k)
                    f["land_d"] = float(min(np.min(np.where(team == k, d, np.inf)), 40.0))
    sk = 1.0 if k == 0 else -1.0
    f["strong_k"] = max((q - p for p, q in ea.runs(mr[s:t] * sk > 1.0)), default=0) / ea.FPS
    f["m_max"] = float(np.max(ms[s:t] * sk))
    f["enc_start_dead"] = float(a > 0 and pr[a - 1] == 2)
    f["enc_end_dead"] = float(e < n and pr[e] == 2)
    nd = np.flatnonzero(pr[t:] == 2)
    f["until_dead"] = min(int(nd[0]) / ea.FPS, 60.0) if len(nd) else 60.0
    other = (ms[a:e] > 0) != (pr[a] == 0)
    f["flips_enc"] = sum(1 for p, q in ea.runs(other) if q - p >= 6) / max((e - a) / ea.FPS / 60, 1e-3)
    return f


def rows_for(src, blocks, cues, gt_primary, preds, smo, raw, gru, min_len, i0=0, v2=False, extra=None):
    """extra (round 23): per block, (s, t, k) insert candidates of src/short_specialist.py added to those of the
    +-25 probability; their rows carry kpu_cand = 1."""
    rows = []
    for i, (b, gt, pr) in enumerate(zip(blocks, gt_primary, preds)):
        c = cues[i]
        ms, mr = ea.logit(smo[i]), ea.logit(raw[i])
        mg = ea.logit(gru[i]) if gru is not None else None
        fav = np.where(ms > 0, 0, 1)
        restart = np.flatnonzero((pr[1:] < 2) & (pr[:-1] == 2)) + 1
        segs = eo.spells(pr)
        for j, (a, e, k) in enumerate(segs):
            prev = pr[a - 1] if a > 0 else 3
            nxt = pr[e] if e < len(pr) else 3
            prev_len = (a - segs[j - 1][0]) / ea.FPS if j > 0 and segs[j - 1][1] == a else 0.0
            next_len = (segs[j + 1][1] - e) / ea.FPS if j + 1 < len(segs) and segs[j + 1][0] == e else 0.0
            cands = [(a + s, a + t, 1 - k, "ins") for s, t in ea.runs(fav[a:e] != k) if t - s >= ea.RUN]
            if extra is not None:
                cands += [(s, t, kk, "kins") for s, t, kk in extra[i] if a <= s and t <= e and kk == 1 - k]
            if e - a < ea.SHORT:
                cands.append((a, e, k, "del"))
            for s, t, kk, typ in cands:
                frac = float(np.mean(gt[s:t] == kk))
                iou = ec.best_iou(gt, s, t, kk)
                y = 1 if (frac >= 0.5 and iou >= 0.3) else 0 if frac < 0.2 else -1
                f = ea.item(c, gt, ms, mr, mg, s, t, kk)
                f.pop("since_restart")
                r = restart[restart <= s]
                f["since_dead"] = min((s - r[-1]) / ea.FPS, 60.0) if len(r) else 60.0
                f["enc_len"] = (e - a) / ea.FPS
                f["prev_len"], f["next_len"] = (prev_len, next_len) if typ == "del" else (
                    (s - a) / ea.FPS, (e - t) / ea.FPS)
                f["prev_dead"] = float(prev == 2) if typ == "del" else 0.0
                f["next_dead"] = float(nxt == 2) if typ == "del" else 0.0
                if v2:
                    f.update(feats2(c, ms, mr, pr, s, t, kk, a, e))
                f.update(y=y, typ="ins" if typ == "kins" else typ, src=src, group=src + "/" + str(b["group"]), i=i0 + i,
                         s=s, t=t, k=kk, a=a, e=e, min_len=min_len, kpu_cand=float(typ == "kins"))
                rows.append(f)
    return rows


def predict_lomo(rows, keys, params=PARAMS):
    """Leave-one-match-out probabilities, pooled over the sources, per candidate type."""
    p = np.full(len(rows), np.nan)
    for typ in ("ins", "del"):
        idx = np.array([n for n, r in enumerate(rows) if r["typ"] == typ])
        if not len(idx):
            continue
        X = np.array([[rows[n][k] for k in keys] for n in idx], float)
        y = np.array([rows[n]["y"] for n in idx])
        g = np.array([rows[n]["group"] for n in idx])
        for grp in np.unique(g):
            tr = (g != grp) & (y >= 0)
            te = g == grp
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), ROUNDS)
            p[idx[te]] = m.predict(X[te])
    return p


def apply(pred, rows, p, use_ins, use_del, t_ins, t_del):
    out = pred.copy()
    order = sorted(range(len(rows)), key=lambda n: rows[n]["typ"] != "del")      # deletions first
    for n in order:
        r, q = rows[n], p[n]
        s, t, k = r["s"], r["t"], r["k"]
        if r["typ"] == "del" and use_del and q < t_del:
            prev = out[s - 1] if s > 0 else 3
            nxt = out[t] if t < len(out) else 3
            if (prev == 1 - k and nxt in (1 - k, 2)) or (nxt == 1 - k and prev == 2):
                out[s:t] = 1 - k
        elif r["typ"] == "ins" and use_ins and q >= t_ins:
            if not np.all(out[s:t] == 1 - k):
                continue
            left = s - r["a"]
            right = r["e"] - t
            if (0 < left < r["min_len"]) or (0 < right < r["min_len"]):
                continue
            out[s:t] = k
    return out


def _path(kind, *parts):
    return os.path.join(CACHE, kind + "_" + "_".join(str(x).replace("@", "-").replace(" ", "") for x in parts) + ".pkl")


def build(only=None):
    """Stage 1: candidates and v10 decodes per source x seed x dead channel, cached."""
    for src, convs, dec, prim in SOURCES:
        if only and only != src:
            continue
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        cues = [ea.frame_cues(b) for b in blocks]
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        true_d = [np.where(g == 2, 1 - es.EPS, es.EPS) for g in labs[prim]]
        gru = gru_mean(src)
        for seed in es.SEEDS:
            raw = es.v10_raw(seed, src)
            smo = exp_smooth.smooth(raw, 25)
            for dname, pds in zip(DEADS, (learned, true_d)):
                base = es.decode(dec, 48.0, pds, smo)
                rows = rows_for(src, blocks, cues, labs[prim], base, smo, raw, gru, int(dec[1]))
                os.makedirs(CACHE, exist_ok=True)
                bench._dump(dict(base=base, rows=rows), _path("rows", src, seed, dname))
        print("  candidates built for %s" % src, flush=True)


def train(seed, dname, vseed):
    """Stage 2: leave-one-match-out verifier probabilities for one v10 seed and dead channel, cached."""
    rows = [r for src, *_ in SOURCES for r in bench._load(_path("rows", src, seed, dname))["rows"]]
    params = dict(PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    probs = {g: predict_lomo(rows, BASE + (["m_gru"] if g == "G" else []), params) for g in ("no G", "G")}
    bench._dump(probs, _path("probs", seed, dname, "v%d" % vseed))
    yy = np.array([r["y"] for r in rows])
    for g, pp in probs.items():
        for typ in ("ins", "del"):
            for src, *_ in SOURCES:
                sel = np.array([(r["typ"] == typ and r["src"] == src and r["y"] >= 0) for r in rows])
                if sel.sum() > 10 and len(np.unique(yy[sel])) == 2:
                    print("    seed %d %s %s %-4s %-10s AUC %.3f (n %d, positive %.0f%%)" % (
                        seed, dname, g, typ, src, ea.auc(pp[sel & (yy == 1)], pp[sel & (yy == 0)]), sel.sum(),
                        100 * yy[sel].mean()), flush=True)


def train_pooled(seed, vseed):
    """Stage 2b, deployment-like: one leave-one-match-out verifier on both dead channels' candidates pooled (as
    src/short_verifier.py trains the shipped model), probabilities written per dead channel under tag "p<vseed>"."""
    parts = {dname: [r for src, *_ in SOURCES for r in bench._load(_path("rows", src, seed, dname))["rows"]]
             for dname in DEADS}
    rows = parts[DEADS[0]] + parts[DEADS[1]]
    params = dict(PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    probs = {g: predict_lomo(rows, BASE + (["m_gru"] if g == "G" else []), params) for g in ("no G", "G")}
    n = len(parts[DEADS[0]])
    for k, dname in enumerate(DEADS):
        sl = slice(0, n) if k == 0 else slice(n, len(rows))
        bench._dump({g: pp[sl] for g, pp in probs.items()}, _path("probs", seed, dname, "p%d" % vseed))


def evaluate(vseed, variants=None):
    """Stage 3: apply the decision variants to the cached decodes and score them (src/exp_refine.py protocol)."""
    variants = variants or VARIANTS
    cells = {}
    for src, convs, dec, prim in SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in DEADS:
                probs = bench._load(_path("probs", seed, dname, vseed if isinstance(vseed, str) else "v%d" % vseed))
                n0 = 0
                for s2, *_ in SOURCES:
                    if s2 == src:
                        break
                    n0 += len(bench._load(_path("rows", s2, seed, dname))["rows"])
                d = bench._load(_path("rows", src, seed, dname))
                base, rows = d["base"], d["rows"]
                by_half = {}
                for n, r in enumerate(rows):
                    by_half.setdefault(r["i"], []).append(n)
                outs = {"v10": base}
                for name, (use_g, use_ins, use_del, t_ins, t_del) in variants.items():
                    pp = probs["G" if use_g else "no G"]
                    outs[name] = [apply(pr, [rows[n] for n in by_half.get(i, [])],
                                        pp[[n0 + n for n in by_half.get(i, [])]], use_ins, use_del, t_ins, t_del)
                                  for i, pr in enumerate(base)]
                for c in convs:
                    for name, preds in outs.items():
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    return er.summarize(cells, ["v10"] + list(variants))


# -- round 16: the verifier with the round-16 cues (feats2), judged against v13 on identical candidates ---------------

FEATSETS = {"G1": BASE + ["m_gru"], "G2": BASE2 + ["m_gru"], "noG2": BASE2}
VARIANTS2 = {"v13": ("G1", True, True, 0.65, 0.35), "v2 strict": ("G2", True, True, 0.65, 0.35),
             "v2 0.6/0.4": ("G2", True, True, 0.6, 0.4), "v2 0.7/0.3": ("G2", True, True, 0.7, 0.3),
             "v2 no G strict": ("noG2", True, True, 0.65, 0.35)}


def build2(only=None):
    """Stage 1 with the round-16 cues; one block's frame cues at a time (they now carry player positions)."""
    for src, convs, dec, prim in SOURCES:
        if only and only != src:
            continue
        blocks = bench.blocks(src)
        labs = bench.labels(src, prim)
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        true_d = [np.where(g == 2, 1 - es.EPS, es.EPS) for g in labs]
        gru = gru_mean(src)
        raws = {seed: es.v10_raw(seed, src) for seed in es.SEEDS}
        smos = {seed: exp_smooth.smooth(raws[seed], 25) for seed in es.SEEDS}
        bases = {(seed, dname): es.decode(dec, 48.0, pds, smos[seed])
                 for seed in es.SEEDS for dname, pds in zip(DEADS, (learned, true_d))}
        rows = {key: [] for key in bases}
        for i, b in enumerate(blocks):
            c = ea.frame_cues(b)
            for (seed, dname), base in bases.items():
                rows[(seed, dname)] += rows_for(src, [b], [c], [labs[i]], [base[i]], [smos[seed][i]], [raws[seed][i]],
                                                [gru[i]] if gru is not None else None, int(dec[1]), i0=i, v2=True)
            del c
        os.makedirs(CACHE, exist_ok=True)
        for (seed, dname), base in bases.items():
            bench._dump(dict(base=base, rows=rows[(seed, dname)]), _path("rows2", src, seed, dname))
        print("  round-16 candidates built for %s" % src, flush=True)


def train_pooled2(seed, vseed):
    """Stage 2, deployment-like (both dead channels pooled): leave-one-match-out probabilities per feature set."""
    parts = {dname: [r for src, *_ in SOURCES for r in bench._load(_path("rows2", src, seed, dname))["rows"]]
             for dname in DEADS}
    rows = parts[DEADS[0]] + parts[DEADS[1]]
    params = dict(PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    probs = {fs: predict_lomo(rows, keys, params) for fs, keys in FEATSETS.items()}
    n = len(parts[DEADS[0]])
    for k, dname in enumerate(DEADS):
        sl = slice(0, n) if k == 0 else slice(n, len(rows))
        bench._dump({fs: pp[sl] for fs, pp in probs.items()}, _path("probs2", seed, dname, "q%d" % vseed))
    yy = np.array([r["y"] for r in rows])
    for fs, pp in probs.items():
        cells = []
        for typ in ("ins", "del"):
            for src, *_ in SOURCES:
                sel = np.array([(r["typ"] == typ and r["src"] == src and r["y"] >= 0) for r in rows])
                cells.append("%s %s %.3f" % (typ, src.split("@")[0], ea.auc(pp[sel & (yy == 1)], pp[sel & (yy == 0)])))
        print("    seed %d %-5s AUC %s" % (seed, fs, " | ".join(cells)), flush=True)


def evaluate2(vseed):
    cells = {}
    for src, convs, dec, prim in SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in DEADS:
                probs = bench._load(_path("probs2", seed, dname, "q%d" % vseed))
                n0 = 0
                for s2, *_ in SOURCES:
                    if s2 == src:
                        break
                    n0 += len(bench._load(_path("rows2", s2, seed, dname))["rows"])
                d = bench._load(_path("rows2", src, seed, dname))
                base, rows = d["base"], d["rows"]
                by_half = {}
                for n, r in enumerate(rows):
                    by_half.setdefault(r["i"], []).append(n)
                outs = {"v10": base}
                for name, (fs, use_ins, use_del, t_ins, t_del) in VARIANTS2.items():
                    pp = probs[fs]
                    outs[name] = [apply(pr, [rows[n] for n in by_half.get(i, [])],
                                        pp[[n0 + n for n in by_half.get(i, [])]], use_ins, use_del, t_ins, t_del)
                                  for i, pr in enumerate(base)]
                for c in convs:
                    for name, preds in outs.items():
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    er.summarize(cells, ["v10"] + list(VARIANTS2))
    return er.summarize(cells, list(VARIANTS2), ref="v13")


# -- round 16b: the verifier on top of the round-13 BiGRU blend (Metrica and SkillCorner only: the blend has no DFL
#    validation). How much would a DFL-validated blend add on top of v13? ------------------------------------------

def build3(only=None):
    """Candidates of the blend's decode (0.25 v10 + 0.75 BiGRU probability, paired seeds, then +-25) for Metrica and
    SkillCorner; DFL keeps v10's round-16 candidates (rows2) in the pooled training."""
    for src, convs, dec, prim in SOURCES[1:]:
        if only and only != src:
            continue
        blocks = bench.blocks(src)
        labs = bench.labels(src, prim)
        learned = bench.dead_ext("base", "v3", src)
        true_d = [np.where(g == 2, 1 - es.EPS, es.EPS) for g in labs]
        gru = gru_mean(src)
        members = es.bigru_members(src)
        raws = {seed: es.blend(es.v10_raw(seed, src), members[seed]) for seed in es.SEEDS}
        smos = {seed: exp_smooth.smooth(raws[seed], 25) for seed in es.SEEDS}
        bases = {(seed, dname): es.decode(dec, 48.0, pds, smos[seed])
                 for seed in es.SEEDS for dname, pds in zip(DEADS, (learned, true_d))}
        rows = {key: [] for key in bases}
        for i, b in enumerate(blocks):
            c = ea.frame_cues(b)
            for (seed, dname), base in bases.items():
                rows[(seed, dname)] += rows_for(src, [b], [c], [labs[i]], [base[i]], [smos[seed][i]], [raws[seed][i]],
                                                [gru[i]], int(dec[1]), i0=i, v2=True)
            del c
        for (seed, dname), base in bases.items():
            bench._dump(dict(base=base, rows=rows[(seed, dname)]), _path("rows3", src, seed, dname))
        print("  blend candidates built for %s" % src, flush=True)


def _rows3(src, seed, dname):
    return bench._load(_path("rows2" if src.startswith("dfl") else "rows3", src, seed, dname))


def train_pooled3(seed, vseed):
    parts = {dname: [r for src, *_ in SOURCES for r in _rows3(src, seed, dname)["rows"]] for dname in DEADS}
    rows = parts[DEADS[0]] + parts[DEADS[1]]
    params = dict(PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    pp = predict_lomo(rows, FEATSETS["G1"], params)
    n = len(parts[DEADS[0]])
    for k, dname in enumerate(DEADS):
        sl = slice(0, n) if k == 0 else slice(n, len(rows))
        bench._dump({"G1": pp[sl]}, _path("probs3", seed, dname, "q%d" % vseed))
    print("    seed %d blend-verifier probabilities written" % seed, flush=True)


def evaluate3(vseed):
    """v10, v13, the blend alone and the blend + verifier on the six Metrica / SkillCorner contexts."""
    cells = {}
    t_ins, t_del = 0.65, 0.35
    for src, convs, dec, prim in SOURCES[1:]:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in DEADS:
                outs = {}
                for tag, rows_kind, probs_kind in (("v13", "rows2", "probs2"), ("blend + verifier", "rows3", "probs3")):
                    n0 = 0
                    for s2, *_ in SOURCES:
                        if s2 == src:
                            break
                        n0 += len(bench._load(_path("rows2" if s2.startswith("dfl") or rows_kind == "rows2" else
                                                        rows_kind, s2, seed, dname))["rows"])
                    d = bench._load(_path(rows_kind, src, seed, dname))
                    pp = bench._load(_path(probs_kind, seed, dname, "q%d" % vseed))["G1"]
                    by_half = {}
                    for n, r in enumerate(d["rows"]):
                        by_half.setdefault(r["i"], []).append(n)
                    outs["v10" if tag == "v13" else "blend"] = d["base"]
                    outs[tag] = [apply(pr, [d["rows"][n] for n in by_half.get(i, [])],
                                       pp[[n0 + n for n in by_half.get(i, [])]], True, True, t_ins, t_del)
                                 for i, pr in enumerate(d["base"])]
                for c in convs:
                    for name, preds in outs.items():
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    names = ["v10", "v13", "blend", "blend + verifier"]
    er.summarize(cells, names)
    return er.summarize(cells, names[1:], ref="v13")


# -- round 17: DFL with a leave-one-match-out BiGRU (src/exp_bigru_lomo.py on Colab), so that the blend and the
#    blend + verifier are judged on all ten contexts ------------------------------------------------------------

def lomo_bigru():
    """Held-out BiGRU probabilities per DFL half (bench.blocks order), one list per training seed (v10 seed order)."""
    import exp_bigru_lomo as bl
    d = bench._load(os.path.join(os.path.dirname(bench.CACHE), "bigru_lomo", "dfl_lomo_probs.pkl"))
    keys = ["%s/%s" % (b["group"], b["key"].split("/")[-1]) for b in bench.blocks("dfl@v3")]
    return [[np.asarray(d[s][k], np.float64) for k in keys] for s in bl.SEEDS]


def gru_mean4(src):
    """The BiGRU's equal mean as an evaluation may use it: held out on DFL (leave-one-match-out), shared elsewhere."""
    if not src.startswith("dfl"):
        return gru_mean(src)
    members = lomo_bigru()
    return [np.mean([m[i] for m in members], axis=0) for i in range(len(members[0]))]


def build4():
    """The blend's candidates on DFL (paired seeds), with the held-out BiGRU margin (mean of the three seeds)."""
    src, convs, dec, prim = SOURCES[0]
    blocks = bench.blocks(src)
    labs = bench.labels(src, prim)
    learned = bench.dead_probs(src, "base")
    true_d = [np.where(g == 2, 1 - es.EPS, es.EPS) for g in labs]
    members = lomo_bigru()
    gru = gru_mean4(src)
    raws = {seed: es.blend(es.v10_raw(seed, src), members[seed]) for seed in es.SEEDS}
    smos = {seed: exp_smooth.smooth(raws[seed], 25) for seed in es.SEEDS}
    bases = {(seed, dname): es.decode(dec, 48.0, pds, smos[seed])
             for seed in es.SEEDS for dname, pds in zip(DEADS, (learned, true_d))}
    rows = {key: [] for key in bases}
    for i, b in enumerate(blocks):
        c = ea.frame_cues(b)
        for (seed, dname), base in bases.items():
            rows[(seed, dname)] += rows_for(src, [b], [c], [labs[i]], [base[i]], [smos[seed][i]], [raws[seed][i]],
                                            [gru[i]], int(dec[1]), i0=i, v2=True)
        del c
    for (seed, dname), base in bases.items():
        bench._dump(dict(base=base, rows=rows[(seed, dname)]), _path("rows4", src, seed, dname))
    print("  DFL blend candidates built", flush=True)


def _rows4(src, seed, dname):
    return bench._load(_path("rows4" if src.startswith("dfl") else "rows3", src, seed, dname))


def train_pooled4(seed, vseed):
    parts = {dname: [r for src, *_ in SOURCES for r in _rows4(src, seed, dname)["rows"]] for dname in DEADS}
    rows = parts[DEADS[0]] + parts[DEADS[1]]
    params = dict(PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    pp = predict_lomo(rows, FEATSETS["G1"], params)
    n = len(parts[DEADS[0]])
    for k, dname in enumerate(DEADS):
        sl = slice(0, n) if k == 0 else slice(n, len(rows))
        bench._dump({"G1": pp[sl]}, _path("probs4", seed, dname, "q%d" % vseed))
    yy = np.array([r["y"] for r in rows])
    cells = []
    for typ in ("ins", "del"):
        for src, *_ in SOURCES:
            sel = np.array([(r["typ"] == typ and r["src"] == src and r["y"] >= 0) for r in rows])
            cells.append("%s %s %.3f" % (typ, src.split("@")[0], ea.auc(pp[sel & (yy == 1)], pp[sel & (yy == 0)])))
    print("    seed %d blend-verifier AUC %s" % (seed, " | ".join(cells)), flush=True)


def evaluate4(vseed):
    """v10, v13, the blend and the blend + verifier on all ten contexts (DFL with the held-out BiGRU)."""
    cells = {}
    for src, convs, dec, prim in SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in DEADS:
                outs = {}
                for tag, loader, probs_kind in (("v13", lambda s2: bench._load(_path("rows2", s2, seed, dname)),
                                                 "probs2"),
                                                ("blend + verifier", lambda s2: _rows4(s2, seed, dname), "probs4")):
                    n0 = 0
                    for s2, *_ in SOURCES:
                        if s2 == src:
                            break
                        n0 += len(loader(s2)["rows"])
                    d = loader(src)
                    pp = bench._load(_path(probs_kind, seed, dname, "q%d" % vseed))["G1"]
                    by_half = {}
                    for n, r in enumerate(d["rows"]):
                        by_half.setdefault(r["i"], []).append(n)
                    outs["v10" if tag == "v13" else "blend"] = d["base"]
                    outs[tag] = [apply(pr, [d["rows"][n] for n in by_half.get(i, [])],
                                       pp[[n0 + n for n in by_half.get(i, [])]], True, True, 0.65, 0.35)
                                 for i, pr in enumerate(d["base"])]
                for c in convs:
                    for name, preds in outs.items():
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    names = ["v10", "v13", "blend", "blend + verifier"]
    er.summarize(cells, names)
    return er.summarize(cells, names[1:], ref="v13")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if stage == "build4":
        return build4()
    if stage == "trainpool4":
        return train_pooled4(int(sys.argv[2]), int(sys.argv[3]))
    if stage == "eval4":
        return evaluate4(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    if stage == "build3":
        return build3(sys.argv[2] if len(sys.argv) > 2 else None)
    if stage == "trainpool3":
        return train_pooled3(int(sys.argv[2]), int(sys.argv[3]))
    if stage == "eval3":
        return evaluate3(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    if stage == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == "build2":
        build2(sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == "train":
        train(int(sys.argv[2]), DEADS[int(sys.argv[3])], int(sys.argv[4]))
    elif stage == "trainpool":
        train_pooled(int(sys.argv[2]), int(sys.argv[3]))
    elif stage == "trainpool2":
        train_pooled2(int(sys.argv[2]), int(sys.argv[3]))
    elif stage == "eval2":
        evaluate2(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        tag = sys.argv[2] if len(sys.argv) > 2 else "0"
        evaluate(int(tag) if tag.isdigit() else tag)


if __name__ == "__main__":
    main()
