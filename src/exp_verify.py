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


def rows_for(src, blocks, cues, gt_primary, preds, smo, raw, gru, min_len):
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
                f.update(y=y, typ=typ, src=src, group=src + "/" + str(b["group"]), i=i, s=s, t=t, k=kk, a=a, e=e,
                         min_len=min_len)
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if stage == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == "train":
        train(int(sys.argv[2]), DEADS[int(sys.argv[3])], int(sys.argv[4]))
    elif stage == "trainpool":
        train_pooled(int(sys.argv[2]), int(sys.argv[3]))
    else:
        tag = sys.argv[2] if len(sys.argv) > 2 else "0"
        evaluate(int(tag) if tag.isdigit() else tag)


if __name__ == "__main__":
    main()
