"""Round 24 (agent A1's method 2): supervised boundary regression for short possessions. AGGREGATE output only.

Half of the short possessions the verifier inserts correctly miss IoU 0.7 with their true spell (round 24 diagnostic),
and rule-based shrinking (round 19) and snapping to the specialist's change peaks (round 23) both lost. Here two
LightGBM regressors learn, from context, how far each boundary of a short candidate (v14's insert candidates and its
decoded possessions < 4 s) sits from the boundary of its true spell (same team, best IoU, clipped to +-50 frames).
Inputs: profiles of the blend's raw and +-25 logits, the BiGRU's and v10's logits and, optionally, the short-possession
specialist's control / gain / lose outputs, sampled at -36..+36 frames around each boundary, plus duration, type and
dead-ball neighbours. Trained on true short candidates only (y == 1), leave-one-match-out over the 30 matches, both
dead channels pooled, per v10 seed. Applied to the candidates the verifier keeps or inserts.

Gate (pre-registered, docs/RAPOR.md Tur 24): held-out true short candidates gain >= 5 points in the share with IoU >= 0.7;
then spell F1 under the round-24 decision rule.

  python -I src/exp_boundary.py fit [KPU]        held-out offsets for v14's candidates (KPU: S1 / C1 / none)
  python -I src/exp_boundary.py intrinsic [KPU]  IoU >= 0.5 / 0.7 shares before and after
  python -I src/exp_boundary.py eval [KPU]       ten contexts against v14 (verifier seeds 0 and 1)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_kpu  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import short_specialist as ss  # noqa: E402

OFFS = (-36, -24, -12, -6, 0, 6, 12, 24, 36)
CLIP = 50
PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0, deterministic=True,
              force_row_wise=True)
ROUNDS = 300


def _true_spell(g, s, t, k):
    """Boundaries of the same-team true spell with the best IoU against [s, t), or None."""
    lo, hi = max(0, s - 250), min(len(g), t + 250)
    best, out = 0.0, None
    seg = g[lo:hi]
    st = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    for a, b in zip(st, np.r_[st[1:], len(seg)]):
        a, b = a + lo, b + lo
        if g[a] != k or b <= s or a >= t:
            continue
        iou = (min(b, t) - max(a, s)) / (max(b, t) - min(a, s))
        if iou > best:
            best, out = iou, (a, b)
    return out


def _series(src, seed):
    """Per half: the blend's raw and +-25 logits, the BiGRU's and v10's logits (home-oriented)."""
    members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
    v10 = es.v10_raw(seed, src)
    raw = es.blend(v10, members[seed])
    smo = exp_smooth.smooth(raw, 25)
    gru = [np.mean([m[i] for m in members], axis=0) for i in range(len(members[0]))]
    return [dict(raw=ea.logit(r), smo=ea.logit(s), gru=ea.logit(g), v10=ea.logit(v)) for r, s, g, v in
            zip(raw, smo, gru, v10)]


def _features(r, ser, kpu):
    s, t, k = r["s"], r["t"], r["k"]
    sign = 1.0 if k == 0 else -1.0
    n = len(ser["raw"])
    f = {"dur": (t - s) / 25.0, "ins": float(r["typ"] == "ins"), "prev_dead": r.get("prev_dead", 0.0),
         "next_dead": r.get("next_dead", 0.0), "since_dead": r.get("since_dead", 60.0)}
    for side, b in (("s", s), ("e", t)):
        idx = np.clip(np.array(OFFS) + b, 0, n - 1)
        for name in ("raw", "smo", "gru", "v10"):
            for o, v in zip(OFFS, sign * ser[name][idx]):
                f["%s_%s_%d" % (side, name, o)] = float(v)
        if kpu is not None:
            pk = exp_kpu.pk(kpu, k)
            gain, lose = kpu[:, 3 + k], kpu[:, 5 + k]
            for o, a, g, l in zip(OFFS, pk[idx], gain[idx], lose[idx]):
                f["%s_kc_%d" % (side, o)] = float(a)
                f["%s_kg_%d" % (side, o)] = float(g)
                f["%s_kl_%d" % (side, o)] = float(l)
    return f


def _path(kpu, seed):
    return os.path.join(ev.CACHE, "boundary_%s_s%d.pkl" % (kpu or "none", seed))


def fit(kpu_variant=None):
    import lightgbm as lgb
    exp_kpu.use("v14")
    for k in es.SEEDS:
        rows, X, Y = [], [], []
        for src, convs, dec, prim in ev.SOURCES:
            labs = bench.labels(src, prim)
            ser = _series(src, k)
            outs = exp_kpu.outputs(kpu_variant, src)[ss.SEEDS[k]] if kpu_variant else None
            for dname in ev.DEADS:
                for r in ev._rows4(src, k, dname)["rows"]:
                    kp = np.asarray(outs[r["i"]], np.float32) if outs is not None else None
                    f = _features(r, ser[r["i"]], kp)
                    tr = _true_spell(np.asarray(labs[r["i"]]), r["s"], r["t"], r["k"]) if r["y"] == 1 else None
                    rows.append(dict(src=src, dname=dname, i=r["i"], s=r["s"], t=r["t"], k=r["k"], typ=r["typ"],
                                     group=r["group"], true=tr))
                    X.append(f)
                    Y.append((np.clip(tr[0] - r["s"], -CLIP, CLIP), np.clip(tr[1] - r["t"], -CLIP, CLIP))
                             if tr is not None else (np.nan, np.nan))
            bench._blocks.clear()
            bench._labels.clear()
        keys = sorted(X[0])
        Xa = np.array([[x[c] for c in keys] for x in X], float)
        Ya = np.array(Y, float)
        g = np.array([r["group"] for r in rows])
        pred = np.zeros((len(rows), 2))
        for grp in np.unique(g):
            tr = (g != grp) & np.isfinite(Ya[:, 0])
            te = g == grp
            for j in (0, 1):
                m = lgb.train(PARAMS, lgb.Dataset(Xa[tr], Ya[tr, j]), ROUNDS)
                pred[te, j] = m.predict(Xa[te])
        bench._dump(dict(rows=rows, pred=pred), _path(kpu_variant, k))
        print("  v10 seed %d: %d candidates (%d with a true spell), offsets fitted" % (
            k, len(rows), int(np.isfinite(Ya[:, 0]).sum())), flush=True)


def intrinsic(kpu_variant=None):
    for k in es.SEEDS:
        d = bench._load(_path(kpu_variant, k))
        stats = {}
        for r, p in zip(d["rows"], d["pred"]):
            if r["true"] is None:
                continue
            a, b = r["true"]
            s0, t0 = r["s"], r["t"]
            s1, t1 = int(round(s0 + p[0])), int(round(t0 + p[1]))
            if t1 - s1 < 3:
                s1, t1 = s0, t0
            iou0 = (min(b, t0) - max(a, s0)) / (max(b, t0) - min(a, s0))
            iou1 = max(0.0, min(b, t1) - max(a, s1)) / (max(b, t1) - min(a, s1))
            key = (r["src"].split("@")[0], r["typ"])
            st = stats.setdefault(key, np.zeros(5))
            st += (1, iou0 >= 0.5, iou0 >= 0.7, iou1 >= 0.5, iou1 >= 0.7)
        print("v10 seed %d " % k + " | ".join("%s %s n %d: IoU>=0.5 %.2f -> %.2f, >=0.7 %.2f -> %.2f" % (
            src, typ, st[0], st[1] / st[0], st[3] / st[0], st[2] / st[0], st[4] / st[0])
            for (src, typ), st in sorted(stats.items())), flush=True)


def _adjust(preds, rows, offs, applied):
    """Move the boundaries of the candidates the verifier applied (inserted / kept) by their predicted offsets."""
    out = [p.copy() for p in preds]
    for r, o in zip(rows, offs):
        if (r["i"], r["s"], r["t"], r["typ"]) not in applied:
            continue
        p, k = out[r["i"]], r["k"]
        s0, t0 = r["s"], r["t"]
        s1 = int(np.clip(round(s0 + o[0]), s0 - 25, t0 - 3))
        t1 = int(np.clip(round(t0 + o[1]), s1 + 3, t0 + 25))
        n = len(p)
        if s1 < s0:
            p[max(0, s1):s0] = k
        elif s1 > s0:
            p[s0:s1] = p[s0 - 1] if s0 > 0 else p[s1]
        if t1 > t0:
            p[t0:min(n, t1)] = k
        elif t1 < t0:
            p[t1:t0] = p[t0] if t0 < n else p[t1 - 1]
    return out


def evaluate(kpu_variant=None, vseeds=(0, 1)):
    import exp_blend_weight as bw
    import exp_refine as er
    exp_kpu.use("v14")
    cells = {}
    names = ["v14", "v14+bound"]
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for k in es.SEEDS:
            d = bench._load(_path(kpu_variant, k))
            for dname in ev.DEADS:
                sel = [(r, p) for r, p in zip(d["rows"], d["pred"]) if r["src"] == src and r["dname"] == dname]
                rows = ev._rows4(src, k, dname)["rows"]
                for vseed in vseeds:
                    base, preds = bw._verified(src, k, dname, vseed, lambda s2: ev._rows4(s2, k, dname), "probs4")
                    applied = set()
                    for r in rows:
                        pr = preds[r["i"]]
                        if r["typ"] == "ins" and np.mean(pr[r["s"]:r["t"]] == r["k"]) > 0.9:
                            applied.add((r["i"], r["s"], r["t"], r["typ"]))
                        if r["typ"] == "del" and np.mean(pr[r["s"]:r["t"]] == r["k"]) > 0.9:
                            applied.add((r["i"], r["s"], r["t"], r["typ"]))
                    adj = _adjust(preds, [r for r, _ in sel], [p for _, p in sel], applied)
                    for c in convs:
                        cells.setdefault((src, c, dname, "v14"), []).append(er.by_match(blocks, labs[c], preds))
                        cells.setdefault((src, c, dname, "v14+bound"), []).append(er.by_match(blocks, labs[c], adj))
        bench._blocks.clear()
        bench._labels.clear()
    mean, regret = er.summarize(cells, names, ref="v14")
    bw.regret_boot(mean, names, "v14", "v14+bound")
    bw.decide(mean, "v14", "v14+bound")
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage = sys.argv[1]
    kpu = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    if stage == "fit":
        fit(kpu)
    elif stage == "intrinsic":
        intrinsic(kpu)
    elif stage == "eval":
        evaluate(kpu)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
