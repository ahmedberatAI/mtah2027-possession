"""Round 18 (literature review, agent A1): judge v14's short-possession edits by their effect on the metric.

v14's verifier is a binary "is this short spell real?" classifier with fixed thresholds (insert p >= 0.65, delete
p < 0.35). The literature on F-measure-optimal decisions (Waegeman et al., JMLR 2014; Lipton et al., ECML 2014: for
calibrated probabilities the F1-optimal threshold is half the attainable F1) and on empirical minimum-Bayes-risk
decoding suggests predicting the metric credit of each decision instead. Here every candidate edit of the blend
decode (v14's candidates: src/exp_verify.py rows, blend weight .5) is applied alone and scored with the official
greedy spell matching against the primary labels: gain = change in matched spells averaged over IoU 0.3-0.7, dn =
change in the number of predicted spells. A pooled leave-one-match-out LightGBM regressor learns the gain from v14's
verifier features; an edit is kept when predicted gain > lam x dn (accept when 2 dtp > F dn, lam = F_ref / 2).
AGGREGATE output only.

  python -I src/exp_gain.py gains                  per-candidate gains + the oracle ceiling (true gain > lam x dn)
  python -I src/exp_gain.py trainpool SEED VSEED   pooled leave-one-match-out gain regressor
  python -I src/exp_gain.py eval VSEED[,VSEED]     v14 against the gain rule, ten contexts
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_verify as ev  # noqa: E402

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
LAMS = (0.30, 0.35, 0.40)                     # F_ref 0.6 / 0.7 / 0.8, fixed before the first evaluation


def rows_kind(src):
    return "rows4w50" if src.startswith("dfl") else "rows3w50"


def spells(codes):
    """(start, stop, team) arrays of maximal H / A runs."""
    c = np.asarray(codes)
    st = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    en = np.r_[st[1:], len(c)]
    keep = c[st] < 2
    return st[keep], en[keep], c[st][keep]


def matched(gt, pr):
    """Matched spells per IoU threshold (the official one-to-one greedy match on IoU-descending pairs)."""
    gs, ge, gk = gt
    ps, pe, pk = pr
    ious, gi, pj = [], [], []
    i0 = 0
    for j in range(len(ps)):
        while i0 < len(gs) and ge[i0] <= ps[j]:
            i0 += 1
        i = i0
        while i < len(gs) and gs[i] < pe[j]:
            if gk[i] == pk[j]:
                inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
                union = max(ge[i], pe[j]) - min(gs[i], ps[j])
                ious.append(inter / union)
                gi.append(i)
                pj.append(j)
            i += 1
    if not ious:
        return np.zeros(len(IOUS))
    ious, gi, pj = np.array(ious), np.array(gi), np.array(pj)
    order = np.lexsort((pj, gi, -ious))
    out = np.zeros(len(IOUS))
    for n, th in enumerate(IOUS):
        used_g, used_p = set(), set()
        for o in order:
            if ious[o] < th:
                break
            if gi[o] in used_g or pj[o] in used_p:
                continue
            used_g.add(gi[o])
            used_p.add(pj[o])
        out[n] = len(used_g)
    return out


def gains():
    """Per candidate: gain (matched-spell change averaged over IoUs), dn, ok (the edit applies); the oracle ceiling."""
    for src, convs, dec, prim in ev.SOURCES:
        labs = bench.labels(src, prim)
        gts = [spells(g) for g in labs]
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                d = bench._load(ev._path(rows_kind(src), src, seed, dname))
                base = d["base"]
                ref = {i: matched(gts[i], spells(b)) for i, b in enumerate(base)}
                npred = {i: len(spells(b)[0]) for i, b in enumerate(base)}
                g = np.zeros(len(d["rows"]))
                dn = np.zeros(len(d["rows"]), np.int64)
                ok = np.zeros(len(d["rows"]), bool)
                for n, r in enumerate(d["rows"]):
                    i = r["i"]
                    q = np.array([0.0 if r["typ"] == "del" else 1.0])
                    after = ev.apply(base[i], [r], q, True, True, 0.5, 0.5)
                    if np.array_equal(after, base[i]):
                        continue
                    sp = spells(after)
                    ok[n] = True
                    g[n] = float(np.mean(matched(gts[i], sp) - ref[i]))
                    dn[n] = len(sp[0]) - npred[i]
                bench._dump(dict(gain=g, dn=dn, ok=ok), ev._path("gain", src, seed, dname))
                typ = np.array([r["typ"] for r in d["rows"]])
                for t in ("ins", "del"):
                    sel = ok & (typ == t)
                    print("  %-8s s%d %-9s %s: %4d edits, mean gain %+.3f, dn %s, share with gain > 0.35 dn %.2f" % (
                        src.split("@")[0], seed, dname, t, sel.sum(), g[sel].mean() if sel.any() else 0.0,
                        np.unique(dn[sel]).tolist(), np.mean(g[sel] > 0.35 * dn[sel]) if sel.any() else 0.0),
                          flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    oracle()


def _decide(rows, pred_gain, dn, ok, lam):
    """Verifier-style scores for ev.apply (t 0.5 / 0.5): 1 = insert, 0 = delete, neutral otherwise."""
    p = np.array([0.0 if r["typ"] == "ins" else 1.0 for r in rows])
    keep = ok & (pred_gain > lam * dn)
    for n in np.flatnonzero(keep):
        p[n] = 1.0 if rows[n]["typ"] == "ins" else 0.0
    return p


def _apply_all(d, p):
    by_half = {}
    for n, r in enumerate(d["rows"]):
        by_half.setdefault(r["i"], []).append(n)
    return [ev.apply(pr, [d["rows"][n] for n in by_half.get(i, [])], p[by_half.get(i, [])], True, True, 0.5, 0.5)
            for i, pr in enumerate(d["base"])]


def oracle():
    """Ceiling: every edit whose TRUE gain exceeds lam x dn applied together, against the blend decode and v14."""
    import exp_blend_weight as bw
    bw.use_weight(0.5)
    cells = {}
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                d = bench._load(ev._path(rows_kind(src), src, seed, dname))
                gg = bench._load(ev._path("gain", src, seed, dname))
                _, v14 = bw._verified(src, seed, dname, 0, lambda s2: ev._rows4(s2, seed, dname), "probs4")
                outs = {"blend .5": d["base"], "v14": v14}
                for lam in LAMS:
                    outs["oracle %.2f" % lam] = _apply_all(d, _decide(d["rows"], gg["gain"], gg["dn"], gg["ok"], lam))
                for c in convs:
                    for name, preds in outs.items():
                        cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    er.summarize(cells, ["v14", "blend .5"] + ["oracle %.2f" % lam for lam in LAMS], ref="v14")


FEATS = ev.FEATSETS["G1"] + ["dn"]


def _pooled(seed):
    """All sources' candidates for both dead channels in a fixed order, with their gains; cut = per-channel ends."""
    rows, gain, ok, cut = [], [], [], [0]
    for dname in ev.DEADS:
        for src, *_ in ev.SOURCES:
            d = bench._load(ev._path(rows_kind(src), src, seed, dname))
            gg = bench._load(ev._path("gain", src, seed, dname))
            for r, dn in zip(d["rows"], gg["dn"]):
                r["dn"] = float(dn)
            rows += d["rows"]
            gain.append(gg["gain"])
            ok.append(gg["ok"])
        cut.append(len(rows))
    return rows, np.concatenate(gain), np.concatenate(ok), cut


def trainpool(seed, vseed):
    """Leave-one-match-out gain regressor per candidate type, pooled over sources and dead channels."""
    import lightgbm as lgb
    rows, gain, ok, cut = _pooled(seed)
    params = dict(ev.PARAMS, objective="regression", seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    pred = np.zeros(len(rows))
    for typ in ("ins", "del"):
        idx = np.array([n for n, r in enumerate(rows) if r["typ"] == typ and ok[n]])
        X = np.array([[rows[n][k] for k in FEATS] for n in idx], float)
        y = gain[idx]
        g = np.array([rows[n]["group"] for n in idx])
        for grp in np.unique(g):
            m = lgb.train(params, lgb.Dataset(X[g != grp], y[g != grp]), ev.ROUNDS)
            pred[idx[g == grp]] = m.predict(X[g == grp])
    for k, dname in enumerate(ev.DEADS):
        bench._dump({"gain": pred[cut[k]:cut[k + 1]]}, ev._path("gainp", seed, dname, "q%d" % vseed))
    cells = []
    for typ in ("ins", "del"):
        for src, *_ in ev.SOURCES:
            sel = ok & np.array([r["typ"] == typ and r["src"] == src for r in rows])
            cells.append("%s %s %.2f" % (typ, src.split("@")[0], np.corrcoef(pred[sel], gain[sel])[0, 1]))
    print("    seed %d q%d gain regressor, held-out correlation: %s" % (seed, vseed, " | ".join(cells)), flush=True)


def evaluate(vseeds):
    """v14 (its verifier, probs4w50) against the gain rule for each lam, ten contexts, paired match bootstrap."""
    import exp_blend_weight as bw
    bw.use_weight(0.5)
    cells = {}
    names = ["v14"] + ["gain %.2f" % lam for lam in LAMS]
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                d = bench._load(ev._path(rows_kind(src), src, seed, dname))
                gg = bench._load(ev._path("gain", src, seed, dname))
                n0 = 0
                for s2, *_ in ev.SOURCES:
                    if s2 == src:
                        break
                    n0 += len(bench._load(ev._path(rows_kind(s2), s2, seed, dname))["rows"])
                for vseed in vseeds:
                    _, v14 = bw._verified(src, seed, dname, vseed, lambda s2: ev._rows4(s2, seed, dname), "probs4")
                    pg = bench._load(ev._path("gainp", seed, dname, "q%d" % vseed))["gain"][n0:n0 + len(d["rows"])]
                    outs = {"v14": v14}
                    for lam in LAMS:
                        outs["gain %.2f" % lam] = _apply_all(d, _decide(d["rows"], pg, gg["dn"], gg["ok"], lam))
                    for c in convs:
                        for name, preds in outs.items():
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    mean, regret = er.summarize(cells, names, ref="v14")
    for name in names[1:]:
        bw.regret_boot(mean, names, "v14", name)
        bw.regret_boot(mean, ["v14", name], "v14", name)
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage = sys.argv[1]
    if stage == "gains":
        return gains()
    if stage == "oracle":
        return oracle()
    if stage == "trainpool":
        return trainpool(int(sys.argv[2]), int(sys.argv[3]))
    if stage == "eval":
        return evaluate(tuple(int(x) for x in sys.argv[2].split(",")))
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
