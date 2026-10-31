"""Round 18 (literature review, agent A2; RAPOR's untried R-8): a spell-balanced team classifier under v14.

v10's LightGBM team classifier (DFL native + Metrica ctrl x2, v7 + ball-free features, sim.V3) is retrained with
every frame of a true possession spell of L frames weighted min(cap, sqrt(100 / max(L, 6))) (100 frames = 4 s; long
spells < 1, short spells up to the cap), normalised to mean 1 over each half's live frames, so that each spell weighs
roughly alike instead of in proportion to its length (Pang et al., ECCV 2024: duration imbalance in temporal
segmentation; the cap guards against the extra flips). Caps 2 and 4 were fixed before any run. Held-out
probabilities follow src/exp_combo.py (DFL and Metrica leave-one-match-out, SkillCorner from the model on all), three
LightGBM seeds; then v14's blend (0.5 BiGRU) and its verifier are rebuilt on them and judged against v14 on the ten
contexts. AGGREGATE output only.

Variants are named sb<cap> (spell-balanced team classifier), g<cap> (the spell-balanced BiGRU of
src/exp_bigru_sb.py, trained on Colab) or both, e.g. sb2g2; v14 = neither.

  SB_CAP=2 LGB_SEED=k python -I src/exp_sb.py probs       held-out team probabilities
  python -I src/exp_sb.py build VARIANT                     blend candidates (src/exp_verify.py build3 / build4)
  python -I src/exp_sb.py trainpool VARIANT SEED VSEED      pooled leave-one-match-out verifier
  python -I src/exp_sb.py eval VSEED[,VSEED] VARIANT,...    v14 against the variants
"""
import os
import re
import sys
import warnings

import numpy as np

os.environ.setdefault("COMBO_WM", "2")                 # v10's Metrica weight
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_verify as ev  # noqa: E402

CAPS = (2.0, 4.0)
L0, LMIN = 100, 6
_path = ev._path
_v10_raw = es.v10_raw
_bigru_members = es.bigru_members
_lomo_bigru = ev.lomo_bigru


def spell_weight(lab, cap):
    lab = np.asarray(lab)
    w = np.ones(len(lab), np.float32)
    st = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    en = np.r_[st[1:], len(lab)]
    for a, b in zip(st, en):
        if lab[a] < 2:
            w[a:b] = min(cap, np.sqrt(L0 / max(b - a, LMIN)))
    live = lab < 2
    if live.any():
        w[live] /= w[live].mean()
    return w


def rows(feats, labs, w, cap):
    XF, XH, XA, y, wt = [], [], [], [], []
    for f, lab in zip(feats, labs):
        sl = slice(0, None, bench.STRIDE)
        XF.append(f["XF"][sl]); XH.append(f["XH"][sl]); XA.append(f["XA"][sl]); y.append(lab[sl])
        wt.append(spell_weight(lab, cap)[sl])
    XF, XH, XA, y, wt = (np.concatenate(XF), np.concatenate(XH), np.concatenate(XA), np.concatenate(y),
                         np.concatenate(wt))
    alive = (y == bench.H) | (y == bench.A)
    Xt = np.concatenate([np.hstack([XF[alive], XH[alive]]), np.hstack([XF[alive], XA[alive]])])
    yt = np.concatenate([y[alive] == bench.H, y[alive] == bench.A]).astype(int)
    return Xt, yt, (w * np.concatenate([wt[alive], wt[alive]])).astype(np.float32)


def probs_path(seed, cap):
    return os.path.join(exp_combo.OUT, "probs_a2_bfree%s_wm2_sb%g.pkl" % ("" if seed == 0 else "_seed%d" % seed, cap))


def probs(cap):
    path = probs_path(exp_a2.SEED, cap)
    if os.path.isfile(path):
        return bench._load(path)
    fd, fm = exp_combo.aug_feats("dfl@v3"), exp_combo.aug_feats("metrica@v3")
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    bd, bm = bench.blocks("dfl@v3"), bench.blocks("metrica@v3")
    wm = exp_combo.W_M / 3.0
    res = {"dfl@v3": [None] * len(bd), "metrica@v3": [None] * len(bm)}
    m_all = rows(fm, lm, wm, cap)
    for m in sorted({b["group"] for b in bd}):
        tr = [i for i, b in enumerate(bd) if b["group"] != m]
        mdl = exp_a2.fit([rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0, cap), m_all])
        for i, b in enumerate(bd):
            if b["group"] == m:
                res["dfl@v3"][i] = bench.predict_team(mdl, fd[i])
        print("  cap %g seed %d DFL fold %s done" % (cap, exp_a2.SEED, m), flush=True)
    d_all = rows(fd, ld, 1.0, cap)
    for g in sorted({b["group"] for b in bm}):
        tr = [i for i, b in enumerate(bm) if b["group"] != g]
        mdl = exp_a2.fit([d_all, rows([fm[i] for i in tr], [lm[i] for i in tr], wm, cap)])
        for i, b in enumerate(bm):
            if b["group"] == g:
                res["metrica@v3"][i] = bench.predict_team(mdl, fm[i])
        print("  cap %g seed %d Metrica fold %s done" % (cap, exp_a2.SEED, g), flush=True)
    mdl = exp_a2.fit([d_all, m_all])
    del fd, fm
    res["sc@v3"] = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats("sc@v3")]
    bench._dump(res, path)
    return res


def parse(variant):
    """'sb2g4' -> (2.0, 4.0); 'v14' -> (None, None)."""
    m = re.fullmatch(r"(?:sb(\d+))?(?:g(\d+))?", "" if variant == "v14" else variant)
    return tuple(None if x is None else float(x) for x in m.groups())


def use_variant(variant):
    """Point exp_short's v10 probabilities and BiGRU members (and exp_verify's held-out DFL BiGRU) at the variant,
    blend weight .5, and exp_verify's blend caches at w50 (v14) or w50<variant>."""
    cap, gcap = parse(variant)
    es.W_BIGRU = 0.5
    tag = "w50" if variant == "v14" else "w50" + variant
    es.v10_raw = _v10_raw if cap is None else (lambda seed, src: bench._load(probs_path(seed, cap))[src])
    if gcap is None:
        es.bigru_members, ev.lomo_bigru = _bigru_members, _lomo_bigru
    else:
        import exp_bigru_sb as gsb
        es.bigru_members = lambda src: gsb.members(src, gcap)
        ev.lomo_bigru = lambda: gsb.lomo_members(gcap)

    def tagged(kind, *parts):
        return _path(kind + tag if kind in ("rows3", "rows4", "probs3", "probs4") else kind, *parts)
    ev._path = tagged


def evaluate(vseeds, variants):
    import exp_blend_weight as bw
    cells = {}
    names = ["v14"] + list(variants)
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for seed in es.SEEDS:
            for dname in ev.DEADS:
                for vseed in vseeds:
                    outs = {}
                    for name in names:
                        use_variant(name)
                        _, outs[name] = bw._verified(src, seed, dname, vseed, lambda s2: ev._rows4(s2, seed, dname),
                                                     "probs4")
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
    if stage == "probs":
        probs(float(os.environ.get("SB_CAP", "2")))
    elif stage == "build":
        use_variant(sys.argv[2])
        ev.build3()
        ev.build4()
    elif stage == "trainpool":
        use_variant(sys.argv[2])
        ev.train_pooled4(int(sys.argv[3]), int(sys.argv[4]))
    elif stage == "eval":
        evaluate(tuple(int(x) for x in sys.argv[2].split(",")), sys.argv[3].split(","))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
