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
src/exp_bigru_sb.py, trained on Colab), gm or gm<cap> (round 21: the BiGRU of src/exp_bigru_m.py, the three Metrica
games in its training, optionally spell-balanced), gp (round 21: the same script's plain mode, v14's BiGRU recipe
retrained on this machine -- the control), gv (round 21: src/exp_bigru_pv3.py, the recipe with PFF degraded by
sim.V3 like DFL), an x after the BiGRU token (round 21: its members averaged seed by seed with
v14's, e.g. gpx = six members of v14's recipe, gmx = v14's and the Metrica-trained), d<10 x beta> (round 19, analyst idea 3: dominance-aware length prior -- a
first decode gives each half's possession share s, then each team's soft-minimum-length cost is 48 x (s_team / 0.5)^beta
with s clipped to [0.3, 0.7], so the team with less of the ball may end its possessions sooner; beta 0.5 / 1 fixed in
advance), s<w> (round 20: the team probability smoothed over +-w frames instead of v14's +-25), c<cost> (round 20:
the soft-minimum-length cost instead of 48) or combinations, e.g. sb2g2; v14 = none.

  SB_CAP=2 LGB_SEED=k python -I src/exp_sb.py probs       held-out team probabilities
  python -I src/exp_sb.py build VARIANT                     blend candidates (src/exp_verify.py build3 / build4)
  python -I src/exp_sb.py trainpool VARIANT SEED VSEED      pooled leave-one-match-out verifier
  python -I src/exp_sb.py eval VSEED[,VSEED] VARIANT,... [REF]   the variants against v14 (or against REF, e.g. gp)
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
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402

CAPS = (2.0, 4.0)
L0, LMIN = 100, 6
_path = ev._path
_v10_raw = es.v10_raw
_decode = es.decode
_smooth = exp_smooth.smooth
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
    """'sb2g4d10s12c36' -> (2.0, 4.0, 1.0, 12, 36.0, None); 'gm2' -> (None, 2.0, ..., 'm'); 'gp' -> gcap None and
    mode 'p' (src/exp_bigru_m.py with / without the Metrica games); 'v14' -> five Nones and mode None."""
    m = re.fullmatch(r"(?:sb(\d+))?(?:g(\d+)|g([mpv])(\d*))?(x)?(?:d(\d+))?(?:s(\d+))?(?:c(\d+))?",
                     "" if variant == "v14" else variant)
    cap, g, mode, gm, mix, d, w, c = m.groups()
    gcap = g or gm or None
    return (None if cap is None else float(cap), None if gcap is None else float(gcap),
            None if d is None else float(d) / 10, None if w is None else int(w), None if c is None else float(c), mode,
            mix is not None)


def dominance_decode(beta):
    """exp_short.decode with team-specific soft-minimum-length costs from a first decode's possession share."""
    import decode2
    import exp_v7
    import train_dfl as td

    def decode(dec, c, pds, phs):
        out = []
        for pd_, ph, first in zip(pds, phs, _decode(dec, c, pds, phs)):
            live = first < 2
            share = min(max(float(np.mean(first[live] == 0)) if live.any() else 0.5, 0.3), 0.7)
            macros = [dict(emis=k, out=k, L=175, min_exit=dec[1] - 1,
                           exit_cost=exp_v7.ramp(175, c * ((share if k == 0 else 1 - share) / 0.5) ** beta))
                      for k in (0, 1)] + [dict(emis=2, out=2, L=dec[2])]
            pen = [[0, 12.0, dec[0]], [12.0, 0, dec[0]], [dec[0], dec[0], 0]]
            out.append(decode2.Spec(macros, pen).decode(td.emissions(pd_, np.clip(ph, es.EPS, 1 - es.EPS), 1.0, 1.0))
                       .astype(np.int8))
        return out
    return decode


def _mix(a, b):
    """Seed-by-seed equal average of two member lists (members x halves)."""
    return [[0.5 * (np.asarray(x, np.float64) + np.asarray(y, np.float64)) for x, y in zip(ma, mb)] for ma, mb in zip(a, b)]


def use_variant(variant):
    """Point exp_short's v10 probabilities and BiGRU members (and exp_verify's held-out DFL BiGRU) at the variant,
    blend weight .5, and exp_verify's blend caches at w50 (v14) or w50<variant>."""
    cap, gcap, beta, width, cost, mode, mix = parse(variant)
    es.W_BIGRU = 0.5
    base = _decode if beta is None else dominance_decode(beta)
    es.decode = base if cost is None else (lambda dec, c, pds, phs: base(dec, cost, pds, phs))
    exp_smooth.smooth = _smooth if width is None else (lambda x, w: _smooth(x, width))
    tag = "w50" if variant == "v14" else "w50" + variant
    es.v10_raw = _v10_raw if cap is None else (lambda seed, src: bench._load(probs_path(seed, cap))[src])
    if mode == "v":
        import exp_bigru_pv3 as pv
        es.bigru_members, ev.lomo_bigru = pv.members, lambda: pv.members("dfl@v3")
    elif mode:
        import exp_bigru_m as gm
        es.bigru_members = lambda src: gm.members(src, gcap, mode == "m")
        ev.lomo_bigru = lambda: gm.members("dfl@v3", gcap, mode == "m")
    elif gcap is None:
        es.bigru_members, ev.lomo_bigru = _bigru_members, _lomo_bigru
    else:
        import exp_bigru_sb as gsb
        es.bigru_members = lambda src: gsb.members(src, gcap)
        ev.lomo_bigru = lambda: gsb.lomo_members(gcap)
    if mix:
        own_members, own_lomo = es.bigru_members, ev.lomo_bigru
        es.bigru_members = lambda src: _mix(_bigru_members(src), own_members(src))
        ev.lomo_bigru = lambda: _mix(_lomo_bigru(), own_lomo())

    def tagged(kind, *parts):
        return _path(kind + tag if kind in ("rows3", "rows4", "probs3", "probs4") else kind, *parts)
    ev._path = tagged


def evaluate(vseeds, variants, ref="v14"):
    import exp_blend_weight as bw
    cells = {}
    names = [ref] + [v for v in ["v14"] + list(variants) if v != ref]
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
    mean, regret = er.summarize(cells, names, ref=ref)
    for name in names[1:]:
        bw.regret_boot(mean, names, ref, name)
        bw.regret_boot(mean, [ref, name], ref, name)
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
        evaluate(tuple(int(x) for x in sys.argv[2].split(",")), sys.argv[3].split(","),
                 sys.argv[4] if len(sys.argv) > 4 else "v14")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
