"""Round 40 (docs/RAPOR.md, pre-registered): dead-ball evidence in BAR-ATH's condition. BAR-ATH's rules are v2's tracking
rules + broadcast validity; DFL / Metrica have the tracking but no broadcast, SkillCorner has a real view flag but no
continuous tracking. Here the real view process of SkillCorner is transplanted onto DFL / Metrica / SoccerTrack: for every
true live / dead segment of a half a SkillCorner segment of the same state and nearly the same length is taken and its
flag copied (time-stretched), and the rule evidence follows BAR-ATH's formula (model_v3: -1.5 without a valid view, +0.4
with live players). Then

  rules      BAR-ATH's kind of dead-ball evidence
  rules+DT   + the boundary timing of v16 (the current recipe, before the verifier)
  mix w      rules + w x the learned dead model's alive log-odds, w 0.25 / 0.5 (0.5 set by N21 on 23 October)
  mix.5+DT   + the boundary timing
  learned    the learned dead model alone

Team probability: v17's recipe (three v10 seeds, each with one J1 seed, +-25 smoothing), BAR-ATH's decoder, before the
verifier. Three transplant draws. AGGREGATE output only.

  python -I src/exp_deadmix.py eval        the arms on all contexts + the decision rule (rules+DT against each mix)
  python -I src/exp_deadmix.py st          SoccerTrack v2 (not used to design the round)
  python -I src/exp_deadmix.py v3g         Metrica with the de-glued ball for the dead-ball evidence
  python -I src/exp_deadmix.py stress      how the mix fares if the rules are better / the learned model worse than measured
  python -I src/exp_deadmix.py dm          the mix with a dead model trained on more sources (DM, DMS)
  python -I src/exp_deadmix.py verifier    the verifier on the mix's decode (rows: src/exp_kpu.py build X1abETm, X1abETmS)
  python -I src/exp_deadmix.py shape       screen around the 0.5 mix: bounds, view weights, bias, decoder penalties
  python -I src/exp_deadmix.py head        the specialist's own dead head as evidence
  python -I src/exp_deadmix.py oracle      where the rest of the true-dead gap of the mix lies (existence, timing)
  python -I src/exp_deadmix.py dm2         how to use the multi-source models (alone, with the view flag, with the rules; LOPO)
  python -I src/exp_deadmix.py qnorm       the team uncertainty out of the live / dead contest
  python -I src/exp_deadmix.py dm3         provider-balanced training and decode profiles of the multi-source models
  python -I src/exp_deadmix.py dm4         PFF added to the balanced multi-source dead model
  python -I src/exp_deadmix.py stress2     round 41's bet: better rules, worse model, hedges
  python -I src/exp_deadmix.py st2         SoccerTrack v2 with the multi-source model
  python -I src/exp_deadmix.py head2       screen on top of the multi-source model + view flag
  python -I src/exp_deadmix.py stress3 SRC hedges: sums of the channels under better rules and worse learned channels
  python -I src/exp_deadmix.py eval2       round 41's decision run: E1 = rules + view + 0.5 DFL model + 0.5 multi-source model
  python -I src/exp_deadmix.py st4         SoccerTrack v2 with the channels and their sums
  python -I src/exp_deadmix.py pff         round 42: PFF's 64 matches as an independent test of the dead-ball evidence
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import dead_timing as dt  # noqa: E402
import exp_blend  # noqa: E402
import exp_dsplit as ds  # noqa: E402
import exp_kpu  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import model_v3  # noqa: E402
import short_specialist as ss  # noqa: E402

P = dict(model_v3.DEFAULT)
CONVS = {"dfl@v3": ("evctrl", "trk"), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp", "evctrl")}
DRAWS = (0, 1, 2)
SEED = 20261104
NAMES = ("rules", "rules+DT", "mix.25", "mix.5", "mix.5+DT", "learned")
TAG = "BAR-like D"
WEIGHT = 0.5
_lib = {}


def segs(d):
    st = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    return zip(st, np.r_[st[1:], len(d)])


def library():
    """SkillCorner's view flag per true live / dead segment (dead = its phase labels' D)."""
    if _lib:
        return _lib
    labs = bench.labels("sc@v3", "phase")
    lib = {False: [], True: []}
    for (valid, ndet), g in zip(sd.view_cues(), labs):
        d = np.asarray(g) == 2
        for a, e in segs(d):
            lib[bool(d[a])].append((e - a, valid[a:e].copy(), (valid[a:e] & (ndet[a:e] >= 4)).copy()))
    bench._labels.clear()
    _lib.update(lib)
    return _lib


def transplant(g, rng):
    """(valid, live) for one half with truth g: per true segment, a SkillCorner segment of the same state among the 20
    nearest in length, stretched in time."""
    lib = library()
    d = np.asarray(g) == 2
    valid = np.ones(len(d), bool)
    live = np.zeros(len(d), bool)
    lens = {s: np.log(np.array([x[0] for x in lib[s]], float)) for s in lib}
    for a, e in segs(d):
        s = bool(d[a])
        near = np.argsort(np.abs(lens[s] - np.log(e - a)))[:20]
        _, v, l = lib[s][int(rng.choice(near))]
        ix = np.minimum((np.arange(e - a) * len(v) / (e - a)).astype(int), len(v) - 1)
        valid[a:e] = v[ix]
        live[a:e] = l[ix]
    return valid, live


def rule_view(src, blocks, truth, draw):
    """BAR-ATH's formula on one source: SkillCorner's own flag, or the transplanted one."""
    a_rule = ds.rule_evidence(src, blocks)
    if src.startswith("sc"):
        return a_rule
    out = []
    for i, (a, g) in enumerate(zip(a_rule, truth)):
        v, l = transplant(g, np.random.default_rng([SEED, draw, i]))
        out.append(a - P["w_binvalid"] * (~v) + P["w_blive"] * l)
    return out


def mix_evidence(a_rb, a_lrn, w=WEIGHT):
    return [a + w * b for a, b in zip(a_rb, a_lrn)]


def team(src, variant="J1"):
    """{v10 seed: smoothed team probability per half} with the given specialist."""
    members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
    outs = exp_kpu.outputs(variant, src)
    return {k: exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(es.blend(es.v10_raw(k, src), members[k], 0.5), outs[ss.SEEDS[k]])], 25) for k in es.SEEDS}


def learned(src):
    dl = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", src)
    return [exp_blend.logit_alive(x) for x in dl]


def arms(a_rb, a_lrn, smo):
    rule, lrn = sd.decode_scores(a_rb, smo), sd.decode_scores(a_lrn, smo)
    mix = {w: sd.decode_scores([a + w * b for a, b in zip(a_rb, a_lrn)], smo) for w in (0.25, 0.5)}
    return {"rules": rule, "rules+DT": [dt.retime(r, l) for r, l in zip(rule, lrn)], "mix.25": mix[0.25], "mix.5": mix[0.5],
            "mix.5+DT": [dt.retime(r, l) for r, l in zip(mix[0.5], lrn)], "learned": lrn}


def evaluate():
    import exp_blend_weight as bw
    cells = {}
    for src, convs in CONVS.items():
        blocks = bench.blocks(src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        a_lrn = learned(src)
        smos = team(src)
        for draw in (DRAWS if not src.startswith("sc") else (0,)):
            a_rb = rule_view(src, blocks, labs[convs[0]], draw)
            for k in es.SEEDS:
                for name, preds in arms(a_rb, a_lrn, smos[k]).items():
                    for c in convs:
                        cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s draw %d done" % (src, draw), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(NAMES), ref="rules+DT")
    for cand in ("mix.25", "mix.5", "mix.5+DT"):
        bw.regret_boot(mean, ["rules+DT", cand], "rules+DT", cand)
        bw.decide(mean, "rules+DT", cand)
    bw.regret_boot({k: v for k, v in mean.items() if k[:2] in bw.PRIMARY}, ["mix.25", "mix.5"], "mix.25", "mix.5")
    bw.regret_boot({k: v for k, v in mean.items() if k[:2] in bw.PRIMARY}, ["mix.5", "mix.5+DT"], "mix.5", "mix.5+DT")


def soccertrack():
    import exp_st
    src = exp_st.SRC
    blocks = bench.blocks(src)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in exp_st.CONVS}
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % src.replace("@", "-")))
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", src)]
    cells = {}
    for draw in DRAWS:
        a_rb = rule_view(src, blocks, labs["evctrl"], draw)
        for k in es.SEEDS:
            raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
            smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
            for name, preds in arms(a_rb, a_lrn, smo).items():
                for c in exp_st.CONVS:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  draw %d done" % draw, flush=True)
    er.summarize(cells, list(NAMES), ref="rules+DT")


def deglued():
    src = "metrica@v3"
    blocks = bench.blocks("metrica@v3g")
    labs = [np.asarray(g) for g in bench.labels(src, "ctrl")]
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", "metrica@v3g")]
    smos = team(src)
    cells = {}
    for draw in DRAWS:
        a_rb = rule_view("metrica@v3g", blocks, labs, draw)
        for k in es.SEEDS:
            for name, preds in arms(a_rb, a_lrn, smos[k]).items():
                cells.setdefault((src, "ctrl", TAG, name), []).append(er.by_match(blocks, labs, preds))
        print("  draw %d done" % draw, flush=True)
    er.summarize(cells, list(NAMES), ref="rules+DT")


def _dead_f1(preds, truth):
    tp = sum(int(((p == 2) & (t == 2)).sum()) for p, t in zip(preds, truth))
    fp = sum(int(((p == 2) & (t != 2)).sum()) for p, t in zip(preds, truth))
    fn = sum(int(((p != 2) & (t == 2)).sum()) for p, t in zip(preds, truth))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _ar1(n, rng, sigma, tau=250.0):
    """Smooth noise of standard deviation sigma with a time constant of tau frames."""
    rho = np.exp(-1.0 / tau)
    e = rng.standard_normal(n) * np.sqrt(1 - rho * rho)
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return sigma * x


def stress():
    """What if the target differs from the open sets: the rules made better with kappa x the truth (+ alive, - dead), the
    learned log-odds made worse with slow noise of sigma. Seed 0, draw 0. Rows: spellF1 of rules+DT, mix.25, mix.5."""
    for src, conv in (("dfl@v3", "evctrl"), ("metrica@v3", "ctrl"), ("sc@v3", "evctrl")):
        blocks = bench.blocks(src)
        truth = [np.asarray(g) for g in bench.labels(src, conv)]
        a_lrn = learned(src)
        smo = team(src)[es.SEEDS[0]]
        a_rb = rule_view(src, blocks, truth, 0)
        print("\n%s %s: kappa sigma | dead F1 rules, learned | spellF1 rules+DT mix.25 mix.5 | mix.25 - rules+DT, mix.5 - rules+DT" % (src.split("@")[0], conv), flush=True)
        for sigma in (0.0, 3.0, 6.0):
            lrn2 = [a + _ar1(len(a), np.random.default_rng([SEED, 7, i]), sigma) if sigma else a for i, a in enumerate(a_lrn)]
            for kappa in (0.0, 1.0, 2.0, 4.0):
                rb2 = [a + kappa * np.where(g == 2, -1.0, 1.0) for a, g in zip(a_rb, truth)]
                res = arms(rb2, lrn2, smo)
                sc = {n: np.mean(list(er.by_match(blocks, truth, res[n]).values())) for n in ("rules+DT", "mix.25", "mix.5")}
                print("  %.0f %.0f | %.3f %.3f | %.4f %.4f %.4f | %+.4f %+.4f" % (kappa, sigma, _dead_f1(res["rules"], truth), _dead_f1(res["learned"], truth), sc["rules+DT"], sc["mix.25"], sc["mix.5"],
                                                                                 sc["mix.25"] - sc["rules+DT"], sc["mix.5"] - sc["rules+DT"]), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


def _groups(blocks):
    return [b["group"] for b in blocks]


def dm_probs(name, src):
    """P(dead) per half from a dead model on more sources, each half from models that never saw its match. DM: DFL +
    Metrica (de-glued input, target-convention D). DMS: + SkillCorner. Cached in cache/bench/dead/."""
    path = os.path.join(bench.CACHE, "dead", "%s_%s.pkl" % (name, src.replace("@", "-")))
    if os.path.isfile(path):
        return bench._load(path)
    train = {"dfl@v3": (bench.features("base", "dfl@v3"), bench.labels("dfl", "native"), _groups(bench.blocks("dfl@v3"))),
             "metrica@v3g": (bench.features("base", "metrica@v3g"), bench.labels("metrica@v3", "ctrl"), _groups(bench.blocks("metrica@v3g")))}
    if name == "DMS":
        train["sc@v3"] = (bench.features("base", "sc@v3"), bench.labels("sc@v3", "evctrl"), _groups(bench.blocks("sc@v3")))
    fsrc = bench.features("base", src)
    gr = _groups(bench.blocks(src))
    un = sorted(set(gr), key=str)
    folds = [un[i::5] for i in range(5)] if src.startswith("sc") else [[g] for g in un]
    if src not in train and not (src == "metrica@v3" and "metrica@v3g" in train):
        folds = [[]]
    res = [None] * len(fsrc)
    for fold in folds:
        X, Y = [], []
        for s, (fs, ls, gs) in train.items():
            for x, y, g in zip(fs, ls, gs):
                if g not in fold:
                    X.append(x)
                    Y.append(y)
        model = bench.fit_dead(X, Y)
        for i, (x, g) in enumerate(zip(fsrc, gr)):
            if (not fold) or g in fold:
                res[i] = model.predict(x["XF"], num_threads=bench.THREADS)
        print("    %s %s fold %s done" % (name, src, fold), flush=True)
    bench._dump(res, path)
    return res


def multi_source():
    import exp_blend_weight as bw
    names = ("mix.5", "mix.5 DM", "mix.5 DMS", "DM", "DMS", "learned")
    cells = {}
    for src, convs in CONVS.items():
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = {"": [exp_blend.logit_alive(x) for x in base]}
        for nm in ("DM", "DMS"):
            a_lrn[nm] = [exp_blend.logit_alive(x) for x in dm_probs(nm, ev_src)]
        smos = team(src)
        for draw in (DRAWS if not src.startswith("sc") else (0,)):
            a_rb = rule_view(ev_src, blocks, labs[convs[0]], draw)
            for k in es.SEEDS:
                res = {"mix.5": sd.decode_scores([a + 0.5 * b for a, b in zip(a_rb, a_lrn[""])], smos[k]), "learned": sd.decode_scores(a_lrn[""], smos[k])}
                for nm in ("DM", "DMS"):
                    res["mix.5 " + nm] = sd.decode_scores([a + 0.5 * b for a, b in zip(a_rb, a_lrn[nm])], smos[k])
                    res[nm] = sd.decode_scores(a_lrn[nm], smos[k])
                for name, preds in res.items():
                    for c in convs:
                        cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s draw %d done" % (src, draw), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="mix.5")
    for cand in ("mix.5 DM", "mix.5 DMS"):
        bw.decide(mean, "mix.5", cand)


def verifier(mixed="X1abETmS", seeds=es.SEEDS, vseeds=(0, 1)):
    """On the mix's decode (rows X1abETmS, transplant draw 0): no verifier; v17's (trained on the learned + true dead ball
    rows of the other matches, J1abETmS); + the mix rows of the other matches; only the mix rows. Before and after the
    boundary timing."""
    import exp_rules_ctx as rc
    shipped = "J1abETmS"
    keys = exp_kpu.feats(shipped)
    names = ("decoded", "shipped", "added", "mix only")
    cells, acc = {}, {}
    for seed in seeds:
        ship = rc.rows_of(shipped, seed, ev.DEADS)
        mix = rc.rows_of(mixed, seed, (ev.DEADS[0],))
        test = [r for src in exp_kpu.SRCS for r in mix[(src, ev.DEADS[0])]["rows"]]
        tr_ship = [r for v in ship.values() for r in v["rows"]]
        for q in vseeds:
            ps = {"shipped": rc.lomo(tr_ship, test, keys, q), "added": rc.lomo(tr_ship + test, test, keys, q), "mix only": rc.lomo(test, test, keys, q)}
            n0 = 0
            for src in exp_kpu.SRCS:
                v = mix[(src, ev.DEADS[0])]
                idx = np.arange(n0, n0 + len(v["rows"]))
                n0 += len(v["rows"])
                blocks = bench.blocks(src)
                ref = sd.decode_scores(learned(src), team(src)[seed])
                by = {}
                for n, r in enumerate(v["rows"]):
                    by.setdefault(r["i"], []).append(n)
                labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in CONVS[src]}
                ins = np.array([r["typ"] == "ins" for r in v["rows"]])
                for arm in names:
                    if arm == "decoded":
                        preds = [np.asarray(b) for b in v["base"]]
                    else:
                        pp = ps[arm][idx]
                        preds = [ev.apply(v["base"][i], [v["rows"][n] for n in by.get(i, [])], pp[by.get(i, [])], True, True, 0.65, 0.35) for i in range(len(blocks))]
                        acc.setdefault((src, arm), []).append(((pp[ins] >= 0.65).sum() / (len(blocks) / 2), (pp[~ins] < 0.35).sum() / (len(blocks) / 2), np.quantile(pp[ins], 0.9), np.quantile(pp[ins], 0.99)))
                    for c in CONVS[src]:
                        cells.setdefault((src, c, TAG, arm), []).append(er.by_match(blocks, labs[c], preds))
                        cells.setdefault((src, c, TAG, arm + "+DT"), []).append(er.by_match(blocks, labs[c], [dt.retime(np.asarray(x), np.asarray(y)) for x, y in zip(preds, ref)]))
                bench._blocks.clear()
                bench._labels.clear()
        print("  seed %d done" % seed, flush=True)
    exp_kpu.use("v14")
    all_names = [n for a in names for n in (a, a + "+DT")]
    mean, _ = er.summarize(cells, all_names, ref="decoded")
    print("\naccepted per match: inserts, deletes, insert p 90% / 99%")
    for (src, arm), v in acc.items():
        print("  %-8s %-9s %.1f %.1f %.2f %.2f" % ((src.split("@")[0], arm) + tuple(np.mean(v, axis=0))), flush=True)
    import exp_blend_weight as bw
    for cand in ("shipped", "added", "mix only", "decoded+DT", "shipped+DT"):
        bw.decide(mean, "decoded", cand)


def shape():
    """Screen around the 0.5 mix (seed 0, draw 0): how the learned log-odds are bounded, stronger view weights (the flag's log-likelihood
    ratio on SkillCorner: +1.2 valid, -2.5 invalid), a bias towards alive, decoder penalties for the new evidence scale."""
    arms_def = {"mix.5": {}, "clip6": dict(clip=6.0), "clip4": dict(clip=4.0), "view llr": dict(view=True), "bias+.5": dict(bias=0.5), "bias+1": dict(bias=1.0), "bias-.5": dict(bias=-0.5),
                "pen20": dict(pen_dead=20.0), "pen45": dict(pen_dead=45.0), "min25": dict(min_dead=25), "min75": dict(min_dead=75)}
    cells = {}
    for src, convs in CONVS.items():
        blocks = bench.blocks(src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        a_lrn = learned(src)
        smo = team(src)[es.SEEDS[0]]
        a_rb = rule_view(src, blocks, labs[convs[0]], 0)
        a_llr = None
        if True:
            a_rule = ds.rule_evidence(src, blocks)
            if src.startswith("sc"):
                a_llr = [a + (2.5 - P["w_binvalid"]) * (~v) * -1 + 1.2 * v - P["w_blive"] * (v & (n >= 4)) for a, (v, n) in zip(a_rule, sd.view_cues())]
            else:
                a_llr = []
                for i, (a, g) in enumerate(zip(a_rule, labs[convs[0]])):
                    v, l = transplant(g, np.random.default_rng([SEED, 0, i]))
                    a_llr.append(a - 2.5 * (~v) + 1.2 * v)
        for name, o in arms_def.items():
            lrn = [np.clip(x, -o["clip"], o["clip"]) for x in a_lrn] if "clip" in o else a_lrn
            base = a_llr if o.get("view") else a_rb
            ev2 = [a + WEIGHT * b + o.get("bias", 0.0) for a, b in zip(base, lrn)]
            preds = sd.decode_scores(ev2, smo, pen_dead=o.get("pen_dead", 30.0), min_dead=o.get("min_dead", 50))
            for c in convs:
                cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  %s done" % src, flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    er.summarize(cells, list(arms_def), ref="mix.5")


def head():
    """The specialist's own dead head (J1, a sequence model: PFF + the ten games, held out by match) as evidence, alone and with the
    LightGBM dead model. Seed-paired, transplant draw 0."""
    import exp_blend_weight as bw
    names = ("mix.5", "J1 alone", "mix.5 J1", "mix.5 mean", "mix.5 sum", "mix.75 mean")
    cells = {}
    for src, convs in CONVS.items():
        blocks = bench.blocks(src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        a_lrn = learned(src)
        smos = team(src)
        outs = exp_kpu.outputs("J1", src)
        a_rb = rule_view(src, blocks, labs[convs[0]], 0)
        for k in es.SEEDS:
            a_j = [dt.alive_logit(np.asarray(o, np.float64)[:, 2]) for o in outs[ss.SEEDS[k]]]
            res = {"mix.5": mix_evidence(a_rb, a_lrn), "J1 alone": a_j, "mix.5 J1": mix_evidence(a_rb, a_j),
                   "mix.5 mean": [r + 0.25 * x + 0.25 * y for r, x, y in zip(a_rb, a_lrn, a_j)],
                   "mix.5 sum": [r + 0.5 * x + 0.5 * y for r, x, y in zip(a_rb, a_lrn, a_j)],
                   "mix.75 mean": [r + 0.375 * x + 0.375 * y for r, x, y in zip(a_rb, a_lrn, a_j)]}
            for name, ev2 in res.items():
                preds = sd.decode_scores(ev2, smos[k])
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            if k == es.SEEDS[0]:
                truth = labs[convs[0]]
                print("  %s: J1 alive logit q05/50/95 %s | dead F1 learned %.3f, J1 %.3f, mix.5 %.3f, mix.5 mean %.3f" % (
                    src, np.round(np.quantile(np.concatenate(a_j), (.05, .5, .95)), 2), _dead_f1(sd.decode_scores(a_lrn, smos[k]), truth), _dead_f1(sd.decode_scores(a_j, smos[k]), truth),
                    _dead_f1(sd.decode_scores(res["mix.5"], smos[k]), truth), _dead_f1(sd.decode_scores(res["mix.5 mean"], smos[k]), truth)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="mix.5")
    for cand in names[1:]:
        bw.decide(mean, "mix.5", cand)


def _stretches(d):
    return [(a, e) for a, e in segs(d) if d[a]]


def oracle():
    """Where is the rest of the true-dead gap of the mix? Seed 0, draw 0. existence: true dead stretches with no overlap get added, predicted ones with no overlap removed, the rest untouched; timing: overlapping
    stretches take the true boundaries, none added or removed; stops / restarts: only those; both. Team probability re-decoded with the resulting dead mask as hard evidence, so the team side is the decoder's."""
    for src, conv in (("dfl@v3", "evctrl"), ("metrica@v3", "ctrl"), ("sc@v3", "evctrl")):
        blocks = bench.blocks(src)
        truth = [np.asarray(g) for g in bench.labels(src, conv)]
        a_lrn = learned(src)
        smo = team(src)[es.SEEDS[0]]
        a_rb = rule_view(src, blocks, truth, 0)
        mix = sd.decode_scores(mix_evidence(a_rb, a_lrn), smo)
        masks = {k: [] for k in ("existence", "timing", "stops", "restarts", "both")}
        counts = np.zeros(4)
        for pr, g in zip(mix, truth):
            dp, dg = np.asarray(pr) == 2, g == 2
            sp, sg = _stretches(dp), _stretches(dg)
            ex = dp.copy()
            for a, e in sp:
                if not dg[a:e].any():
                    ex[a:e] = False
                    counts[0] += 1
            for a, e in sg:
                if not dp[a:e].any():
                    ex[a:e] = True
                    counts[1] += 1
            masks["existence"].append(ex)
            tm, st, re_ = dp.copy(), dp.copy(), dp.copy()
            for a, e in sp:
                ov = [(x, y) for x, y in sg if x < e and a < y]
                if not ov:
                    continue
                x0, y1 = ov[0][0], ov[-1][1]
                tm[a:e] = False
                tm[x0:y1] = True
                st[a:e] = False
                st[x0:e] = True
                re_[a:e] = False
                re_[a:y1] = True
                counts[2] += len(ov) > 1
            masks["timing"].append(tm)
            masks["stops"].append(st)
            masks["restarts"].append(re_)
            bo = tm.copy()
            for a, e in sp:
                if not dg[a:e].any():
                    bo[a:e] = False
            for a, e in sg:
                if not dp[a:e].any():
                    bo[a:e] = True
            masks["both"].append(bo)
        base = np.mean(list(er.by_match(blocks, truth, mix).values()))
        tr = np.mean(list(er.by_match(blocks, truth, sd.decode_scores([np.where(g == 2, -12.0, 12.0) for g in truth], smo)).values()))
        print("\n%s %s: mix %.4f, true dead %.4f | per match: predicted stretches with no true overlap %.1f, true with no predicted overlap %.1f, predicted overlapping several true %.1f" % (
            src.split("@")[0], conv, base, tr, counts[0] / (len(truth) / 2), counts[1] / (len(truth) / 2), counts[2] / (len(truth) / 2)))
        for k, ms2 in masks.items():
            preds = sd.decode_scores([np.where(m, -12.0, 12.0) for m in ms2], smo)
            print("  %-10s %+.4f" % (k, np.mean(list(er.by_match(blocks, truth, preds).values())) - base), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


def _base_of(src):
    return src.split("@")[0]


def lopo_probs(src):
    """P(dead) per half from a dead model on the OTHER providers only: how a source never seen would fare. Trained on de-glued Metrica, DFL, SkillCorner minus the source itself."""
    path = os.path.join(bench.CACHE, "dead", "LOPO_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    train = {"dfl": (bench.features("base", "dfl@v3"), bench.labels("dfl", "native")),
             "metrica": (bench.features("base", "metrica@v3g"), bench.labels("metrica@v3", "ctrl")),
             "sc": (bench.features("base", "sc@v3"), bench.labels("sc@v3", "evctrl"))}
    X, Y = [], []
    for name, (fs, ls) in train.items():
        if name != _base_of(src):
            X += list(fs)
            Y += list(ls)
    model = bench.fit_dead(X, Y)
    res = [model.predict(x["XF"], num_threads=bench.THREADS) for x in bench.features("base", src)]
    bench._dump(res, path)
    print("    LOPO %s done" % src, flush=True)
    return res


def view_only(src, blocks, truth, draw):
    """The view flag's terms alone."""
    a_rule = ds.rule_evidence(src, blocks)
    a_rb = rule_view(src, blocks, truth, draw)
    if src.startswith("sc"):
        import model_v2
        return [a - model_v2.aliveness(exp_blend.rule_f(b), P) for a, b in zip(a_rb, blocks)]
    return [a - b for a, b in zip(a_rb, a_rule)]


def multi_source2():
    """How to use the multi-source models. M alone, M + view flag, rules + view + w x M, for M = DMS (all three providers, held out by match) and LOPO (the two other providers), against the 0.5 mix with the DFL model. Transplant draw 0, 3 seeds."""
    import exp_blend_weight as bw
    names = ["mix.5"]
    for m in ("DMS", "LOPO"):
        names += [m, m + "+view", m + " r.5", m + " r1", m + " r2"]
    cells = {}
    for src, convs in CONVS.items():
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        models = {"DMS": [exp_blend.logit_alive(x) for x in dm_probs("DMS", ev_src)], "LOPO": [exp_blend.logit_alive(x) for x in lopo_probs(ev_src)]}
        smos = team(src)
        a_rb = rule_view(ev_src, blocks, labs[convs[0]], 0)
        a_v = view_only(ev_src, blocks, labs[convs[0]], 0)
        for m, al in models.items():
            print("  %s %s alive logit q05/50/95 %s" % (src, m, np.round(np.quantile(np.concatenate(al), (.05, .5, .95)), 2)), flush=True)
        for k in es.SEEDS:
            res = {"mix.5": mix_evidence(a_rb, a_lrn)}
            for m, al in models.items():
                res[m] = al
                res[m + "+view"] = [x + v for x, v in zip(al, a_v)]
                for w, nm in ((0.5, " r.5"), (1.0, " r1"), (2.0, " r2")):
                    res[m + nm] = mix_evidence(a_rb, al, w)
            for name, ev2 in res.items():
                preds = sd.decode_scores(ev2, smos[k])
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            print("  %s seed %d done" % (src, k), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, names, ref="mix.5")
    for cand in names[1:]:
        bw.decide(mean, "mix.5", cand)


def decode_norm(a_list, phl, c=1.0, pen_team=12.0, pen_dead=30.0, min_poss=8, min_dead=50):
    """decode_scores with the team uncertainty taken out of the live / dead contest: a single path pays log max(q, 1 - q) per live frame, which a sum over the two teams would not (Viterbi's one-path bias towards D at contested moments). c = 1 removes it fully."""
    import decode2
    import exp_v7
    sp = decode2.duration_spec(pen_team, pen_dead, min_poss, min_dead, exp_v7.ramp(175, 48.0))
    out = []
    for a, ph in zip(a_list, phl):
        q = np.clip(ph, sd.EPS, 1 - sd.EPS)
        n = -c * np.log(np.maximum(q, 1 - q))
        out.append(sp.decode(np.stack([a / 2 + np.log(q) + n, a / 2 + np.log(1 - q) + n, -a / 2], 1)).astype(np.int8))
    return out


def qnorm():
    import exp_blend_weight as bw
    names = ("mix.5", "qnorm", "qnorm.5", "rules+DT", "rules qnorm+DT")
    cells = {}
    for src, convs in CONVS.items():
        blocks = bench.blocks(src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        a_lrn = learned(src)
        smos = team(src)
        a_rb = rule_view(src, blocks, labs[convs[0]], 0)
        ev2 = mix_evidence(a_rb, a_lrn)
        for k in es.SEEDS:
            ref = sd.decode_scores(a_lrn, smos[k])
            res = {"mix.5": sd.decode_scores(ev2, smos[k]), "qnorm": decode_norm(ev2, smos[k]), "qnorm.5": decode_norm(ev2, smos[k], 0.5),
                   "rules+DT": [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_rb, smos[k]), ref)], "rules qnorm+DT": [dt.retime(r, l) for r, l in zip(decode_norm(a_rb, smos[k]), ref)]}
            for name, preds in res.items():
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
            if k == es.SEEDS[0]:
                truth = labs[convs[0]]
                print("  %s: dead F1 mix.5 %.3f, qnorm %.3f; dead share %.3f, %.3f, true %.3f" % (src, _dead_f1(res["mix.5"], truth), _dead_f1(res["qnorm"], truth), np.mean(np.concatenate(res["mix.5"]) == 2), np.mean(np.concatenate(res["qnorm"]) == 2), np.mean(np.concatenate(truth) == 2)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="mix.5")
    for cand in ("qnorm", "qnorm.5"):
        bw.decide(mean, "mix.5", cand)


SOURCES = {"dfl": ("dfl@v3", "dfl", "native"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}


def fit_balanced(parts):
    """A dead model on several providers, each provider the same total weight: parts = {provider: (features, labels)} (bench.fit_dead's parameters)."""
    import lightgbm as lgb
    Xs, ys, ws = [], [], []
    for fs, ls in parts.values():
        x = np.concatenate([f["XF"][::bench.STRIDE] for f in fs])
        y = np.concatenate([np.asarray(l)[::bench.STRIDE] for l in ls])
        g = y < 3
        Xs.append(x[g])
        ys.append((y[g] == 2).astype(int))
        ws.append(np.full(int(g.sum()), 1.0 / int(g.sum())))
    w = np.concatenate(ws)
    return lgb.train(bench.LGB_DEAD, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys), weight=w * len(w) / w.sum()), bench.ROUNDS)


def balanced_probs(name, src):
    """DMSb: all three providers, equal weight, held out by match (SkillCorner by five folds). LOPOb: the two other providers, equal weight. Cached."""
    path = os.path.join(bench.CACHE, "dead", "%s_%s.pkl" % (name, src.replace("@", "-")))
    if os.path.isfile(path):
        return bench._load(path)
    train = {k: (bench.features("base", v[0]), bench.labels(v[1], v[2]), _groups(bench.blocks(v[0]))) for k, v in SOURCES.items()}
    own = _base_of(src)
    fsrc = bench.features("base", src)
    gr = _groups(bench.blocks(src))
    un = sorted(set(gr), key=str)
    if name == "LOPOb" or own not in train:
        folds = [[]]
    else:
        folds = [un[i::5] for i in range(5)] if own == "sc" else [[g] for g in un]
    res = [None] * len(fsrc)
    for fold in folds:
        parts = {}
        for k, (fs, ls, gs) in train.items():
            if name == "LOPOb" and k == own:
                continue
            keep = [i for i, g in enumerate(gs) if g not in fold]
            parts[k] = ([fs[i] for i in keep], [ls[i] for i in keep])
        model = fit_balanced(parts)
        for i, (x, g) in enumerate(zip(fsrc, gr)):
            if (not fold) or g in fold:
                res[i] = model.predict(x["XF"], num_threads=bench.THREADS)
        print("    %s %s fold %s done" % (name, src, fold), flush=True)
    bench._dump(res, path)
    return res


def multi_source3():
    """Provider-balanced training, and the decode profile that goes with each: against the 0.5 mix, all with the view flag. Transplant draw 0, 3 seeds."""
    import exp_blend_weight as bw
    import metrica_sample as ms
    names = ("mix.5", "DMS+view", "DMSb+view", "LOPO+view", "LOPOb+view")
    cells = {}
    for src, convs in CONVS.items():
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        models = {"DMS": dm_probs("DMS", ev_src), "DMSb": balanced_probs("DMSb", ev_src), "LOPO": lopo_probs(ev_src), "LOPOb": balanced_probs("LOPOb", ev_src)}
        models = {k: [exp_blend.logit_alive(x) for x in v] for k, v in models.items()}
        smos = team(src)
        truth = labs[convs[0]]
        a_rb = rule_view(ev_src, blocks, truth, 0)
        a_v = view_only(ev_src, blocks, truth, 0)
        nm = len(truth) / 2
        print("\n%s: true dead share %.3f, possessions / match %.1f (< 4 s %.1f)" % (src, np.mean(np.concatenate(truth) == 2), sum(len(ms.spells(g)) for g in truth) / nm, sum(sum(e - a < 100 for a, e, _ in ms.spells(g)) for g in truth) / nm), flush=True)
        for k in es.SEEDS:
            res = {"mix.5": sd.decode_scores(mix_evidence(a_rb, a_lrn), smos[k])}
            for m, al in models.items():
                res[m + "+view"] = sd.decode_scores([x + v for x, v in zip(al, a_v)], smos[k])
            for name, preds in res.items():
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
                if k == es.SEEDS[0]:
                    print("  %-11s dead share %.3f | possessions / match %.1f (< 4 s %.1f) | dead F1 %.3f" % (name, np.mean(np.concatenate(preds) == 2), sum(len(ms.spells(p)) for p in preds) / nm, sum(sum(e - a < 100 for a, e, _ in ms.spells(p)) for p in preds) / nm, _dead_f1(preds, truth)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="mix.5")
    for cand in names[1:]:
        bw.decide(mean, "mix.5", cand)
    bw.decide(mean, "DMS+view", "DMSb+view")


_pff = {}


def pff_part():
    """PFF as a dead-model provider: the prepared per-half features (raw broadcast-domain tracking, no extra degradation), 'base' columns only, labels with their D from the events, unknown frames masked."""
    if _pff:
        return _pff["part"]
    import gradient
    import pathlib
    names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    fs, ls = [], []
    for match in gradient.matches():
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                idx = [f["fk"].index(k) for k in names]
                fs.append({"XF": f["XF"][:, idx].astype(np.float32)})
                ls.append(np.asarray(d["h%d_labels" % int(per)]))
    print("    PFF: %d halves, dead share %.3f" % (len(fs), np.mean(np.concatenate([l[l < 3] for l in ls]) == 2)), flush=True)
    _pff["part"] = (fs, ls)
    return _pff["part"]


def pff_probs(src):
    """DMSPb: DFL, Metrica, SkillCorner, PFF, equal provider weight, held out by match on the source itself. Cached."""
    path = os.path.join(bench.CACHE, "dead", "DMSPb_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    train = {k: (bench.features("base", v[0]), bench.labels(v[1], v[2]), _groups(bench.blocks(v[0]))) for k, v in SOURCES.items()}
    own = _base_of(src)
    fsrc = bench.features("base", src)
    gr = _groups(bench.blocks(src))
    un = sorted(set(gr), key=str)
    folds = ([un[i::5] for i in range(5)] if own == "sc" else [[g] for g in un]) if own in train else [[]]
    res = [None] * len(fsrc)
    for fold in folds:
        parts = {"pff": pff_part()}
        for k, (fs, ls, gs) in train.items():
            keep = [i for i, g in enumerate(gs) if g not in fold]
            parts[k] = ([fs[i] for i in keep], [ls[i] for i in keep])
        model = fit_balanced(parts)
        for i, (x, g) in enumerate(zip(fsrc, gr)):
            if (not fold) or g in fold:
                res[i] = model.predict(x["XF"], num_threads=bench.THREADS)
        print("    DMSPb %s fold %s done" % (src, fold), flush=True)
    bench._dump(res, path)
    return res


def multi_source4():
    """PFF added to the balanced multi-source dead model. Transplant draw 0, 3 seeds."""
    import exp_blend_weight as bw
    import metrica_sample as ms
    names = ("DMSb+view", "DMSPb+view", "mix.5")
    cells = {}
    for src, convs in CONVS.items():
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        a_b = [exp_blend.logit_alive(x) for x in balanced_probs("DMSb", ev_src)]
        a_p = [exp_blend.logit_alive(x) for x in pff_probs(ev_src)]
        smos = team(src)
        truth = labs[convs[0]]
        a_rb = rule_view(ev_src, blocks, truth, 0)
        a_v = view_only(ev_src, blocks, truth, 0)
        nm = len(truth) / 2
        for k in es.SEEDS:
            res = {"mix.5": sd.decode_scores(mix_evidence(a_rb, a_lrn), smos[k]), "DMSb+view": sd.decode_scores([x + v for x, v in zip(a_b, a_v)], smos[k]), "DMSPb+view": sd.decode_scores([x + v for x, v in zip(a_p, a_v)], smos[k])}
            for name, preds in res.items():
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
                if k == es.SEEDS[0]:
                    print("  %-8s %-11s dead share %.3f | possessions / match %.1f (< 4 s %.1f) | dead F1 %.3f" % (src.split("@")[0], name, np.mean(np.concatenate(preds) == 2), sum(len(ms.spells(p)) for p in preds) / nm, sum(sum(e - a < 100 for a, e, _ in ms.spells(p)) for p in preds) / nm, _dead_f1(preds, truth)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="DMSb+view")
    bw.decide(mean, "DMSb+view", "DMSPb+view")


def dmsb_evidence(src, draw=0):
    """Round 41's dead-ball evidence on an open source: the provider-balanced multi-source dead model's alive log-odds (held out by match) + the view flag's terms. The evidence comes from de-glued Metrica."""
    ev_src = "metrica@v3g" if src.startswith("metrica") else src
    blocks = bench.blocks(ev_src)
    truth = [np.asarray(g) for g in bench.labels(src, CONVS[src][0])]
    return [exp_blend.logit_alive(x) + v for x, v in zip(balanced_probs("DMSb", ev_src), view_only(ev_src, blocks, truth, draw))]


def e1_evidence(src, draw=0):
    """E1, round 41's evidence on an open source: rules + view + 0.5 x the DFL dead model + 0.5 x the multi-source dead model, each channel held out from the match. The evidence comes from de-glued Metrica."""
    ev_src = "metrica@v3g" if src.startswith("metrica") else src
    blocks = bench.blocks(ev_src)
    truth = [np.asarray(g) for g in bench.labels(src, CONVS[src][0])]
    base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
    return [r + 0.5 * exp_blend.logit_alive(x) + 0.5 * exp_blend.logit_alive(y) for r, x, y in zip(rule_view(ev_src, blocks, truth, draw), base, balanced_probs("DMSb", ev_src))]


def evaluate2():
    """Round 41's decision run: E1 and the others against the 0.5 mix and v17's recipe, 3 seeds, 3 transplant draws (SkillCorner its real flag). E1 lopo: the multi-source channel from the other providers only."""
    import exp_blend_weight as bw
    names = ("rules+DT", "mix.5", "DMSb+view", "E1", "E1 lopo")
    cells = {}
    for src, convs in CONVS.items():
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        a_m = [exp_blend.logit_alive(x) for x in balanced_probs("DMSb", ev_src)]
        a_o = [exp_blend.logit_alive(x) for x in balanced_probs("LOPOb", ev_src)]
        smos = team(src)
        truth = labs[convs[0]]
        for draw in (DRAWS if not src.startswith("sc") else (0,)):
            a_rb = rule_view(ev_src, blocks, truth, draw)
            a_v = view_only(ev_src, blocks, truth, draw)
            for k in es.SEEDS:
                ref = sd.decode_scores(a_lrn, smos[k])
                res = {"rules+DT": [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_rb, smos[k]), ref)], "mix.5": sd.decode_scores(mix_evidence(a_rb, a_lrn), smos[k]),
                       "DMSb+view": sd.decode_scores([m + v for m, v in zip(a_m, a_v)], smos[k]), "E1": sd.decode_scores([r + 0.5 * l + 0.5 * m for r, l, m in zip(a_rb, a_lrn, a_m)], smos[k]),
                       "E1 lopo": sd.decode_scores([r + 0.5 * l + 0.5 * m for r, l, m in zip(a_rb, a_lrn, a_o)], smos[k])}
                for name, preds in res.items():
                    for c in convs:
                        cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
                if draw == 0 and k == es.SEEDS[0]:
                    print("  %s: dead F1 %s" % (src, " ".join("%s %.3f" % (n, _dead_f1(res[n], truth)) for n in names)), flush=True)
            print("  %s draw %d done" % (src, draw), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="mix.5")
    for cur, cand in (("mix.5", "E1"), ("mix.5", "E1 lopo"), ("rules+DT", "E1"), ("E1", "DMSb+view")):
        bw.decide(mean, cur, cand)


def soccertrack4():
    """SoccerTrack v2, the channels and their sums, 3 seeds, 3 draws (its rule channel is broken: its ball is rebuilt from the events)."""
    import exp_st
    src = exp_st.SRC
    blocks = bench.blocks(src)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in exp_st.CONVS}
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % src.replace("@", "-")))
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", src)]
    a_m = [exp_blend.logit_alive(x) for x in balanced_probs("DMSb", src)]
    names = ("mix.5", "E1", "DMSb+view", "DFL+view", "both+view")
    cells = {}
    for draw in DRAWS:
        a_rb = rule_view(src, blocks, labs["evctrl"], draw)
        a_v = view_only(src, blocks, labs["evctrl"], draw)
        for k in es.SEEDS:
            raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
            smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
            res = {"mix.5": mix_evidence(a_rb, a_lrn), "E1": [r + 0.5 * l + 0.5 * m for r, l, m in zip(a_rb, a_lrn, a_m)], "DMSb+view": [m + v for m, v in zip(a_m, a_v)],
                   "DFL+view": [m + v for m, v in zip(a_lrn, a_v)], "both+view": [0.5 * l + 0.5 * m + v for l, m, v in zip(a_lrn, a_m, a_v)]}
            for name, ev2 in res.items():
                preds = sd.decode_scores(ev2, smo)
                for c in exp_st.CONVS:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  draw %d done" % draw, flush=True)
    er.summarize(cells, list(names), ref="mix.5")


def pff_test():
    """Round 42: PFF's 64 World Cup matches as an independent test of the dead-ball evidence: neither dead model nor the team model
    below ever saw them, and they played no part in designing the fusion. Raw broadcast-domain tracking (no extra degradation), labels in the
    controlled reading, the view flag transplanted, the all-DFL team LightGBM of v7 (not the blend: its BiGRU and specialist trained on PFF).
    The dead models: all-DFL (models/dead_base_v3.txt) and the all-three-providers one (cache/bench/dead/dead_multi_v1.txt)."""
    import gradient
    import lightgbm as lgb
    import model_v2
    import model_v7
    import pathlib
    names = ("rules+DT", "mix.5", "L+view", "M+view", "both+view", "r.5 both", "E1")
    tag = "%s_native_v3" % bench._safe(model_v7.VERSION)
    team_model = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", tag + ".txt"))
    l_model = bench.load_booster(os.path.join(bench.ROOT, "models", "dead_base_v3.txt"))
    m_model = bench.load_booster(os.path.join(bench.CACHE, "dead", "dead_multi_v1.txt"))
    base_names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    cells = {}
    n = 0
    tot = {k: [0, 0, 0] for k in ("rules", "L", "M", "true")}
    for match in gradient.matches():
        blocks, labs, a_t, a_v, a_l, a_m, smo = [], [], [], [], [], [], []
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                tagp = "h%d" % int(per)
                b = dict(P=d[tagp + "_players"], B=d[tagp + "_ball"], team=d["team"], gk=d["gk"], pitch=tuple(d["pitch"]), group=match)
                lab = np.asarray(d[tagp + "_labels"])
                idx = [f["fk"].index(k) for k in base_names]
                x = f["XF"][:, idx]
                blocks.append(b)
                labs.append(lab)
                a_t.append(model_v2.aliveness(exp_blend.rule_f(b), P))
                v, l = transplant(lab, np.random.default_rng([SEED, n]))
                a_v.append(-P["w_binvalid"] * (~v) + P["w_blive"] * l)
                a_l.append(exp_blend.logit_alive(l_model.predict(x, num_threads=bench.THREADS)))
                a_m.append(exp_blend.logit_alive(m_model.predict(x, num_threads=bench.THREADS)))
                smo.append(bench.predict_team(team_model, f))
                n += 1
        if not blocks:
            continue
        smo = exp_smooth.smooth(smo, 25)
        rb = [a + v for a, v in zip(a_t, a_v)]
        ref = sd.decode_scores(a_l, smo)
        res = {"rules+DT": [dt.retime(r, l) for r, l in zip(sd.decode_scores(rb, smo), ref)], "mix.5": sd.decode_scores(mix_evidence(rb, a_l), smo),
               "L+view": sd.decode_scores([x + v for x, v in zip(a_l, a_v)], smo), "M+view": sd.decode_scores([x + v for x, v in zip(a_m, a_v)], smo),
               "both+view": sd.decode_scores([0.5 * x + 0.5 * y + v for x, y, v in zip(a_l, a_m, a_v)], smo),
               "r.5 both": sd.decode_scores([0.5 * t + 0.5 * x + 0.5 * y + v for t, x, y, v in zip(a_t, a_l, a_m, a_v)], smo),
               "E1": sd.decode_scores([r + 0.5 * x + 0.5 * y for r, x, y in zip(rb, a_l, a_m)], smo)}
        for name, preds in res.items():
            cells.setdefault(("pff", "ctrl", TAG, name), []).append(er.by_match(blocks, labs, preds))
        print("  %s done" % match, flush=True)
    merged = {}
    for k, v in cells.items():
        d = {}
        for x in v:
            d.update(x)
        merged[k] = [d]
    er.summarize(merged, list(names), ref="mix.5")


def versions(src):
    """v16, v17 and v18 side by side in BAR-ATH's condition, paired: same seeds, same transplant draws, the same evidence input (Metrica de-glued, for all three). v16 = C1 + rules + view + timing, v17 = J1 + rules + view + timing, v18 = J1 + rules + view + 0.5 x learned; all before the verifier. One source per process."""
    convs = CONVS[src]
    ev_src = "metrica@v3g" if src.startswith("metrica") else src
    blocks = bench.blocks(ev_src)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
    base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
    a_lrn = [exp_blend.logit_alive(x) for x in base]
    smo16, smo17 = team(src, "G1"), team(src, "J1")
    cells = {}
    for draw in (DRAWS if not src.startswith("sc") else (0,)):
        a_rb = rule_view(ev_src, blocks, labs[convs[0]], draw)
        for k in es.SEEDS:
            res = {"v16": [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_rb, smo16[k]), sd.decode_scores(a_lrn, smo16[k]))],
                   "v17": [dt.retime(r, l) for r, l in zip(sd.decode_scores(a_rb, smo17[k]), sd.decode_scores(a_lrn, smo17[k]))],
                   "v18": sd.decode_scores(mix_evidence(a_rb, a_lrn), smo17[k])}
            for name, preds in res.items():
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  draw %d done" % draw, flush=True)
    er.summarize(cells, ["v16", "v17", "v18"], ref="v16")


def stress2():
    """Round 41's bet: the rules go. What if the rules are better on the target than anything measured (kappa x the truth) or the multi-source model worse (slow noise sigma on its and the DFL model's log-odds)? Seed 0, draw 0, SkillCorner's first ten matches. Also the hedges rules + 1 / 2 x the model."""
    names = ("rules+DT", "mix.5", "DMSb+view", "r1", "r2")
    regret = {n: {} for n in names}
    for src, conv in (("dfl@v3", "evctrl"), ("metrica@v3", "ctrl"), ("sc@v3", "evctrl")):
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        truth = [np.asarray(g) for g in bench.labels(src, conv)]
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        a_m = [exp_blend.logit_alive(x) for x in balanced_probs("DMSb", ev_src)]
        smo = team(src)[es.SEEDS[0]]
        a_rb = rule_view(ev_src, blocks, truth, 0)
        a_v = view_only(ev_src, blocks, truth, 0)
        if src.startswith("sc"):
            keep = list(range(20))
            blocks, truth, a_lrn, a_m, smo, a_rb, a_v = [blocks[i] for i in keep], [truth[i] for i in keep], [a_lrn[i] for i in keep], [a_m[i] for i in keep], [smo[i] for i in keep], [a_rb[i] for i in keep], [a_v[i] for i in keep]
        print("\n%s %s: kappa sigma | dead F1 rules, DFL model, multi-source | spellF1 %s" % (src.split("@")[0], conv, " ".join(names)), flush=True)
        for sigma in (0.0, 4.0):
            noise = [_ar1(len(a), np.random.default_rng([SEED, 11, i]), sigma) if sigma else 0.0 for i, a in enumerate(a_lrn)]
            lrn2 = [a + n for a, n in zip(a_lrn, noise)]
            m2 = [a + n for a, n in zip(a_m, noise)]
            ref = sd.decode_scores(lrn2, smo)
            dmsb = sd.decode_scores([x + v for x, v in zip(m2, a_v)], smo)
            f_m = _dead_f1(sd.decode_scores(m2, smo), truth)
            for kappa in (0.0, 1.0, 2.0):
                rb2 = [a + kappa * np.where(g == 2, -1.0, 1.0) for a, g in zip(a_rb, truth)]
                rule = sd.decode_scores(rb2, smo)
                res = {"rules+DT": [dt.retime(r, l) for r, l in zip(rule, ref)], "mix.5": sd.decode_scores(mix_evidence(rb2, lrn2), smo), "DMSb+view": dmsb, "r1": sd.decode_scores(mix_evidence(rb2, m2, 1.0), smo), "r2": sd.decode_scores(mix_evidence(rb2, m2, 2.0), smo)}
                sc = {n: float(np.mean(list(er.by_match(blocks, truth, res[n]).values()))) for n in names}
                best = max(sc.values())
                for n in names:
                    regret[n][(src, kappa, sigma)] = best - sc[n]
                print("  %.0f %.0f | %.3f %.3f %.3f | %s" % (kappa, sigma, _dead_f1(rule, truth), _dead_f1(ref, truth), f_m, " ".join("%.4f" % sc[n] for n in names)), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    print("\nmax regret over all scenarios: " + " | ".join("%s %.4f" % (n, max(regret[n].values())) for n in names))
    for ks in ((0.0,), (0.0, 1.0)):
        sel = lambda k: k[1] in ks
        print("max regret, kappa in %s: " % (ks,) + " | ".join("%s %.4f" % (n, max(v for k, v in regret[n].items() if sel(k))) for n in names), flush=True)


def stress3(src):
    """Hedges: which sum of the channels keeps most of the multi-source model's gain and stays safe if a channel is much worse or better on the target than measured? kappa x the truth on the rules; slow noise sigma, drawn apart, on each learned channel. Seed 0, draw 0, SkillCorner's first ten matches. One source per process; the table goes to tools/runs/r41_stress3_<source>.json."""
    import json
    conv = CONVS[src][0]
    names = ("rules+DT", "mix.5", "DMSb+view", "E1", "E2", "E3", "E4")
    ev_src = "metrica@v3g" if src.startswith("metrica") else src
    blocks = bench.blocks(ev_src)
    truth = [np.asarray(g) for g in bench.labels(src, conv)]
    base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
    a_lrn = [exp_blend.logit_alive(x) for x in base]
    a_m = [exp_blend.logit_alive(x) for x in balanced_probs("DMSb", ev_src)]
    smo = team(src)[es.SEEDS[0]]
    a_rb = rule_view(ev_src, blocks, truth, 0)
    a_v = view_only(ev_src, blocks, truth, 0)
    if src.startswith("sc"):
        keep = list(range(20))
        blocks, truth, a_lrn, a_m, smo, a_rb, a_v = [blocks[i] for i in keep], [truth[i] for i in keep], [a_lrn[i] for i in keep], [a_m[i] for i in keep], [smo[i] for i in keep], [a_rb[i] for i in keep], [a_v[i] for i in keep]
    out = {}
    print("%s %s: kappa sigma | %s" % (src, conv, " ".join(names)), flush=True)
    for kappa, sigma in ((0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (0, 4), (1, 4), (2, 4)):
        lrn2 = [a + _ar1(len(a), np.random.default_rng([SEED, 21, i]), sigma) if sigma else a for i, a in enumerate(a_lrn)]
        m2 = [a + _ar1(len(a), np.random.default_rng([SEED, 22, i]), sigma) if sigma else a for i, a in enumerate(a_m)]
        rb2 = [a + kappa * np.where(g == 2, -1.0, 1.0) for a, g in zip(a_rb, truth)]
        ref = sd.decode_scores(lrn2, smo)
        ev2 = {"mix.5": [r + 0.5 * l for r, l in zip(rb2, lrn2)], "DMSb+view": [m + v for m, v in zip(m2, a_v)],
               "E1": [r + 0.5 * l + 0.5 * m for r, l, m in zip(rb2, lrn2, m2)], "E2": [r + 0.5 * l + m for r, l, m in zip(rb2, lrn2, m2)],
               "E3": [r + 0.25 * l + m for r, l, m in zip(rb2, lrn2, m2)], "E4": [r + 0.5 * l + 2 * m for r, l, m in zip(rb2, lrn2, m2)]}
        res = {k: sd.decode_scores(v, smo) for k, v in ev2.items()}
        res["rules+DT"] = [dt.retime(r, l) for r, l in zip(sd.decode_scores(rb2, smo), ref)]
        sc = {n: float(np.mean(list(er.by_match(blocks, truth, res[n]).values()))) for n in names}
        out["%d %d" % (kappa, sigma)] = sc
        print("  %d %d | %s" % (kappa, sigma, " ".join("%.4f" % sc[n] for n in names)), flush=True)
    with open(os.path.join(bench.ROOT, "tools", "runs", "r41_stress3_%s.json" % src.split("@")[0]), "w") as fh:
        json.dump(out, fh)


def soccertrack2():
    """SoccerTrack v2 (never used for the dead model or the design): rules+DT, the 0.5 mix, the multi-source model on an unseen provider, with the transplanted view flag."""
    import exp_st
    src = exp_st.SRC
    blocks = bench.blocks(src)
    labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in exp_st.CONVS}
    gru = exp_st.gru_mean()
    outs = bench._load(os.path.join(ss.OUT, "J1_%s.pkl" % src.replace("@", "-")))
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", src)]
    a_m = [exp_blend.logit_alive(x) for x in balanced_probs("DMSb", src)]
    names = ("mix.5", "DMSb+view", "DMSb", "learned")
    cells = {}
    for draw in DRAWS:
        a_rb = rule_view(src, blocks, labs["evctrl"], draw)
        a_v = view_only(src, blocks, labs["evctrl"], draw)
        for k in es.SEEDS:
            raw14 = es.blend(exp_st.v10_raw(k), gru, 0.5)
            smo = exp_smooth.smooth([0.75 * r + 0.25 * ss.pk(np.asarray(x, np.float32), 0) for r, x in zip(raw14, outs[ss.SEEDS[k]])], 25)
            res = {"mix.5": mix_evidence(a_rb, a_lrn), "DMSb+view": [x + v for x, v in zip(a_m, a_v)], "DMSb": a_m, "learned": a_lrn}
            for name, ev2 in res.items():
                preds = sd.decode_scores(ev2, smo)
                for c in exp_st.CONVS:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  draw %d done" % draw, flush=True)
    er.summarize(cells, list(names), ref="mix.5")


def head2():
    """Small screen on top of DMSb + view: the specialist's dead head, the DFL model, the rules' tracking terms. Draw 0, 3 seeds."""
    import exp_blend_weight as bw
    names = ("DMSb+view", "+.25 J1", "+.5 J1", "+.25 DFL", "+.25 rules")
    cells = {}
    for src, convs in CONVS.items():
        ev_src = "metrica@v3g" if src.startswith("metrica") else src
        blocks = bench.blocks(ev_src)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
        base = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", ev_src)
        a_lrn = [exp_blend.logit_alive(x) for x in base]
        a_e = dmsb_evidence(src)
        a_rb = rule_view(ev_src, blocks, labs[convs[0]], 0)
        smos = team(src)
        outs = exp_kpu.outputs("J1", src)
        for k in es.SEEDS:
            a_j = [dt.alive_logit(np.asarray(o, np.float64)[:, 2]) for o in outs[ss.SEEDS[k]]]
            res = {"DMSb+view": a_e, "+.25 J1": mix_evidence(a_e, a_j, 0.25), "+.5 J1": mix_evidence(a_e, a_j, 0.5), "+.25 DFL": mix_evidence(a_e, a_lrn, 0.25), "+.25 rules": mix_evidence(a_e, a_rb, 0.25)}
            for name, ev2 in res.items():
                preds = sd.decode_scores(ev2, smos[k])
                for c in convs:
                    cells.setdefault((src, c, TAG, name), []).append(er.by_match(blocks, labs[c], preds))
        print("  %s done" % src, flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(names), ref="DMSb+view")
    for cand in names[1:]:
        bw.decide(mean, "DMSb+view", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"eval": evaluate, "st": soccertrack, "v3g": deglued, "stress": stress, "dm": multi_source, "verifier": verifier, "shape": shape, "head": head, "oracle": oracle, "dm2": multi_source2, "qnorm": qnorm, "dm3": multi_source3, "dm4": multi_source4,
     "stress2": stress2, "st2": soccertrack2, "head2": head2, "stress3": stress3, "eval2": evaluate2, "st4": soccertrack4, "pff": pff_test, "versions": versions}[sys.argv[1]](*sys.argv[2:])
