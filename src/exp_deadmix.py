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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"eval": evaluate, "st": soccertrack, "v3g": deglued, "stress": stress}[sys.argv[1]]()
