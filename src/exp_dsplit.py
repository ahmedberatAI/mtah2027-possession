"""Round 26 (docs/RAPOR.md, pre-registered): dead-ball split (D-split). BAR-ATH's dead-ball side (v3's rules + the
aligned broadcast) keeps its boundaries; inside one of its dead stretches, a complete possession of the learned dead-ball
model's own decode (same team probability, same decoder), at least MARGIN frames from both edges of the stretch, is
added with its team and boundaries. No new threshold.

Judged with BAR-ATH-like dead-ball evidence on the external sets: v3's rules from tracking (DFL, Metrica) and rules +
view cues (SkillCorner, idea N21); v15's team probability (three v10 seeds, each paired with one C1 seed, +-25
smoothing), BAR-ATH's decoder settings, before the verifier (the verifier works inside possessions, the split inside dead
stretches; in production the split is the last step). AGGREGATE output only.

  python -I src/exp_dsplit.py eval        rules against rules + D-split, ten contexts + SkillCorner evctrl, decision rule
  python -I src/exp_dsplit.py barath      label-free: what the split would add to v15 on BAR-ATH
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_blend  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import model_v2  # noqa: E402
import model_v3  # noqa: E402
import short_specialist as ss  # noqa: E402

MARGIN = 25
CONVS = {"dfl@v3": ("evctrl", "trk"), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp", "evctrl")}
NAMES = ("rules", "rules + D-split")


def split(rule, learned, margin=MARGIN):
    """One half: the learned decode's possessions lying wholly inside a dead stretch of the rule decode, at least
    margin frames from both of its edges, are copied in."""
    out = np.asarray(rule).copy()
    lr = np.asarray(learned)
    st = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    for s, e, k in zip(st, np.r_[st[1:], len(out)], out[st]):
        if k != 2 or e - s <= 2 * margin:
            continue
        seg = lr[s + margin:e - margin]
        st2 = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        for a, b, kk in zip(st2, np.r_[st2[1:], len(seg)], seg[st2]):
            if kk < 2 and a > 0 and b < len(seg):
                out[s + margin + a:s + margin + b] = kk
    return out


def added(rule, new):
    """Possessions in new that are not in rule (frames that were dead), with their lengths in frames."""
    rule, new = np.asarray(rule), np.asarray(new)
    st = np.flatnonzero(np.r_[True, new[1:] != new[:-1]])
    return [b - a for a, b, k in zip(st, np.r_[st[1:], len(new)], new[st]) if k < 2 and np.all(rule[a:b] == 2)]


def rule_evidence(src, blocks):
    p = dict(model_v3.DEFAULT)
    if src.startswith("sc"):
        return sd.rule_alive(blocks, sd.view_cues(), p)
    return [model_v2.aliveness(exp_blend.rule_f(b), p) for b in blocks]


def evaluate():
    import exp_blend_weight as bw
    import exp_kpu
    import exp_refine as er
    import exp_verify as ev
    cells = {}
    for src, convs in CONVS.items():
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        a_rule = rule_evidence(src, blocks)
        dl = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", src)
        a_lrn = [exp_blend.logit_alive(x) for x in dl]
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
        outs = exp_kpu.outputs("C1", src)
        n_add, lens, alive = 0, [], 0.0
        for k in es.SEEDS:
            raws = [0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(es.blend(es.v10_raw(k, src), members[k], 0.5),
                                                                  outs[ss.SEEDS[k]])]
            smos = exp_smooth.smooth(raws, 25)
            rule = sd.decode_scores(a_rule, smos)
            spl = [split(r, lr) for r, lr in zip(rule, sd.decode_scores(a_lrn, smos))]
            for r, s2 in zip(rule, spl):
                ad = added(r, s2)
                n_add += len(ad)
                lens += ad
                alive += (np.asarray(r) < 2).sum() / 25 / 60
            for name, preds in zip(NAMES, (rule, spl)):
                for c in convs:
                    cells.setdefault((src, c, "rules D", name), []).append(er.by_match(blocks, labs[c], preds))
        print("%s: the split adds %.3f possessions per alive minute (median %.1f s, %.0f%% under 4 s)" % (
            src.split("@")[0], n_add / alive, np.median(lens) / 25 if lens else 0,
            100 * np.mean(np.array(lens) < 100) if lens else 0), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(NAMES), ref=NAMES[0])
    bw.regret_boot(mean, list(NAMES), NAMES[0], NAMES[1])
    bw.decide(mean, NAMES[0], NAMES[1])


def barath():
    import pandas as pd
    import model_v15
    lab = pd.read_csv(os.path.join(bench.ROOT, "submissions", "candidate_v15.csv")).iloc[:, 1].astype(str).to_numpy()
    code = {"H": 0, "A": 1, "D": 2}
    probs = bench._load(model_v15.ensure_probs())
    bench._barath_inputs()
    blocks = bench.blocks("barath")
    dl = bench.dead_models("base", "v3")["barath"]
    n_add, lens, changed, alive = 0, [], 0, 0.0
    for b, pdd in zip(blocks, dl):
        cur = np.array([code.get(v, 3) for v in lab[b["a"]:b["b"] + 1]], np.int8)
        lrn = sd.decode_scores([exp_blend.logit_alive(pdd)], [probs[b["a"]][1]])[0]
        new = split(cur, lrn)
        ad = added(cur, new)
        n_add += len(ad)
        lens += ad
        changed += int((new != cur).sum())
        alive += (cur < 2).sum() / 25 / 60
    print("BAR-ATH (label-free): the split would add %d possessions (%.3f per alive minute; median %.1f s, %d under 4 s),"
          " %.2f%% of the frames" % (n_add, n_add / alive, np.median(lens) / 25 if lens else 0,
                                    int(np.sum(np.array(lens) < 100)), 100 * changed / len(lab)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"eval": evaluate, "barath": barath}[sys.argv[1]]()
