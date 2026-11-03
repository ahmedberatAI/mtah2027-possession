"""Round 29 (docs/RAPOR.md, pre-registered): dead-ball boundary timing from the learned dead model. BAR-ATH's dead-ball
side (v3's rules + the aligned broadcast) keeps its dead stretches; a boundary of the rules decode is moved onto the
learned dead model's nearest boundary of the same type (stop: alive -> dead, restart: dead -> alive) within +-2 s,
when no other boundary lies in between. Two variants: DT (stops and restarts), DTr (restarts only). No new threshold.

Judged as round 26 (src/exp_dsplit.py): BAR-ATH-like dead-ball evidence on the external sets (v3's rules on DFL /
Metrica, rules + view cues on SkillCorner), v15's team probability (three v10 seeds, each paired with one C1 seed,
+-25 smoothing), BAR-ATH's decoder settings, before the verifier. SoccerTrack v2 (never used to design the round): v3's
rules from tracking, its provider labels, v15's open-data components (src/exp_st.py). AGGREGATE output only.

  python -I src/exp_dtime.py eval       rules against DT / DTr: decision contexts, SkillCorner evctrl, SoccerTrack
  python -I src/exp_dtime.py barath     label-free: how many of v15's BAR-ATH boundaries the variants would move
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_blend  # noqa: E402
import exp_dsplit as ds  # noqa: E402
import exp_scdead as sd  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402
import model_v2  # noqa: E402
import model_v3  # noqa: E402
import short_specialist as ss  # noqa: E402

from dead_timing import RESTART, STOP, dbounds, retime  # noqa: E402

NAMES = ("rules", "rules + DT", "rules + DTr")


def variants(rule, learned):
    return {NAMES[0]: rule, NAMES[1]: [retime(r, l) for r, l in zip(rule, learned)],
            NAMES[2]: [retime(r, l, (RESTART,)) for r, l in zip(rule, learned)]}


def evaluate():
    import exp_blend_weight as bw
    import exp_kpu
    import exp_refine as er
    import exp_verify as ev
    cells, info = {}, {}
    for src, convs in ds.CONVS.items():
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        a_rule = ds.rule_evidence(src, blocks)
        dl = bench.dead_probs(src, "base") if src.startswith("dfl") else bench.dead_ext("base", "v3", src)
        a_lrn = [exp_blend.logit_alive(x) for x in dl]
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
        outs = exp_kpu.outputs("C1", src)
        for k in es.SEEDS:
            raws = [0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(es.blend(es.v10_raw(k, src), members[k]),
                                                                  outs[ss.SEEDS[k]])]
            smos = exp_smooth.smooth(raws, 25)
            for name, preds in variants(sd.decode_scores(a_rule, smos), sd.decode_scores(a_lrn, smos)).items():
                for c in convs:
                    (info if (src, c) == ("sc@v3", "evctrl") else cells).setdefault(
                        (src, c, "rules D", name), []).append(er.by_match(blocks, labs[c], preds))
        print("  %s done" % src, flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    mean, _ = er.summarize(cells, list(NAMES), ref=NAMES[0])
    for name in NAMES[1:]:
        bw.regret_boot(mean, [NAMES[0], name], NAMES[0], name)
        bw.decide({key: v for key, v in mean.items() if key[3] in (NAMES[0], name)}, NAMES[0], name)
    print("\ninformation only, not in the decision rule:", flush=True)
    er.summarize(info, list(NAMES), ref=NAMES[0])
    st_check()


def st_check():
    """SoccerTrack v2 (8 labelled matches): v3's rules from tracking against the variants, v15's open-data parts."""
    import exp_refine as er
    import exp_st
    src = exp_st.SRC
    blocks = bench.blocks(src)
    labs = {c: bench.labels(src, c) for c in exp_st.CONVS}
    p = dict(model_v3.DEFAULT)
    a_rule = [model_v2.aliveness(exp_blend.rule_f(b), p) for b in blocks]
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", src)]
    gru, c1 = exp_st.gru_mean(), exp_st.c1_outputs()
    cells = {}
    for k in es.SEEDS:
        raws = [0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(es.blend(exp_st.v10_raw(k), gru, 0.5),
                                                              c1[ss.SEEDS[k]])]
        smos = exp_smooth.smooth(raws, 25)
        for name, preds in variants(sd.decode_scores(a_rule, smos), sd.decode_scores(a_lrn, smos)).items():
            for c in exp_st.CONVS:
                cells.setdefault((src, c, "rules D", name), []).append(er.by_match(blocks, labs[c], preds))
    print("\nSoccerTrack v2, independent check (not used to design the round):", flush=True)
    er.summarize(cells, list(NAMES), ref=NAMES[0])


def barath():
    import pandas as pd
    import model_v15
    lab = pd.read_csv(os.path.join(bench.ROOT, "submissions", "candidate_v15.csv")).iloc[:, 1].astype(str).to_numpy()
    code = {"H": 0, "A": 1, "D": 2}
    probs = bench._load(model_v15.ensure_probs())
    bench._barath_inputs()
    dl = bench.dead_models("base", "v3")["barath"]
    for name, kinds in (("DT", (STOP, RESTART)), ("DTr", (RESTART,))):
        moved, shifts, changed, n_b = 0, [], 0, 0
        for b, pdd in zip(bench.blocks("barath"), dl):
            cur = np.array([code.get(v, 3) for v in lab[b["a"]:b["b"] + 1]], np.int8)
            lrn = sd.decode_scores([exp_blend.logit_alive(pdd)], [probs[b["a"]][1]])[0]
            new = retime(cur, lrn, kinds)
            f0, k0 = dbounds(cur)
            n_b += int(np.isin(k0, kinds).sum())
            f1, _ = dbounds(new)
            if len(f1) == len(f0):
                d = f1 - f0
                moved += int((d != 0).sum())
                shifts += list(d[d != 0])
            changed += int((new != cur).sum())
        print("BAR-ATH (label-free) %s: %d of %d boundaries moved (median shift %+.0f frames, |.| median %.0f), "
              "%.2f%% of the frames" % (name, moved, n_b, np.median(shifts) if shifts else 0,
                                        np.median(np.abs(shifts)) if shifts else 0, 100 * changed / len(lab)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"eval": evaluate, "barath": barath}[sys.argv[1]]()
