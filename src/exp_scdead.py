"""Idea N21 (docs/FIKIRLER.md): SkillCorner as a labelled testbed for BAR-ATH's alive/dead evidence.

BAR-ATH's dead-ball side is v3's hand-made rules + the aligned broadcast (homography invalid in 76% of GT-D
frames vs ~3.5% of alive frames); so far it could only be checked on the 4 stoppages of the public window.
SkillCorner is broadcast tracking with a per-frame "view shows the pitch" flag that behaves the same way
(valid in 94% of alive frames, 25% of dead ones), and its phases of play give labelled D for 20 matches.
Here v3's aliveness is computed from the degraded SkillCorner tracking (model_v2.aliveness, v3 look-ahead)
plus the view cues (invalid view -w_binvalid, live view +w_blive), and compared with the learned DFL dead
model, blends and true D. Team side = v7's team model, decoder = v7's (B2 soft minimum). AGGREGATE only.

  python -I src/exp_scdead.py [grid]
"""
import itertools
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_blend  # noqa: E402
import exp_v7  # noqa: E402
import model_v2  # noqa: E402
import model_v3  # noqa: E402
import model_v7  # noqa: E402
import skillcorner as scr  # noqa: E402

SRC = "sc@v3"
TAG = "%s_native_v3" % bench._safe(model_v7.VERSION)
EPS = 1e-4


def view_cues():
    out = []
    for m in scr.matches():
        d = scr.parse(m)
        for h in (1, 2):
            out.append((d["h%d_valid" % h], d["h%d_ndet" % h]))
    return out


def rule_alive(blocks, cues, p):
    res = []
    for b, (valid, ndet) in zip(blocks, cues):
        a = model_v2.aliveness(exp_blend.rule_f(b), p)
        a[~valid] -= p["w_binvalid"]
        a[valid & (ndet >= 4)] += p["w_blive"]
        res.append(a)
    return res


def decode_scores(a_list, phl, pen_team=12.0, pen_dead=30.0, min_poss=8, min_dead=50):
    sp = decode2.duration_spec(pen_team, pen_dead, min_poss, min_dead, exp_v7.ramp(175, 48.0))
    out = []
    for a, ph in zip(a_list, phl):
        q = np.clip(ph, EPS, 1 - EPS)
        S = np.stack([a / 2 + np.log(q), a / 2 + np.log(1 - q), -a / 2], 1)
        out.append(sp.decode(S).astype(np.int8))
    return out


def evaluate(preds, gts, blocks):
    r = bench.spell_by_group(blocks, gts, preds)
    sp = np.array([r[k][0] for k in sorted(r)])
    mf = np.array([r[k][1] for k in sorted(r)])
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        tp += np.sum((p < 2) & (g < 2))
        fp += np.sum((p < 2) & (g == 2))
        fn += np.sum((p == 2) & (g < 2))
    return sp, mf, 2 * tp / (2 * tp + fp + fn)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    bl = bench.blocks(SRC)
    gts = bench.labels(SRC, "phase")
    cues = view_cues()
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", TAG + ".txt"))
    phl = [bench.predict_team(mdl, f) for f in bench.features(model_v7.VERSION, SRC)]
    a_lrn = [exp_blend.logit_alive(x) for x in bench.dead_ext("base", "v3", SRC)]
    p = dict(model_v3.DEFAULT)
    a_rule = rule_alive(bl, cues, p)
    a_true = [np.where(g == 2, -12.0, 12.0) for g in gts]
    variants = [("learned DFL dead model", a_lrn), ("v3 rules + view cues", a_rule)]
    for w in (0.25, 0.5, 1.0):
        variants.append(("rules + %.2f learned" % w, [x + w * y for x, y in zip(a_rule, a_lrn)]))
    variants.append(("true D (oracle)", a_true))
    ref = None
    for name, al in variants:
        sp, mf, adf1 = evaluate(decode_scores(al, phl), gts, bl)
        if ref is None:
            ref = sp
        print("%-26s spellF1 %.4f (%+.4f vs learned, wins %d/20) macroF1 %.4f alive/dead F1 %.4f" % (
            name, sp.mean(), sp.mean() - ref.mean(), int((sp > ref).sum()), mf.mean(), adf1), flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        base_sp = evaluate(decode_scores(a_rule, phl), gts, bl)[0]
        rows = []
        for wb, sh, pdd, md in itertools.product((1.0, 1.5, 2.5, 4.0), (25, 37, 50), (20.0, 30.0, 45.0), (25, 50)):
            q = dict(p, w_binvalid=wb, spd_shift=sh)
            sp, mf, adf1 = evaluate(decode_scores(rule_alive(bl, cues, q), phl, pen_dead=pdd, min_dead=md), gts, bl)
            rows.append(((wb, sh, pdd, md), sp, mf, adf1))
        rows.sort(key=lambda r: -r[1].mean())
        print("\nrules grid (w_binvalid, spd_shift, pen_dead, min_dead), vs v3 defaults (1.5, 37, 30, 50) %.4f:" %
              base_sp.mean())
        for k, sp, mf, adf1 in rows[:10]:
            d = sp - base_sp
            print("  %-22s spellF1 %.4f (%+.4f, wins %d/20) macroF1 %.4f adF1 %.4f" % (
                k, sp.mean(), d.mean(), int((d > 0).sum()), mf.mean(), adf1), flush=True)


if __name__ == "__main__":
    main()
