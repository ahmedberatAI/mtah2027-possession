"""Round-2 ideas N1 / E2 (docs/RAPOR.md). AGGREGATE output only.

N1  alive/dead evidence = blend of the hand-made rules (model_v2.aliveness with v3's look-ahead; no
    broadcast cues outside BAR-ATH) and the learned dead-ball model (sim V3), a = w_r * a_rule +
    w_l * logit(1 - p_dead). Sim V3 gives realistic player speeds, so the rule thresholds (m/s) mean the
    same on degraded DFL / Metrica as on BAR-ATH and the blend can be tuned on labelled data.
E2  team probabilities = logit average of the native-label and trk-label team models.

  python -I src/exp_blend.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import features as FT  # noqa: E402
import model_v2  # noqa: E402
import model_v3  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
BLENDS = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0), (0.75, 0.25), (0.25, 0.75)]


def rule_f(b):
    """The fields model_v2.aliveness reads, computed like features.build from a bench block."""
    L, W = b["pitch"]
    pxy = b["P"][:, :, :2] * np.array([L, W], np.float32)
    bxy = b["B"][:, :2] * np.array([L, W], np.float32)
    seen = np.isfinite(bxy[:, 0])
    present = np.isfinite(pxy[:, :, 0])
    f = {"n": len(bxy), "seen": seen, "bxy": bxy}
    f["bvel"] = FT.ball_kinematics(bxy, seen)
    f["bspeed"] = np.linalg.norm(f["bvel"], axis=1)
    out_x = np.maximum(-bxy[:, 0], bxy[:, 0] - L)
    out_y = np.maximum(-bxy[:, 1], bxy[:, 1] - W)
    f["ball_out"] = np.where(seen, np.maximum(out_x, out_y), np.nan)
    f["since_seen"] = FT.gap_context(seen)[0]
    spd = np.linalg.norm(FT.player_velocity(pxy), axis=2)
    spd = np.where(present, spd, np.nan)
    with np.errstate(invalid="ignore"):
        s = np.nanmean(spd, axis=1)
    f["spd_all_s25"] = FT.rolling_mean(s, 25)
    f["spd_all_s100"] = FT.rolling_mean(s, 100)
    return f


def logit_alive(pd_):
    p = np.clip(pd_, EPS, 1 - EPS)
    return np.log((1 - p) / p)


def scores(a, ph):
    ph = np.clip(ph, EPS, 1 - EPS)
    return np.stack([a / 2 + np.log(ph), a / 2 + np.log(1 - ph), -a / 2], 1)


def blend_curve(src, a_rule, a_lrn, phl, conv, dec, w):
    bl = bench.blocks(src)
    gts = bench.labels(src, conv)
    out = {}
    for pt in (4, 8, 12, 20, 32, 48):
        preds = []
        for ar, al, ph in zip(a_rule, a_lrn, phl):
            S = scores(w[0] * ar + w[1] * al, ph)
            preds.append(td.decode_block(S, pt, dec["pen_dead"], dec["min_poss"], dec["min_dead"]).astype(np.int8))
        r = bench.spell_by_group(bl, gts, preds)
        arr = np.array(list(r.values()))
        out[pt] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return out


def ad_acc(a_list, labs):
    acc = []
    for a, lab in zip(a_list, labs):
        g = lab < 3
        acc.append(np.mean((a[g] < 0) == (lab[g] == 2)))
    return float(np.mean(acc))


def ens(p1, p2):
    l1 = np.log(np.clip(p1, EPS, 1 - EPS) / (1 - np.clip(p1, EPS, 1 - EPS)))
    l2 = np.log(np.clip(p2, EPS, 1 - EPS) / (1 - np.clip(p2, EPS, 1 - EPS)))
    return 1 / (1 + np.exp(-(l1 + l2) / 2))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    p = dict(model_v3.DEFAULT)
    sets = {}
    team = {}
    for base in ("dfl", "metrica"):
        src = bench.src_name(base, "v3")
        bl = bench.blocks(src)
        a_rule = [model_v2.aliveness(rule_f(b), p) for b in bl]
        a_lrn = [logit_alive(x) for x in bench.dead_probs(src, "base")]
        sets[base] = (src, a_rule, a_lrn)
        if base == "dfl":
            team[base] = {k: bench._load(os.path.join(bench.CACHE, "oof", "base_%s_v3.pkl" % k))
                          for k in ("native", "trk")}
        else:
            fm = bench.features("base", src)
            team[base] = {k: [bench.predict_team(lgb.Booster(model_file=os.path.join(
                bench.CACHE, "models", "base_%s_v3.txt" % k)), f) for f in fm] for k in ("native", "trk")}
        team[base]["ens"] = [ens(x, y) for x, y in zip(team[base]["native"], team[base]["trk"])]
    print("alive/dead frame accuracy (sign of a vs native D labels):")
    for base, (src, ar, al) in sets.items():
        labs = bench.labels(src, "native")
        print("  %-8s rules %.4f | learned %.4f | 0.5/0.5 %.4f | 1/1 %.4f" % (
            base, ad_acc(ar, labs), ad_acc(al, labs), ad_acc([0.5 * x + 0.5 * y for x, y in zip(ar, al)], labs),
            ad_acc([x + y for x, y in zip(ar, al)], labs)))
    plan = (("dfl", "trk", "dflcv"), ("dfl", "native", "dflcv"), ("metrica", "ctrl", "v5"), ("metrica", "events", "v5"))
    for tk in ("native", "trk", "ens"):
        print("\nteam model %s" % tk)
        for base, conv, dec in plan:
            src, ar, al = sets[base]
            cells = []
            for w in BLENDS:
                c = blend_curve(src, ar, al, team[base][tk], conv, bench.DECODERS[dec], w)
                b = max(c, key=lambda k: c[k][0])
                cells.append("r%.2g/l%.2g %.4f@%d" % (w[0], w[1], c[b][0], b))
            print("  %-7s vs %-6s (%s): %s" % (base, conv, dec, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
