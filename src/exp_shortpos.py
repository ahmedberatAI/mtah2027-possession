"""Round 6, idea 3 (docs/FIKIRLER.md): verify short possessions at the possession level.

Backbone = v7's decode (B2 soft minimum, pen 12). A low-penalty decode (standard decoder, pen 2, min 5 frames)
proposes candidates: its runs of the other team inside a backbone spell (insertions). Backbone spells shorter
than 8 s are candidates for removal. A LightGBM per candidate type answers "is this a real possession?" (a
same-team true spell with IoU >= 0.4), trained on degraded DFL (trk labels) leave one match out and, trained on
all DFL, applied to Metrica and SkillCorner. Insert above / remove below a threshold. AGGREGATE output only.

  python -I src/exp_shortpos.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402

EPS = 1e-4
REAL_IOU = 0.4
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
              force_row_wise=True, num_threads=bench.THREADS)
ROUNDS = 250
TEAM_COLS = ("d_own", "d_diff_c12", "ctrl_w-25_0", "ctrl_w0_25", "ctl_own_dwell_w-25_0", "ctl_own_dwell_w0_25",
             "ctl_own_comove_w-6_6", "ctl_own_closing_w-6_6")


def runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def decodes(S, dec):
    bb = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0)).decode(S).astype(np.int8)
    fine = decode2.std_spec(2.0, dec[0], 5, dec[2]).decode(S).astype(np.int8)
    return bb, fine


def best_iou(a, b, k, gt_spells):
    best = 0.0
    for c, d, k2 in gt_spells:
        if k2 != k or d <= a or c >= b:
            continue
        best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
    return best


def cand_feats(a, b, k, ph, pd_, f, tcol, seen, bb, pc):
    n = len(ph)
    pk = ph if k == 0 else 1 - ph
    own, opp = (f["XH"], f["XA"]) if k == 0 else (f["XA"], f["XH"])
    lo, hi = max(0, a - 50), min(n, b + 50)
    nxt_d = np.flatnonzero(bb[b:min(n, b + 250)] == 2)
    prv_d = np.flatnonzero(bb[max(0, a - 250):a] == 2)
    x = [b - a, pk[a:b].mean(), pk[a:b].max(), pk[a:b].min(), pk[lo:a].mean() if a > lo else np.nan,
         pk[b:hi].mean() if hi > b else np.nan, np.mean(np.abs(np.log(np.clip(pk[a:b], EPS, 1) / np.clip(1 - pk[a:b], EPS, 1)))),
         pd_[a:b].mean(), seen[a:b].mean(), nxt_d[0] if len(nxt_d) else 250, 250 - prv_d[-1] if len(prv_d) else 250]
    x += [pc[max(0, a - 12):a + 13].max(), pc[max(0, b - 12):b + 13].max()] if pc is not None else [np.nan, np.nan]
    for c in tcol:
        x.append(np.nanmean(own[a:b, c] - opp[a:b, c]) if c is not None else np.nan)
    return x


def candidates(bb, fine, ph, pd_, f, tcol, seen, pc):
    ins, rem = [], []
    for a, b in runs((fine < 2) & (bb < 2) & (fine != bb)):
        ins.append((a, b, int(fine[a]), cand_feats(a, b, int(fine[a]), ph, pd_, f, tcol, seen, bb, pc)))
    for a, b, k in exp_oracle.spells(bb):
        if b - a < 200:
            prev = bb[a - 1] if a > 0 else 3
            nxt = bb[b] if b < len(bb) else 3
            x = cand_feats(a, b, int(k), ph, pd_, f, tcol, seen, bb, pc) + [float(prev == nxt), float(prev == 2),
                                                                            float(nxt == 2)]
            rem.append((a, b, int(k), x, prev, nxt))
    return ins, rem


def apply(bb, ins, rem, p_ins, p_rem, t_ins, t_rem):
    out = bb.copy()
    for (a, b, k, _, prev, nxt), p in zip(rem, p_rem):
        if p >= t_rem:
            continue
        if prev == nxt and prev < 2:
            out[a:b] = prev
        elif prev < 2 and nxt == 2:
            out[a:b] = prev
        elif prev == 2 and nxt < 2:
            out[a:b] = nxt
    for (a, b, k, _), p in zip(ins, p_ins):
        if p >= t_ins:
            out[a:b] = k
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    V = model_v7.VERSION
    pcs_path = os.path.join(bench.CACHE, "change", "pc.pkl")
    pcs = bench._load(pcs_path) if os.path.isfile(pcs_path) else {}
    sets = {}
    for src, conv, dec in exp_oracle.SETS:
        phl, pdl = exp_oracle.probs(src)
        feats = bench.features(V, src)
        tk = feats[0]["tk"]
        tcol = [tk.index(c) if c in tk else None for c in TEAM_COLS]
        gts = bench.labels(src, conv)
        seen = [np.isfinite(b["B"][:, 0]) for b in bench.blocks(src)]
        variants = {"learned D": pdl}
        if src != "dfl@v3":
            variants["true D"] = [np.where(g == 2, 1 - EPS, EPS) for g in gts]
        for dname, pds in variants.items():
            blocks = []
            for i, (ph, pd_, f, g, sn) in enumerate(zip(phl, pds, feats, gts, seen)):
                S = td.emissions(pd_, ph, 1.0, 1.0)
                bb, fine = decodes(S, dec)
                pc = pcs.get(src, [None] * len(phl))[i]
                ins, rem = candidates(bb, fine, ph, pd_, f, tcol, sn, pc)
                gsp = exp_oracle.spells(g)
                yi = [best_iou(a, b, k, gsp) >= REAL_IOU for a, b, k, _ in ins]
                yr = [best_iou(a, b, k, gsp) >= REAL_IOU for a, b, k, _, _, _ in rem]
                blocks.append(dict(bb=bb, ins=ins, rem=rem, yi=np.array(yi, bool), yr=np.array(yr, bool), g=g))
            sets[(src, dname)] = blocks
        print("candidates built", src, flush=True)
    dfl_bl = bench.blocks("dfl@v3")
    groups = [b["group"] for b in dfl_bl]
    train = sets[("dfl@v3", "learned D")]

    def xy(blks, kind):
        X = [c[3] for b in blks for c in b[kind]]
        y = np.concatenate([b["y" + kind[0]] for b in blks]) if blks else np.zeros(0)
        return np.array(X, np.float32), y.astype(int)

    probs = {}
    for kind in ("ins", "rem"):
        oof = [None] * len(train)
        for m in sorted(set(groups)):
            tr = [b for b, gname in zip(train, groups) if gname != m]
            X, y = xy(tr, kind)
            mdl = lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS)
            for i, gname in enumerate(groups):
                if gname == m:
                    Xi, _ = xy([train[i]], kind)
                    oof[i] = mdl.predict(Xi) if len(Xi) else np.zeros(0)
        probs[(kind, "dfl@v3", "learned D")] = oof
        X, y = xy(train, kind)
        full = lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS)
        print("%s candidates on DFL: %d, real share %.3f" % (kind, len(y), y.mean()), flush=True)
        for key, blks in sets.items():
            if key[0] == "dfl@v3":
                continue
            probs[(kind, key[0], key[1])] = [full.predict(xy([b], kind)[0]) if len(b[kind]) else np.zeros(0)
                                            for b in blks]
    from sklearn.metrics import roc_auc_score
    for key, blks in sets.items():
        src, dname = key
        conv = [c for s, c, _ in exp_oracle.SETS if s == src][0]
        bl = bench.blocks(src)
        gts = [b["g"] for b in blks]
        pi, pr = probs[("ins", src, dname)], probs[("rem", src, dname)]
        yi = np.concatenate([b["yi"] for b in blks])
        yr = np.concatenate([b["yr"] for b in blks])
        auc_i = roc_auc_score(yi, np.concatenate(pi)) if 0 < yi.mean() < 1 else np.nan
        auc_r = roc_auc_score(yr, np.concatenate(pr)) if 0 < yr.mean() < 1 else np.nan
        r = bench.spell_by_group(bl, gts, [b["bb"] for b in blks])
        ref = np.array([r[k][0] for k in sorted(r)])
        cells = []
        for t_ins, t_rem in ((1.1, -0.1), (0.5, -0.1), (0.6, -0.1), (0.7, -0.1), (1.1, 0.5), (1.1, 0.3),
                             (0.6, 0.3), (0.5, 0.5), (0.7, 0.3)):
            preds = [apply(b["bb"], b["ins"], b["rem"], a, c, t_ins, t_rem) for b, a, c in zip(blks, pi, pr)]
            r = bench.spell_by_group(bl, gts, preds)
            v = np.array([r[k][0] for k in sorted(r)])
            if t_ins > 1 and t_rem < 0:
                continue
            cells.append("ins>%.1f rem<%.1f %.4f (%+.4f %d/%d)" % (min(t_ins, 9), max(t_rem, 0), v.mean(), v.mean() - ref.mean(),
                                                                    int((v > ref).sum()), len(v)))
        print("\n%-11s %-9s v7 %.4f | candidates ins %d (real %.3f, AUC %.3f) rem %d (real %.3f, AUC %.3f)" % (
            src, dname, ref.mean(), len(yi), yi.mean(), auc_i, len(yr), yr.mean(), auc_r))
        for c in cells:
            print("   ", c, flush=True)


if __name__ == "__main__":
    main()
