"""Learned H/A/D emissions trained on degraded open DFL data, validated leave-one-match-out with
the official spell metric, then applied to BAR-ATH.

  .venv/Scripts/python.exe src/train_dfl.py build      # parse + degrade + features (cached)
  .venv/Scripts/python.exe src/train_dfl.py cv         # LOMO CV + decoder tuning on DFL
  .venv/Scripts/python.exe src/train_dfl.py apply      # train on all DFL, label BAR-ATH
"""
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import atd  # noqa: E402
import dfl  # noqa: E402
import evaluate as ev  # noqa: E402
import mlfeat  # noqa: E402
import sim  # noqa: E402
from decode import viterbi_min_dur  # noqa: E402

pm = ev.pm
CACHE = os.path.join(ROOT, "cache", "dfl")
LGB_DEAD = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                verbose=-1, seed=0, num_threads=8, deterministic=True, force_row_wise=True)
LGB_TEAM = dict(LGB_DEAD)
ROUNDS = 400
STRIDE = 2   # train on every 2nd frame (neighbours are near-duplicates)


def block_features(P, B, team, gk, pitch):
    F, Th, Ta = mlfeat.compute(P, B, team, gk, pitch)
    fk = sorted(F)
    tk = sorted(Th)
    XF = np.stack([F[k] for k in fk], 1).astype(np.float32)
    XH = np.stack([Th[k] for k in tk], 1).astype(np.float32)
    XA = np.stack([Ta[k] for k in tk], 1).astype(np.float32)
    return XF, XH, XA, fk, tk


def build():
    os.makedirs(CACHE, exist_ok=True)
    for mi, m in enumerate(dfl.MATCHES):
        out = os.path.join(CACHE, "{}_feat.pkl".format(m))
        if os.path.isfile(out) or not os.path.isfile(os.path.join(CACHE, "{}.npz".format(m))):
            continue
        t = time.time()
        d = dfl.parse(m)
        pitch = tuple(d["pitch"])
        blocks = []
        for h in (1, 2):
            tag = "h%d" % h
            P, B, lab = d[tag + "_players"], d[tag + "_ball"], d[tag + "_labels"]
            Pn, Bn = sim.degrade(P, B, lab, pitch, seed=1000 * mi + h)
            XF, XH, XA, fk, tk = block_features(Pn, Bn, d[tag + "_team"], d[tag + "_gk"], pitch)
            blocks.append({"XF": XF, "XH": XH, "XA": XA, "lab": lab,
                           "seen": np.isfinite(Bn[:, 0])})
        with open(out, "wb") as fh:
            pickle.dump({"blocks": blocks, "fk": fk, "tk": tk}, fh, protocol=4)
        print("built", m, "in %.0fs" % (time.time() - t),
              "ball seen %.1f%%" % (100 * np.mean(np.concatenate([b["seen"] for b in blocks]))))


def load_all():
    data = {}
    for m in dfl.MATCHES:
        with open(os.path.join(CACHE, "{}_feat.pkl".format(m)), "rb") as fh:
            data[m] = pickle.load(fh)
    return data


def stack(data, matches, stride=STRIDE):
    XF, XH, XA, y = [], [], [], []
    for m in matches:
        for b in data[m]["blocks"]:
            sl = slice(0, None, stride)
            XF.append(b["XF"][sl]); XH.append(b["XH"][sl]); XA.append(b["XA"][sl]); y.append(b["lab"][sl])
    return np.concatenate(XF), np.concatenate(XH), np.concatenate(XA), np.concatenate(y)


def fit(XF, XH, XA, y):
    graded = y < 3
    dead = lgb.train(LGB_DEAD, lgb.Dataset(XF[graded], (y[graded] == 2).astype(int)), ROUNDS)
    alive = (y == 0) | (y == 1)
    Xt = np.concatenate([np.hstack([XF[alive], XH[alive]]), np.hstack([XF[alive], XA[alive]])])
    yt = np.concatenate([(y[alive] == 0), (y[alive] == 1)]).astype(int)
    team = lgb.train(LGB_TEAM, lgb.Dataset(Xt, yt), ROUNDS)
    return dead, team


def predict_proba(models, XF, XH, XA):
    dead, team = models
    pd_ = dead.predict(XF)
    ph1 = team.predict(np.hstack([XF, XH]))
    ph2 = 1 - team.predict(np.hstack([XF, XA]))
    return pd_, 0.5 * (ph1 + ph2)


def emissions(pdead, phome, temp_d=1.0, temp_t=1.0, eps=1e-4):
    pdead = np.clip(pdead, eps, 1 - eps)
    phome = np.clip(phome, eps, 1 - eps)
    la = np.log(1 - pdead) * temp_d
    ld = np.log(pdead) * temp_d
    lh = np.log(phome) * temp_t
    lA = np.log(1 - phome) * temp_t
    return np.stack([la + lh, la + lA, ld], 1)


def decode_block(S, pen_team, pen_dead, min_poss, min_dead):
    pen = np.array([[0, pen_team, pen_dead], [pen_team, 0, pen_dead], [pen_dead, pen_dead, 0]], float)
    return viterbi_min_dur(S, pen, np.array([min_poss, min_poss, min_dead]))


def spell_score(gts, preds):
    """Official metric on concatenated halves separated by X."""
    sep = np.full(10, 3, np.int8)
    g = np.concatenate([np.concatenate([a, sep]) for a in gts]).astype(np.int8)
    p = np.concatenate([np.concatenate([a, sep]) for a in preds]).astype(np.int8)
    r = pm.evaluate(g, p)
    return r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"], r["n_pred_spells"]


def cv():
    data = load_all()
    oof = {}
    t = time.time()
    for m in dfl.MATCHES:
        train = [x for x in dfl.MATCHES if x != m]
        models = fit(*stack(data, train))
        oof[m] = [predict_proba(models, b["XF"], b["XH"], b["XA"]) for b in data[m]["blocks"]]
        print("fold", m, "%.0fs" % (time.time() - t))
    with open(os.path.join(CACHE, "oof.pkl"), "wb") as fh:
        pickle.dump(oof, fh, protocol=4)
    # frame-level quality
    for m in dfl.MATCHES:
        accs = []
        for b, (pd_, ph) in zip(data[m]["blocks"], oof[m]):
            y = b["lab"]
            g = y < 3
            pred = np.where(pd_ > 0.5, 2, np.where(ph > 0.5, 0, 1))
            accs.append((np.mean(pred[g] == y[g]), np.mean((pd_[g] > 0.5) == (y[g] == 2)),
                         np.mean((ph > 0.5)[(y == 0) | (y == 1)] == (y[(y == 0) | (y == 1)] == 0))))
        a = np.mean(accs, 0)
        print("  %s frame acc %.3f  alive/dead acc %.3f  team acc (alive) %.3f" % (m, *a))
    tune(data, oof)


def tune(data, oof):
    grid = list(itertools.product([1.0], [0.5, 1.0], [4, 8, 16, 32], [8, 16, 32, 64], [5], [25]))
    best = None
    for temp_d, temp_t, pt, pdd, mp, md in grid:
        res = []
        for m in dfl.MATCHES:
            gts, preds = [], []
            for b, (pd_, ph) in zip(data[m]["blocks"], oof[m]):
                S = emissions(pd_, ph, temp_d, temp_t)
                preds.append(decode_block(S, pt, pdd, mp, md))
                gts.append(b["lab"])
            res.append(spell_score(gts, preds))
        res = np.array(res)
        line = "temp_d %.1f temp_t %.1f pen_team %3d pen_dead %3d min %d/%d | spellF1 %.4f (min %.3f) macroF1 %.4f  n_gt %d n_pred %d" % (
            temp_d, temp_t, pt, pdd, mp, md, res[:, 0].mean(), res[:, 0].min(), res[:, 1].mean(), res[:, 2].sum(), res[:, 3].sum())
        print(line)
        if best is None or res[:, 0].mean() > best[0]:
            best = (res[:, 0].mean(), dict(temp_d=temp_d, temp_t=temp_t, pen_team=pt, pen_dead=pdd,
                                           min_poss=mp, min_dead=md))
    print("BEST", best)
    with open(os.path.join(CACHE, "best_decoder.json"), "w") as fh:
        json.dump(best[1], fh)


def block_raw(b, fk, tk):
    """Per-frame raw cues for snapping, recovered from the stored feature columns."""
    XF, XH, XA = b["XF"], b["XH"], b["XA"]
    col = {k: i for i, k in enumerate(fk)}
    tcol = {k: i for i, k in enumerate(tk)}
    return {"ball_out": XF[:, col["ball_out"]], "bspeed": XF[:, col["bspd_c6"]],
            "seen": XF[:, col["ball_seen"]] > 0.5, "dh": np.nan_to_num(XH[:, tcol["d_own"]], nan=np.inf),
            "da": np.nan_to_num(XA[:, tcol["d_own"]], nan=np.inf)}


def eval_snap(data, oof, p, variants=None):
    import snap
    variants = variants or [dict(), dict(do_team=False), dict(do_dead=False, do_restart=False),
                            dict(win=25), dict(win=75)]
    for v in [None] + variants:
        res = []
        for m in dfl.MATCHES:
            gts, preds = [], []
            fk, tk = data[m]["fk"], data[m]["tk"]
            for b, (pd_, ph) in zip(data[m]["blocks"], oof[m]):
                lab = decode_block(emissions(pd_, ph, p["temp_d"], p["temp_t"]), p["pen_team"],
                                   p["pen_dead"], p["min_poss"], p["min_dead"]).astype(np.int8)
                if v is not None:
                    r = block_raw(b, fk, tk)
                    lab = snap.snap(lab, r["ball_out"], r["bspeed"], r["seen"], r["dh"], r["da"], **v)
                preds.append(lab)
                gts.append(b["lab"])
            res.append(spell_score(gts, preds))
        res = np.array(res)
        print("snap %-45s spellF1 %.4f  macroF1 %.4f  n_pred %d" % (
            "none" if v is None else json.dumps(v), res[:, 0].mean(), res[:, 1].mean(), res[:, 3].sum()))


def barath_blocks():
    """ML features for both graded halves of BAR-ATH (tactical players, merged ball)."""
    d = atd.load("tactical")
    with open(os.path.join(ROOT, "cache", "features_merged.pkl"), "rb") as fh:
        f = pickle.load(fh)
    gk = np.zeros(d["players"].shape[1], bool)
    gk[f["gk_slots"]] = True
    blocks = []
    for _, a, b in atd.HALVES:
        P = d["players"][a:b + 1]
        Bm = f["bxy"][a:b + 1] / np.array(atd.PITCH, np.float32)
        XF, XH, XA, _, _ = block_features(P, Bm, d["team"], gk, atd.PITCH)
        blocks.append({"a": a, "b": b, "XF": XF, "XH": XH, "XA": XA})
    return f, blocks


def ensure_barath_probs(refresh=False):
    """Train (or load) the all-DFL models and write BAR-ATH per-frame probabilities."""
    out = os.path.join(CACHE, "barath_probs.pkl")
    if os.path.isfile(out) and not refresh:
        return out
    build()
    model_path = os.path.join(CACHE, "models_all.pkl")
    if os.path.isfile(model_path) and not refresh:
        models = pickle.load(open(model_path, "rb"))
    else:
        models = fit(*stack(load_all(), dfl.MATCHES))
        pickle.dump(models, open(model_path, "wb"), protocol=4)
    _, blocks = barath_blocks()
    probs = {blk["a"]: predict_proba(models, blk["XF"], blk["XH"], blk["XA"]) for blk in blocks}
    with open(out, "wb") as fh:
        pickle.dump(probs, fh, protocol=4)
    return out


def apply(params=None, out_name="dfl_v1.csv", w_binvalid=1.5, w_blive=0.4):
    p = json.load(open(os.path.join(CACHE, "best_decoder.json")))
    if params:
        p.update(params)
    model_path = os.path.join(CACHE, "models_all.pkl")
    if os.path.isfile(model_path):
        models = pickle.load(open(model_path, "rb"))
    else:
        data = load_all()
        models = fit(*stack(data, dfl.MATCHES))
        pickle.dump(models, open(model_path, "wb"), protocol=4)
    f, blocks = barath_blocks()
    labels = np.full(f["n"], 3, np.int8)
    probs = {}
    for blk in blocks:
        pd_, ph = predict_proba(models, blk["XF"], blk["XH"], blk["XA"])
        a, b = blk["a"], blk["b"]
        S = emissions(pd_, ph, p["temp_d"], p["temp_t"])
        have = f["b_have"][a:b + 1]
        ab = np.zeros(b - a + 1)
        ab[have & ~f["b_valid"][a:b + 1]] -= w_binvalid
        ab[have & f["b_live"][a:b + 1]] += w_blive
        S[:, 0] += ab / 2
        S[:, 1] += ab / 2
        S[:, 2] -= ab / 2
        if a == atd.HALVES[1][1]:
            S[:atd.KICKOFF_2ND - a, 2] += 1e6
        labels[a:b + 1] = decode_block(S, p["pen_team"], p["pen_dead"], p["min_poss"], p["min_dead"])
        probs[a] = (pd_, ph)
    path = ev.write(labels, os.path.join(ROOT, "submissions", out_name))
    import proxy
    print("wrote", path, "decoder", p)
    print("public window :", ev.fmt(ev.score(labels)))
    print("full match    :", ev.fmt(ev.diagnostics(labels)))
    print("proxies       :", ev.fmt(proxy.proxies(labels, f)))
    with open(os.path.join(CACHE, "barath_probs.pkl"), "wb") as fh:
        pickle.dump(probs, fh, protocol=4)
    return labels


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "cv":
        cv()
    elif cmd == "apply":
        apply(json.loads(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "snap":
        data = load_all()
        with open(os.path.join(CACHE, "oof.pkl"), "rb") as fh:
            oof = pickle.load(fh)
        eval_snap(data, oof, json.load(open(os.path.join(CACHE, "best_decoder.json"))))
    elif cmd == "tune":
        data = load_all()
        with open(os.path.join(CACHE, "oof.pkl"), "rb") as fh:
            tune(data, pickle.load(fh))
