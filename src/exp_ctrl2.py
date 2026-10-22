"""Round 7b: the player-level controller (src/ctrl_model.py) as evidence for v7, two better ways than exp_ctrl.py's
raw blend (which lost everywhere: log(q_H / q_A) of two tiny probabilities is noise whenever nobody controls).

(a) bounded blend: z = logit(p_v7) + alpha * log((q_H + c) / (q_A + c)) -- about 0 when nobody controls;
(b) controller features in the team model: per team the pooled q, the best candidate's p, the bounded log ratio and
    its windowed means, frames since / until the team last / next confidently controlled (q > 0.5) and whether the
    latest / next confident controller is the team's own -- appended to the v7 features; team model retrained as v7
    (DFL native labels, sim V3; DFL leave-one-match-out, all-DFL model elsewhere).
Controller probabilities: SkillCorner out of fold (5 folds by match, as exp_ctrl.py), the all-SkillCorner model on
DFL / Metrica / BAR-ATH (it never saw their labels). Paired against v7 on SkillCorner (player-possession and phase
labels; learned / true D), DFL-trk and Metrica-ctrl (learned / true D), plus frame team accuracy. AGGREGATE
output only.

  python -I src/exp_ctrl2.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import ctrl_model as cm  # noqa: E402
import decode2  # noqa: E402
import exp_ctrl  # noqa: E402
import exp_oracle  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import train_dfl as td  # noqa: E402
from mlfeat import since_last, until_next, win_mean  # noqa: E402

OUT = exp_ctrl.OUT
EPS = 1e-4
TAG = "ctrlfeat_" + exp_oracle.TAG
SETS = (("sc@v3", "pp", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)))
BLEND = [(c, a) for c in (0.05, 0.2) for a in (0.5, 1.0, 2.0)]


def ctrl_p(src):
    """Controller probabilities per block row (out of fold on SkillCorner)."""
    path = os.path.join(OUT, "p_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    rows = exp_ctrl.build_rows(src)
    if src == "sc@v3":
        n = len(rows)
        out = [None] * n
        for k in range(5):
            test = set(range(k, 20, 5))
            tr = [i for i in range(n) if i // 2 not in test]
            mdl = lgb.train(exp_ctrl.PARAMS, lgb.Dataset(np.concatenate([rows[i]["X"] for i in tr]),
                                                         np.concatenate([rows[i]["y"] for i in tr])), exp_ctrl.ROUNDS)
            for i in range(n):
                if i // 2 in test:
                    out[i] = mdl.predict(rows[i]["X"])
    else:
        mdl = lgb.Booster(model_file=os.path.join(OUT, "ctrl_all_sc.txt"))
        out = [mdl.predict(r["X"]) for r in rows]
    bench._dump(out, path)
    return out


def per_frame(p, r, n, step=cm.STEP):
    """(n, 2) pooled q = 1 - prod(1 - p) and (n, 2) best candidate p per team; 0 where no candidate (no ball)."""
    g = (n + step - 1) // step
    pos, tm = r["fr"] // step, r["tm"].astype(int)
    logn = np.zeros((g, 2))
    np.add.at(logn, (pos, tm), np.log(np.clip(1 - p, EPS, 1)))
    mx = np.zeros((g, 2))
    np.maximum.at(mx, (pos, tm), p)
    q = 1 - np.exp(logn)
    t, grid = np.arange(n), np.arange(g) * step
    return (np.stack([np.interp(t, grid, q[:, k]) for k in (0, 1)], 1),
            np.stack([np.interp(t, grid, mx[:, k]) for k in (0, 1)], 1))


def bounded(Q, c):
    return np.log((Q[:, 0] + c) / (Q[:, 1] + c))


def ctrl_feats(Q, M):
    """Team-perspective controller features (home, away), each (n, 14)."""
    conf = Q > 0.5
    out = []
    for k in (0, 1):
        o = 1 - k
        z = np.log((Q[:, k] + 0.1) / (Q[:, o] + 0.1))
        s_own, s_opp = since_last(conf[:, k]), since_last(conf[:, o])
        u_own, u_opp = until_next(conf[:, k]), until_next(conf[:, o])
        cols = [Q[:, k], Q[:, o], M[:, k], M[:, o], z, win_mean(z, -12, 12), win_mean(z, -50, 0), win_mean(z, 0, 50),
                s_own, s_opp, u_own, u_opp, s_own < s_opp, u_own < u_opp]
        out.append(np.stack(cols, 1).astype(np.float32))
    return out


def _feat_path(b):
    return os.path.join(bench.CACHE, "feat", bench._safe(model_v7.VERSION), b["key"].replace("/", "_") + ".pkl")


def augmented(src, i, p, r):
    b = bench.blocks(src)[i]
    f = bench._load(_feat_path(b))
    ch, ca = ctrl_feats(*per_frame(p, r, len(b["P"])))
    return dict(XF=f["XF"], XH=np.hstack([f["XH"], ch]), XA=np.hstack([f["XA"], ca]))


def feat_model_probs(src, pc, rows):
    """(b) team probabilities: DFL leave-one-match-out, else the all-DFL model."""
    path = os.path.join(OUT, "ph_%s_%s.pkl" % (TAG, src.replace("@", "-")))
    if os.path.isfile(path):
        return bench._load(path)
    mpath = os.path.join(OUT, TAG + ".txt")
    bl = bench.blocks(src)
    if src == "dfl@v3":
        fd = [augmented(src, i, pc[i], rows[i]) for i in range(len(bl))]
        labs = bench.labels("dfl", "native")
        out = [None] * len(bl)
        for m in sorted({b["group"] for b in bl}):
            tr = [i for i, b in enumerate(bl) if b["group"] != m]
            mdl = bench.fit_team([fd[i] for i in tr], [labs[i] for i in tr], {"num_threads": bench.THREADS})
            for i, b in enumerate(bl):
                if b["group"] == m:
                    out[i] = bench.predict_team(mdl, fd[i])
            print("  LOMO %s done" % m, flush=True)
        if not os.path.isfile(mpath):
            bench.fit_team(fd, labs, {"num_threads": bench.THREADS}).save_model(mpath)
    else:
        mdl = lgb.Booster(model_file=mpath)
        out = [bench.predict_team(mdl, augmented(src, i, pc[i], rows[i])) for i in range(len(bl))]
    bench._dump(out, path)
    return out


def decode_all(dec, pds, phs):
    sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, 48.0))
    return [sp.decode(td.emissions(pd_, np.clip(ph, EPS, 1 - EPS), 1.0, 1.0)).astype(np.int8)
            for pd_, ph in zip(pds, phs)]


def score(src, conv, preds):
    r = bench.spell_by_group(bench.blocks(src), bench.labels(src, conv), preds)
    return np.array([r[k][0] for k in sorted(r)])


def team_acc(phs, labs):
    ok = np.concatenate([((p > 0.5) == (l == 0))[l < 2] for p, l in zip(phs, labs)])
    return ok.mean()


def coverage(phl, Qs, labs, short=100, run=5):
    """Can the controller find the short possessions v7 misses? Per true spell < `short` frames: whether v7's own
    team probability ever favours the true team inside it, and whether the controller is confident (q > 0.5 for
    >= `run` frames in a row) for the true team inside it. False alarms: such confident runs of the OTHER team
    inside true spells >= `short` frames, per alive minute."""
    n_short = never = cov_all = cov_never = alarms = 0
    alive = 0
    for p, Q, lab in zip(phl, Qs, labs):
        alive += int((lab < 2).sum())
        for a, b, k in exp_oracle.spells(lab):
            k = int(k)
            conf = Q[a:b, k] > 0.5
            hit = bool(np.convolve(conf, np.ones(run), "valid").max(initial=0) >= run) if b - a >= run else False
            if b - a < short:
                n_short += 1
                flips = bool(((p[a:b] > 0.5) == (k == 0)).any())
                never += not flips
                cov_all += hit
                cov_never += hit and not flips
            else:
                c2 = (Q[a:b, 1 - k] > 0.5).astype(np.int8)
                r = np.convolve(c2, np.ones(run, np.int8), "valid") >= run
                alarms += int(np.sum(r[1:] & ~r[:-1]) + (r[0] if len(r) else 0))
    return n_short, never / max(n_short, 1), cov_all / max(n_short, 1), cov_never / max(never, 1), \
        alarms / max(alive / 1500.0, 1e-9)


def sig(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src in ("dfl@v3", "sc@v3", "metrica@v3"):
        ctrl_p(src)
    # DFL first: it trains the all-DFL feature model the other sets use
    feat_model_probs("dfl@v3", ctrl_p("dfl@v3"), exp_ctrl.build_rows("dfl@v3"))
    cache = {}
    for src, conv, dec in SETS:
        rows = exp_ctrl.build_rows(src)
        pc = ctrl_p(src)
        bl = bench.blocks(src)
        if src not in cache:
            phl, pdl = exp_oracle.probs(src)
            Qs = [per_frame(p, r, len(b["P"]))[0] for p, r, b in zip(pc, rows, bl)]
            cache[src] = (phl, pdl, Qs, feat_model_probs(src, pc, rows))
        phl, pdl, Qs, phf = cache[src]
        labs = bench.labels(src, conv)
        print("%-10s %-6s frame team acc (alive): v7 %.4f | ctrl features %.4f | blend c0.2 a1 %.4f" % (
            src, conv, team_acc(phl, labs), team_acc(phf, labs),
            team_acc([sig(logit(p) + bounded(q, 0.2)) for p, q in zip(phl, Qs)], labs)), flush=True)
        ns, nv, ca, cn, fa = coverage(phl, Qs, labs)
        print("%-10s %-6s true spells < 4 s: %d, v7 never favours the true team in %.3f; controller confident for the "
              "true team in %.3f of all short / %.3f of those never-favoured; confident wrong-team runs inside longer "
              "spells %.2f per alive minute" % (src, conv, ns, nv, ca, cn, fa), flush=True)
        variants = [("learned D", pdl)]
        if src != "dfl@v3":
            variants.append(("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs]))
        for dname, pds in variants:
            ref = score(src, conv, decode_all(dec, pds, phl))
            cells = []
            for c, a in BLEND:
                phb = [sig(logit(p) + a * bounded(q, c)) for p, q in zip(phl, Qs)]
                v = score(src, conv, decode_all(dec, pds, phb))
                cells.append("c%.2g a%.2g %+.4f (%d/%d)" % (c, a, v.mean() - ref.mean(), int((v > ref).sum()), len(v)))
            v = score(src, conv, decode_all(dec, pds, phf))
            print("%-10s %-6s %-9s v7 %.4f | FEATURES %+.4f (%d/%d) | %s" % (
                src, conv, dname, ref.mean(), v.mean() - ref.mean(), int((v > ref).sum()), len(v), " | ".join(cells)),
                flush=True)


if __name__ == "__main__":
    main()
