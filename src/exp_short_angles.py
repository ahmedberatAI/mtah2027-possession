"""Round 15 diagnostic: what separates a true short possession from a false flip? AGGREGATE output only.

Items on the external sets (held-out v10 seed-0 probabilities, V3 inputs, true labels):
  T  true spells shorter than 4 s, by kind (between two opponent spells / after a restart / before a stoppage /
     between stoppages); "seen" when the +-25 v10 probability favours the spell's team on >= 6 of its frames
  F  false flips: maximal runs >= 12 frames where the +-25 v10 probability favours the other team inside a true
     spell >= 8 s, at least 1 s from its ends
Per item, from the degraded inputs the model sees (team k = the spell's / flip's team): duration; ball missing share;
ball speed and in-flight share (> 12 m/s); share of frames where a team-k player is the nearest; nearest team-k
distance; detected touches (src/touch.py) by k and by the other team in [a - 8, b + 8); distinct team-k players
holding the ball (nearest within 1.5 m for >= 3 frames; a completed-pass proxy) and the longest such hold; contested
share (both teams within 3 m); ball depth towards k's attack; distance to the touchline; time since the last true
restart; team-k and other-team centroid speed towards k's attack, after minus before the item start (transition
reaction); v10 logit margin for k (+-25 and raw); BiGRU margin (Metrica / SkillCorner, member 0).
Reports: kinds and situations of T vs F, feature medians and univariate AUCs (T seen vs F), leave-one-match-out
LightGBM AUC and precision at fixed recall.

  python -I src/exp_short_angles.py
"""
import os
import sys
import warnings

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_smooth  # noqa: E402
import touch  # noqa: E402

FPS = 25.0
SHORT, LONG, RUN, EDGE = 100, 200, 12, 25
CONTEXTS = (("dfl@v3", "trk"), ("metrica@v3", "ctrl"), ("sc@v3", "phase"), ("sc@v3", "pp"))
FEATS = ("dur", "miss", "bspeed", "fly", "near_k", "dmin_k", "touch_k", "touch_o", "two_k", "hold_k", "contested",
         "depth", "edge", "since_restart", "cen_k", "cen_o", "m_s", "m_raw", "m_gru")


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def frame_cues(b):
    L, W = b["pitch"]
    pxy = np.asarray(b["P"], np.float32)[:, :, :2] * np.array([L, W], np.float32)
    bxy = np.asarray(b["B"], np.float32)[:, :2] * np.array([L, W], np.float32)
    team = np.asarray(b["team"])
    gk = np.asarray(b["gk"], bool)
    n = len(bxy)
    miss = ~np.isfinite(bxy[:, 0])
    sp = np.full(n, np.nan)
    if n > 4:
        sp[2:-2] = np.linalg.norm(bxy[4:] - bxy[:-4], axis=1) / (4 / FPS)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    dk = [np.min(np.where(team[None, :] == k, dist, np.inf), 1) for k in (0, 1)]
    j = np.argmin(dist, 1)
    dn = dist[np.arange(n), j]
    near_team = np.where(np.isfinite(dn), team[j], -1)
    holder = np.where(dn <= 1.5, j, -1)
    tt, tk = touch.touches(pxy, bxy, team)
    plus = {}
    cen = {}
    for k in (0, 1):
        m = gk & (team == k)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = np.nanmedian(pxy[:, m, 0]) if m.any() else np.nan
            plus[k] = bool(x < L / 2) if np.isfinite(x) else True
            c = np.nanmean(np.where((team == k) & ~gk, pxy[:, :, 0], np.nan), 1)
        v = np.full(n, np.nan)
        if n > 10:
            v[5:-5] = (c[10:] - c[:-10]) / (10 / FPS)
        cen[k] = v
    return dict(L=L, W=W, bxy=bxy, team=team, miss=miss, sp=sp, dk=dk, near_team=near_team, holder=holder,
                touch_t=tt, touch_k=tk, plus=plus, cen=cen, pxy=pxy, gk=gk)


def longest_same(h):
    best = cur = 0
    prev = -2
    for x in h:
        cur = cur + 1 if (x >= 0 and x == prev) else (1 if x >= 0 else 0)
        prev = x
        best = max(best, cur)
    return best


def item(c, gt, ms, mr, mg, a, b, k):
    o = 1 - k
    seg = slice(a, b)
    pres = ~c["miss"][seg]
    sp = c["sp"][seg]
    sgn = 1.0 if c["plus"][k] else -1.0
    f = dict(dur=(b - a) / FPS, miss=float(np.mean(~pres)))
    f["bspeed"] = float(np.nanmedian(sp)) if np.isfinite(sp).any() else np.nan
    f["fly"] = float(np.mean(sp[np.isfinite(sp)] > 12)) if np.isfinite(sp).any() else np.nan
    nt = c["near_team"][seg]
    f["near_k"] = float(np.mean(nt[pres] == k)) if pres.any() else np.nan
    dmin = c["dk"][k][seg]
    f["dmin_k"] = float(min(np.min(dmin), 30.0)) if np.isfinite(dmin).any() else np.nan
    w = (c["touch_t"] >= a - 8) & (c["touch_t"] < b + 8)
    f["touch_k"] = float(np.sum(c["touch_k"][w] == k))
    f["touch_o"] = float(np.sum(c["touch_k"][w] == o))
    h = np.where(c["team"][np.maximum(c["holder"][seg], 0)] == k, c["holder"][seg], -1)
    held = [p for p in set(h[h >= 0].tolist()) if max((e - s for s, e in runs(h == p)), default=0) >= 3]
    f["two_k"] = float(len(held))
    f["hold_k"] = longest_same(h) / FPS
    both = (c["dk"][0][seg] <= 3) & (c["dk"][1][seg] <= 3)
    f["contested"] = float(np.mean(both[pres])) if pres.any() else np.nan
    bx, by = c["bxy"][seg, 0], c["bxy"][seg, 1]
    if pres.any():
        x = np.nanmedian(bx[pres]) / c["L"]
        f["depth"] = float(x if sgn > 0 else 1 - x)
        f["edge"] = float(np.nanmedian(np.minimum(by[pres], c["W"] - by[pres])))
    else:
        f["depth"] = f["edge"] = np.nan
    r = np.flatnonzero((gt[1:a + 1] < 2) & (gt[:a] == 2))
    f["since_restart"] = min((a - (r[-1] + 1)) / FPS, 60.0) if len(r) else 60.0
    for key, team_ in (("cen_k", k), ("cen_o", o)):
        v = c["cen"][team_] * sgn
        before = v[max(a - 25, 0):a]
        after = v[a:min(b + 25, len(v))]
        f[key] = float(np.nanmean(after) - np.nanmean(before)) if (np.isfinite(before).any() and
                                                                   np.isfinite(after).any()) else np.nan
    s = 1.0 if k == 0 else -1.0
    f["m_s"] = float(np.mean(ms[seg]) * s)
    f["m_raw"] = float(np.mean(mr[seg]) * s)
    f["m_gru"] = float(np.mean(mg[seg]) * s) if mg is not None else np.nan
    return f


def collect(src, conv):
    blocks = bench.blocks(src)
    gts = bench.labels(src, conv)
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))[src]
    smo = exp_smooth.smooth(raw, 25)
    gru = None
    if not src.startswith("dfl"):
        path = os.path.join(os.path.dirname(bench.CACHE), "shared_blend_v1", src.replace("@", "-") + "_members.pkl")
        gru = bench._load(path)["members"][0]
    T, F = [], []
    alive = 0
    for i, (b, gt) in enumerate(zip(blocks, gts)):
        c = frame_cues(b)
        ms, mr = logit(smo[i]), logit(raw[i])
        mg = logit(gru[i]) if gru is not None else None
        fav = np.where(ms > 0, 0, 1)
        group = str(b["group"])
        alive += int((gt < 2).sum())
        sp = eo.spells(gt)
        for j, (a, e, k) in enumerate(sp):
            if e - a < SHORT:
                prev = gt[a - 1] if a > 0 else 3
                nxt = gt[e] if e < len(gt) else 3
                kind = ("between opponents" if prev == 1 - k and nxt == 1 - k else
                        "after restart" if prev == 2 and nxt != 2 else
                        "before stoppage" if prev != 2 and nxt == 2 else
                        "between stoppages" if prev == 2 and nxt == 2 else "other")
                f = item(c, gt, ms, mr, mg, a, e, k)
                f.update(kind=kind, seen=int((fav[a:e] == k).sum() >= 6), group=group)
                T.append(f)
            elif e - a >= LONG:
                for s, t in runs(fav[a + EDGE:e - EDGE] != k):
                    s, t = s + a + EDGE, t + a + EDGE
                    if t - s >= RUN:
                        f = item(c, gt, ms, mr, mg, s, t, 1 - k)
                        f.update(kind="false flip", seen=1, group=group)
                        F.append(f)
    return T, F, alive


def report(src, conv, T, F, alive):
    n_m = alive / FPS / 60
    print("\n=== %s %s: true short spells %d (%.2f / alive min), false flips %d (%.2f / alive min) ===" % (
        src.split("@")[0], conv, len(T), len(T) / n_m, len(F), len(F) / n_m))
    kinds = ("between opponents", "after restart", "before stoppage", "between stoppages", "other")
    print("  true short spells by kind (n, seen by v10 +-25): " + " | ".join(
        "%s %d (%.0f%%)" % (kd, sum(t["kind"] == kd for t in T), 100 * np.mean([t["seen"] for t in T if
                                                                               t["kind"] == kd] or [0]))
        for kd in kinds if any(t["kind"] == kd for t in T)))
    Ts = [t for t in T if t["seen"]]

    def share(items, fn):
        v = [fn(x) for x in items]
        return 100 * np.mean(v) if v else np.nan
    sits = (("ball mostly missing (>50%)", lambda x: x["miss"] > 0.5),
            ("ball in flight >30% of visible", lambda x: np.nan_to_num(x["fly"]) > 0.3),
            ("contested >50% of visible", lambda x: np.nan_to_num(x["contested"]) > 0.5),
            ("team k touch detected", lambda x: x["touch_k"] > 0),
            ("2+ team-k holders", lambda x: x["two_k"] >= 2),
            ("held >= 1 s by one k player", lambda x: x["hold_k"] >= 1.0),
            ("within 5 s of a restart", lambda x: x["since_restart"] <= 5),
            ("within 5 m of a touchline", lambda x: np.nan_to_num(x["edge"], nan=99) <= 5),
            ("final third of team k", lambda x: np.nan_to_num(x["depth"]) > 2 / 3))
    print("  situations, %% of items  [T all | T seen | F]:")
    for name, fn in sits:
        print("    %-32s %5.1f | %5.1f | %5.1f" % (name, share(T, fn), share(Ts, fn), share(F, fn)))
    print("  feature medians [T seen | F] and AUC (T seen vs F):")
    cells = []
    for key in FEATS:
        a_ = np.array([t[key] for t in Ts], float)
        b_ = np.array([x[key] for x in F], float)
        if not np.isfinite(a_).any():
            continue
        cells.append("%s %.2f|%.2f AUC %.2f" % (key, np.nanmedian(a_), np.nanmedian(b_), auc(a_, b_)))
    for i in range(0, len(cells), 4):
        print("    " + " ; ".join(cells[i:i + 4]))
    items = Ts + F
    y = np.array([1] * len(Ts) + [0] * len(F))
    keys = [k for k in FEATS if np.isfinite(np.array([x[k] for x in items], float)).any()]
    X = np.array([[x[k] for k in keys] for x in items], float)
    g = np.array([x["group"] for x in items])
    score = np.full(len(y), np.nan)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0)
    for grp in np.unique(g):
        tr, te = g != grp, g == grp
        if y[tr].min() == y[tr].max():
            continue
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300)
        score[te] = m.predict(X[te])
    ok = np.isfinite(score)
    order = np.argsort(-score[ok])
    yy = y[ok][order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    rec = tp / max(yy.sum(), 1)
    pts = []
    for r in (0.3, 0.5, 0.7, 0.9):
        idx = np.searchsorted(rec, r)
        if idx < len(prec):
            pts.append("recall %.1f -> precision %.2f" % (r, prec[idx]))
    m_only = auc([t["m_s"] for t in Ts], [x["m_s"] for x in F])
    print("  leave-one-match-out LightGBM on all features: AUC %.3f (v10 margin alone %.3f); base rate %.2f; %s" % (
        auc(score[ok][y[ok] == 1], score[ok][y[ok] == 0]), m_only, y.mean(), "; ".join(pts)))
    sys.stdout.flush()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    for src, conv in CONTEXTS:
        T, F, alive = collect(src, conv)
        report(src, conv, T, F, alive)


if __name__ == "__main__":
    main()
