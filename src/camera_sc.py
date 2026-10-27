"""Round 11, idea E6, external check (docs/RAPOR.md): the broadcast-camera framing cue on the 20 SkillCorner matches,
whose tracking carries the broadcast image corners projected on the pitch for every frame. The BAR-ATH recipe
(src/camera.py) is replicated without labels -- the camera model learns from v10's own confident frames, cross-fitted
by half inside each match -- and only the evaluation uses SkillCorner's labels (phases / player possessions),
learned and true dead ball, v10's three LightGBM seeds. AGGREGATE output only.

    python -I src/camera_sc.py truth          # cue vs the TRUE team: AUC, timing around true changes
    python -I src/camera_sc.py eval           # fused decode vs v10, per context and seed
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import camera  # noqa: E402
import exp_contest as ec  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import skillcorner as scr  # noqa: E402
from exp_framing import auc  # noqa: E402

SRC = "sc@v3"
FPS = 25
OUT = os.path.join(bench.CACHE, "camera_sc")
GRID = [(0.5, 0, 4.0), (1.0, 0, 4.0), (1.0, 12, 4.0), (1.0, 25, 4.0), (1.0, 40, 4.0), (1.0, 0, 2.0), (1.0, 0, 8.0),
        (1.5, 0, 4.0), (0.7, 12, 4.0)]


def _homographies(corners):
    """corners (n, 8): TL, TR, BL, BR pitch points -> image(0..1)->pitch homographies (n, 3, 3), NaN if missing."""
    img = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    n = len(corners)
    G = np.full((n, 3, 3), np.nan)
    ok = np.isfinite(corners).all(1)
    pts = corners[ok].reshape(-1, 4, 2)
    A = np.zeros((len(pts), 8, 8))
    b = np.zeros((len(pts), 8))
    for i, (u, v) in enumerate(img):
        x, y = pts[:, i, 0], pts[:, i, 1]
        A[:, 2 * i] = np.stack([np.full_like(x, u), np.full_like(x, v), np.ones_like(x), 0 * x, 0 * x, 0 * x,
                                -u * x, -v * x], 1)
        A[:, 2 * i + 1] = np.stack([0 * x, 0 * x, 0 * x, np.full_like(x, u), np.full_like(x, v), np.ones_like(x),
                                    -u * y, -v * y], 1)
        b[:, 2 * i], b[:, 2 * i + 1] = x, y
    with np.errstate(all="ignore"):
        h = np.linalg.solve(A, b[..., None])[..., 0]
    G[ok] = np.concatenate([h, np.ones((len(h), 1))], 1).reshape(-1, 3, 3)
    return G


def _apply(G, u, v):
    q = G @ np.array([u, v, 1.0])
    return q[:, :2] / q[:, 2:3]


def sc_camera(m):
    """Per half on scr.parse()'s 25 fps grid: aim point (normalised pitch), zoom (image widths per metre at the
    centre), valid."""
    path = os.path.join(scr.CACHE, "%s_cam.npz" % m)
    if os.path.isfile(path):
        return dict(np.load(path))
    d = os.path.join(scr.DATA, m)
    meta = json.load(open(os.path.join(d, "%s_match.json" % m), encoding="utf-8"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    rows = {1: [], 2: []}
    with open(os.path.join(d, "%s_tracking_extrapolated.jsonl" % m), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["period"] in rows:
                c = r["image_corners_projection"]
                vals = [c.get(k) for k in ("x_top_left", "y_top_left", "x_top_right", "y_top_right",
                                           "x_bottom_left", "y_bottom_left", "x_bottom_right", "y_bottom_right")]
                rows[r["period"]].append((r["frame"], [np.nan if v is None else float(v) for v in vals]))
    out = {}
    for per in (1, 2):
        f10 = np.array([r[0] for r in rows[per]], np.int64)
        cor = np.array([r[1] for r in rows[per]], float)
        # SkillCorner clips the far ("top") corners to a line 5 m beyond the far touchline in ~90% of frames, so they
        # are points on the image's side edges, not image corners, and a 4-point homography is wrong. The aim is taken
        # as the centroid of the four points instead (the image's vertical centre line runs from the bottom-edge
        # midpoint to the clipped far-edge midpoint), the zoom as 1 / the bottom (near) edge width in metres.
        pts = cor.reshape(-1, 4, 2)                      # TL, TR, BL, BR
        zoom = 1.0 / np.linalg.norm(pts[:, 3] - pts[:, 2], axis=1)
        # the image's vertical centre line on the pitch: every image column u maps to a pitch line through the
        # camera's foot point N = intersection of the left-edge line (BL, TL) and the right-edge line (BR, TR); the
        # centre column passes through N and the bottom-edge midpoint. Stored as two points of that line.
        n_pt = _intersect(pts[:, 2], pts[:, 0], pts[:, 3], pts[:, 1])
        c_pt = 0.5 * (pts[:, 2] + pts[:, 3])
        bad = ~np.isfinite(n_pt[:, 0])
        n_pt[bad] = c_pt[bad] - np.array([0.0, 1.0])       # parallel edges: vertical centre line
        aim = c_pt
        line_a = np.stack([(n_pt[:, 0] + L / 2) / L, (n_pt[:, 1] + W / 2) / W], 1)
        aim_n = np.stack([(aim[:, 0] + L / 2) / L, (aim[:, 1] + W / 2) / W], 1)
        t10 = (f10 - f10[0]) / scr.FPS_IN
        n25 = int(np.floor(t10[-1] * FPS)) + 1
        t25 = np.arange(n25) / FPS
        a25 = scr._interp(t25, t10, aim_n.astype(np.float32))
        n25p = scr._interp(t25, t10, line_a.astype(np.float32))
        z25 = scr._interp(t25, t10, zoom[:, None].astype(np.float32))[:, 0]
        out["h%d_aim" % per] = a25                     # bottom-edge midpoint (on the centre line)
        out["h%d_foot" % per] = n25p                   # second point of the centre line (camera foot)
        out["h%d_zoom" % per] = z25
    np.savez(path, **out)
    return out


def _intersect(p1, p2, p3, p4):
    """Intersection of line p1-p2 with line p3-p4 (rows), NaN where (nearly) parallel."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    with np.errstate(all="ignore"):
        t = ((p3[:, 0] - p1[:, 0]) * d2[:, 1] - (p3[:, 1] - p1[:, 1]) * d2[:, 0]) / den
    out = p1 + t[:, None] * d1
    out[np.abs(den) < 1e-6] = np.nan
    return out


def line_x_at(a, b, y):
    """x of the line through points a, b (rows, normalised) at height y (normalised)."""
    with np.errstate(all="ignore"):
        s = (y - a[:, 1]) / (b[:, 1] - a[:, 1])
        x = a[:, 0] + s * (b[:, 0] - a[:, 0])
    x[~np.isfinite(x)] = a[~np.isfinite(x), 0]
    return x


def block_features():
    """Camera features per bench block (same layout as camera.FEATS), from the raw SkillCorner ball."""
    path = os.path.join(OUT, "features.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    feats = []
    for b in bench.blocks(SRC):
        m, h = b["group"], int(b["key"].split("/h")[-1])
        cam = sc_camera(m)
        d = scr.parse(m)
        L, W = b["pitch"]
        aim, foot, zoom = cam["h%d_aim" % h], cam["h%d_foot" % h], cam["h%d_zoom" % h]
        n = len(b["lab"])
        aim, foot, zoom = aim[:n], foot[:n], zoom[:n]
        ball = d["h%d_ball" % h][:n, :2].astype(float)
        # where the camera's centre line crosses the ball's depth: the horizontal aim at the ball
        aim = np.stack([line_x_at(aim.astype(float), foot.astype(float), ball[:, 1]), ball[:, 1]], 1)
        P = d["h%d_players" % h][:n, :, 0]
        team, gk = d["h%d_team" % h], d["h%d_gk" % h]
        s = 1.0 if np.nanmean(P[:, (team == 0) & gk]) < 0.5 else -1.0
        ok = np.isfinite(aim[:, 0]) & np.isfinite(ball[:, 0])
        valid = np.isfinite(aim[:, 0])
        since = np.zeros(n)
        run = 0
        for i in range(n):
            run = run + 1 if valid[i] else 0
            since[i] = run
        F = {k: np.full(n, np.nan, np.float32) for k in camera.FEATS}
        F["dx"][ok] = ((aim[:, 0] - ball[:, 0]) * L * s)[ok]
        F["dy"][ok] = ((aim[:, 1] - ball[:, 1]) * W)[ok]
        F["bxn"][ok] = (ball[:, 0] if s > 0 else 1 - ball[:, 0])[ok]
        F["by"][ok] = ball[ok, 1]
        F["zoom"][ok] = zoom[ok]
        F["pan"][ok] = (camera._win_diff(aim[:, 0] * L, 12) * s)[ok]
        F["bvx"][ok] = (camera._win_diff(ball[:, 0] * L, 12) * s)[ok]
        F["since_valid"][ok] = np.minimum(since, 250)[ok]
        feats.append(F)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(feats, path)
    return feats


def smooth(p, w=25):
    z = camera.logit(p)
    cs = np.concatenate(([0.0], np.cumsum(z)))
    n = len(z)
    t = np.arange(n)
    hi, lo = np.minimum(n, t + w + 1), np.maximum(0, t - w)
    return 1 / (1 + np.exp(-(cs[hi] - cs[lo]) / (hi - lo)))


def reference(lab, ph):
    """v10's own labels -> team per frame and confident frames (as camera.reference on BAR-ATH)."""
    n = len(lab)
    team_at = np.where(lab < 2, lab, -1).astype(np.int8)
    conf = np.zeros(n, bool)
    e = np.flatnonzero(np.diff(lab) != 0) + 1
    st, en = np.concatenate(([0], e)), np.concatenate((e, [n]))
    for a, b in zip(st, en):
        if lab[a] < 2 and b - a >= 6 * FPS:
            conf[a + FPS:b - FPS] = True
    conf &= np.abs(ph - 0.5) > 0.4
    return team_at, conf


def camera_probs(feats, refs, delta):
    """Cross-fitted by half inside each match: P(home at t + delta | camera at t)."""
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True, num_threads=8, verbose=-1)
    out = [None] * len(feats)
    groups = {}
    for i, b in enumerate(bench.blocks(SRC)):
        groups.setdefault(b["group"], []).append(i)
    for g, idx in groups.items():
        Xs, ys = {}, {}
        for i in idx:
            X = np.stack([feats[i][k] for k in camera.FEATS], 1).astype(np.float32)
            team_at, conf = refs[i]
            n = len(team_at)
            tgt = np.full(n, -1, np.int8)
            tgt[:n - delta] = np.where(conf[delta:], team_at[delta:], -1) if delta else np.where(conf, team_at, -1)
            Xs[i], ys[i] = X, tgt
        for i in idx:
            tr = [j for j in idx if j != i]
            X = np.concatenate([Xs[j] for j in tr])
            y = np.concatenate([ys[j] for j in tr])
            m = np.isfinite(X[:, 0]) & (y >= 0)
            p = np.full(len(Xs[i]), np.nan)
            if m.sum() > 1000 and 0.05 < (y[m] == 0).mean() < 0.95:
                mdl = lgb.train(params, lgb.Dataset(X[m], (y[m] == 0).astype(np.float32)), 300)
                have = np.isfinite(Xs[i][:, 0])
                p[have] = mdl.predict(Xs[i][have])
                prior = float((y[m] == 0).mean())
            else:
                prior = 0.5
            out[i] = (p, prior)
    return out


def fuse(ph, cam, w, delta, cap):
    p, prior = cam
    llr = np.clip(camera.logit(p) - camera.logit(prior), -cap, cap)
    ev_t = np.zeros(len(ph))
    ev_t[delta:] = np.nan_to_num(llr[:len(ph) - delta] if delta else llr, nan=0.0)
    return 1 / (1 + np.exp(-(camera.logit(ph) + w * ev_t)))


def truth():
    warnings.filterwarnings("ignore")
    feats = block_features()
    labs = bench.labels(SRC, "phase")
    dx = np.concatenate([F["dx"] for F in feats])
    lab = np.concatenate(labs)
    ok = np.isfinite(dx) & (lab < 2)
    print("SkillCorner: frames with a camera view + ball among TRUE alive frames: %.2f; AUC(home | lead room) vs the "
          "true phase team %.3f, median dx home %.1f / away %.1f" % (
              ok.sum() / (lab < 2).sum(), auc(dx[ok], lab[ok] == 0), np.median(dx[ok & (lab == 0)]),
              np.median(dx[ok & (lab == 1)])))
    # timing around TRUE changes (both spells >= 2 s), sign-aligned towards the new team
    offs = list(range(-100, 101, 10))
    acc = {o: [] for o in offs}
    nchg = 0
    for F, lb in zip(feats, labs):
        e = np.flatnonzero(np.diff(lb) != 0) + 1
        st, en = np.concatenate(([0], e)), np.concatenate((e, [len(lb)]))
        sp = [(a, b, lb[a]) for a, b in zip(st, en)]
        for (a0, b0, k0), (a1, b1, k1) in zip(sp[:-1], sp[1:]):
            if k0 > 1 or k1 > 1 or k0 == k1 or b0 - a0 < 50 or b1 - a1 < 50:
                continue
            nchg += 1
            sg = 1.0 if k1 == 0 else -1.0
            for o in offs:
                v = F["dx"][min(len(lb) - 1, max(0, a1 + o))]
                if np.isfinite(v):
                    acc[o].append(v * sg)
    print("true live changes %d; mean lead room towards the NEW team (m) by offset from the true change (s):" % nchg)
    print("  " + " ".join("%+.1f:%.2f" % (o / FPS, np.mean(v)) for o, v in acc.items() if v))
    # short true spells (< 4 s) inside alive play: does the camera turn towards their team inside them?
    n_short = turned = 0
    for F, lb in zip(feats, labs):
        e = np.flatnonzero(np.diff(lb) != 0) + 1
        st, en = np.concatenate(([0], e)), np.concatenate((e, [len(lb)]))
        for a, b in zip(st, en):
            if lb[a] < 2 and b - a < 100:
                n_short += 1
                sg = 1.0 if lb[a] == 0 else -1.0
                seg = F["dx"][a:min(len(lb), b + 25)] * sg
                turned += bool(np.nanmax(seg, initial=-99) > 3)
    print("true spells < 4 s: %d; camera lead room > 3 m towards their team somewhere in [start, end + 1 s]: %.2f" % (
        n_short, turned / max(1, n_short)))


def eval_():
    warnings.filterwarnings("ignore")
    feats = block_features()
    pr = ec.prep(SRC)
    meta = pr["meta"]
    tdd = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in pr["labs"]["phase"]]
    res = {}
    per_match = {}
    for seed in (0, 1, 2):
        raw = bench._load(ec.RAW[seed])[SRC]
        phs = [smooth(np.clip(p, 1e-4, 1 - 1e-4)) for p in raw]
        for dname, pds in (("learned D", pr["pdl"]), ("true D", tdd)):
            base = e2.decode_all(ec.DEC[SRC], pds, phs)
            refs = [reference(lb, ph) for lb, ph in zip(base, phs)]
            base_s = {c: ec.group_scores(meta, pr["labs"][c], base) for c in ("phase", "pp")}
            cams = {}
            for w, delta, cap in GRID:
                if delta not in cams:
                    cams[delta] = camera_probs(feats, refs, delta)
                fused = [fuse(ph, cam, w, delta, cap) for ph, cam in zip(phs, cams[delta])]
                preds = e2.decode_all(ec.DEC[SRC], pds, fused)
                for c in ("phase", "pp"):
                    s = ec.group_scores(meta, pr["labs"][c], preds)
                    res.setdefault((w, delta, cap, dname, c), []).append((base_s[c].mean(), s.mean(),
                                                                          (s > base_s[c]).sum(), len(s)))
                    if seed == 0 and dname == "true D" and c == "phase" and (w, delta, cap) == (1.0, 0, 4.0):
                        gs = sorted({m[0] for m in meta})
                        for gi, g in enumerate(gs):
                            idx = [i for i, m in enumerate(meta) if m[0] == g]
                            dx = np.concatenate([feats[i]["dx"] for i in idx])
                            ta = np.concatenate([refs[i][0] for i in idx])
                            cf = np.concatenate([refs[i][1] for i in idx]) & np.isfinite(dx)
                            per_match[g] = (auc(dx[cf], ta[cf] == 0), s[gi] - base_s[c][gi])
            print("seed %d %s done" % (seed, dname))
            sys.stdout.flush()
    print("fused vs v10 on SkillCorner (mean over 3 seeds; per seed wins / matches):")
    for (w, delta, cap, dname, c), v in sorted(res.items()):
        b = np.mean([x[0] for x in v])
        f = np.mean([x[1] for x in v])
        print("  w %.1f d %2d cap %.0f | %-9s %-5s v10 %.4f -> %.4f (%+.4f) wins %s" % (
            w, delta, cap, dname, c, b, f, f - b, "/".join("%d" % x[2] for x in v)))
    # does the gain grow with the cue's strength? per match: AUC of the lead room vs v10's confident labels
    # (seed 0, true D) against the per-match gain of the (1.0, 0, 4) fusion, true D, phase labels
    bench._dump(per_match, os.path.join(OUT, "per_match.pkl"))
    a = np.array([per_match[g][0] for g in sorted(per_match)])
    gain = np.array([per_match[g][1] for g in sorted(per_match)])
    order = np.argsort(a)
    print("per match (true D, phase, w1 d0 cap4, seed 0): AUC vs gain")
    for i in order:
        print("  AUC %.3f gain %+.4f" % (a[i], gain[i]))
    hi = a >= np.median(a)
    print("  matches with AUC >= median %.3f: mean gain %+.4f; below: %+.4f; corr %.2f" % (
        np.median(a), gain[hi].mean(), gain[~hi].mean(), np.corrcoef(a, gain)[0, 1]))


def veto(lab, llr_home, max_len, tau, delta):
    """Delete short v10 spells the camera contradicts: a spell of team k shorter than max_len frames whose mean camera
    log-likelihood ratio (camera at t - delta, + = home) speaks against k by more than tau becomes the previous spell's
    team when that previous spell (no dead ball between) is the other team."""
    out = lab.copy()
    n = len(lab)
    ev_t = np.zeros(n)
    ev_t[delta:] = np.nan_to_num(llr_home[:n - delta] if delta else llr_home, nan=0.0)
    e = np.flatnonzero(np.diff(lab) != 0) + 1
    st, en = np.concatenate(([0], e)), np.concatenate((e, [n]))
    for i in range(1, len(st)):
        a, b, k = st[i], en[i], lab[st[i]]
        prev = out[st[i - 1]]
        if k > 1 or prev > 1 or prev == k or b - a >= max_len:
            continue
        m = np.mean(ev_t[a:b]) * (1 if k == 0 else -1)
        if m < -tau:
            out[a:b] = prev
    return out


def eval_veto():
    warnings.filterwarnings("ignore")
    feats = block_features()
    pr = ec.prep(SRC)
    meta = pr["meta"]
    tdd = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in pr["labs"]["phase"]]
    res = {}
    for seed in (0, 1, 2):
        raw = bench._load(ec.RAW[seed])[SRC]
        phs = [smooth(np.clip(p, 1e-4, 1 - 1e-4)) for p in raw]
        for dname, pds in (("learned D", pr["pdl"]), ("true D", tdd)):
            base = e2.decode_all(ec.DEC[SRC], pds, phs)
            refs = [reference(lb, ph) for lb, ph in zip(base, phs)]
            cams = camera_probs(feats, refs, 0)
            llrs = [camera.logit(c[0]) - camera.logit(c[1]) for c in cams]
            base_s = {c: ec.group_scores(meta, pr["labs"][c], base) for c in ("phase", "pp")}
            for max_s in (4, 8):
                for tau in (0.5, 1.0, 2.0):
                    preds = [veto(lb, ll, max_s * FPS, tau, 0) for lb, ll in zip(base, llrs)]
                    for c in ("phase", "pp"):
                        s = ec.group_scores(meta, pr["labs"][c], preds)
                        res.setdefault((max_s, tau, dname, c), []).append((base_s[c].mean(), s.mean(),
                                                                           (s > base_s[c]).sum()))
            print("seed %d %s done" % (seed, dname))
            sys.stdout.flush()
    print("camera veto of short v10 spells on SkillCorner (mean over 3 seeds; wins per seed):")
    for (max_s, tau, dname, c), v in sorted(res.items()):
        b, f = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print("  spells < %d s, tau %.1f | %-9s %-5s v10 %.4f -> %.4f (%+.4f) wins %s" % (
            max_s, tau, dname, c, b, f, f - b, "/".join("%d" % x[2] for x in v)))


def _shift(x, k):
    """x[t + k] (NaN outside)."""
    out = np.full(len(x), np.nan, np.float32)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


CORE = ["dx", "bxn", "by", "bvx", "since_valid"]   # camera features without zoom / pan (production-dependent units)


def stage2_rows(F, ph, with_camera=True, fset="all"):
    """Second-stage inputs per frame: v10's smoothed logit (now, +-1 s, +-2 s) and, with_camera, the camera view
    features with their +-1 / +-2 s shifts and a +-0.5 s mean of the lead room. fset: "all" (camera.FEATS), "core"
    (no zoom / pan / dy), "norm" (core, lead room divided by the half's median |lead room|: scale-free)."""
    z = camera.logit(ph).astype(np.float32)
    cols = [z, _shift(z, 25), _shift(z, -25), _shift(z, 50), _shift(z, -50)]
    if with_camera:
        dx = F["dx"]
        if fset == "norm":
            med = np.nanmedian(np.abs(dx))
            dx = (dx / med if np.isfinite(med) and med > 0 else dx).astype(np.float32)
            F = dict(F, dx=dx)
        cs = np.concatenate(([0.0], np.cumsum(np.nan_to_num(dx))))
        cn = np.concatenate(([0.0], np.cumsum(np.isfinite(dx))))
        n = len(dx)
        t = np.arange(n)
        hi, lo = np.minimum(n, t + 13), np.maximum(0, t - 12)
        with np.errstate(all="ignore"):
            dxm = ((cs[hi] - cs[lo]) / (cn[hi] - cn[lo])).astype(np.float32)
        keys = camera.FEATS if fset == "all" else CORE
        cols += [F[k] for k in keys] + [dxm] + [_shift(dx, k) for k in (-50, -25, 25, 50)]
    return np.stack(cols, 1).astype(np.float32)


def eval_stage2():
    """Supervised fusion learned on SkillCorner with TRUE labels, leave-one-match-out: a small LightGBM predicts the
    true team from v10's logit (+ camera features), then fused logit = v10 logit + alpha * (stage-2 logit - v10
    logit), +-25 smoothing, v10's decoder. The camera-free stage 2 isolates the re-sharpening effect."""
    import lightgbm as lgb
    warnings.filterwarnings("ignore")
    feats = block_features()
    pr = ec.prep(SRC)
    meta = pr["meta"]
    labs = pr["labs"]["phase"]
    tdd = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in labs]
    groups = sorted({m[0] for m in meta})
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True, num_threads=8, verbose=-1)
    res = {}
    for seed in (0, 1, 2):
        raw = bench._load(ec.RAW[seed])[SRC]
        phs = [smooth(np.clip(p, 1e-4, 1 - 1e-4)) for p in raw]
        for cam in (False, True):
            X = [stage2_rows(F, ph, cam) for F, ph in zip(feats, phs)]
            p2 = [None] * len(X)
            for g in groups:
                tr = [i for i, m in enumerate(meta) if m[0] != g]
                te = [i for i, m in enumerate(meta) if m[0] == g]
                Xt = np.concatenate([X[i][::4] for i in tr])
                yt = np.concatenate([labs[i][::4] for i in tr])
                ok = yt < 2
                mdl = lgb.train(params, lgb.Dataset(Xt[ok], (yt[ok] == 0).astype(np.float32)), 200)
                for i in te:
                    p2[i] = mdl.predict(X[i])
            for alpha in (0.25, 0.5, 1.0):
                fused = [smooth(1 / (1 + np.exp(-(camera.logit(ph) + alpha * (camera.logit(q) - camera.logit(ph))))))
                         for ph, q in zip(phs, p2)]
                for dname, pds in (("learned D", pr["pdl"]), ("true D", tdd)):
                    base = e2.decode_all(ec.DEC[SRC], pds, phs)
                    preds = e2.decode_all(ec.DEC[SRC], pds, fused)
                    for c in ("phase", "pp"):
                        b = ec.group_scores(meta, pr["labs"][c], base)
                        s = ec.group_scores(meta, pr["labs"][c], preds)
                        res.setdefault(("camera" if cam else "no camera", alpha, dname, c), []).append(
                            (b.mean(), s.mean(), (s > b).sum()))
            print("seed %d stage 2 %s done" % (seed, "with camera" if cam else "without camera"))
            sys.stdout.flush()
    print("supervised stage 2 on SkillCorner, leave-one-match-out (mean over 3 seeds; wins per seed):")
    for (kind, alpha, dname, c), v in sorted(res.items()):
        b, f = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print("  %-9s alpha %.2f | %-9s %-5s v10 %.4f -> %.4f (%+.4f) wins %s" % (
            kind, alpha, dname, c, b, f, f - b, "/".join("%d" % x[2] for x in v)))


S2_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
                 bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True, num_threads=8, verbose=-1)


def stage2_probs(feats, phs, labs, meta, with_camera, conv_tag, fset="all"):
    """Leave-one-match-out stage-2 probabilities (cached per seed tag / camera flag / training convention / fset)."""
    import lightgbm as lgb
    path = os.path.join(OUT, "stage2_%s_%s%s.pkl" % (conv_tag, "cam" if with_camera else "nocam",
                                                     "" if fset == "all" or not with_camera else "_" + fset))
    if os.path.isfile(path):
        return bench._load(path)
    params = S2_PARAMS
    X = [stage2_rows(F, ph, with_camera, fset) for F, ph in zip(feats, phs)]
    p2 = [None] * len(X)
    for g in sorted({m[0] for m in meta}):
        tr = [i for i, m in enumerate(meta) if m[0] != g]
        Xt = np.concatenate([X[i][::4] for i in tr])
        yt = np.concatenate([labs[i][::4] for i in tr])
        ok = yt < 2
        mdl = lgb.train(params, lgb.Dataset(Xt[ok], (yt[ok] == 0).astype(np.float32)), 200)
        for i, m in enumerate(meta):
            if m[0] == g:
                p2[i] = mdl.predict(X[i])
    bench._dump(p2, path)
    return p2


def eval_stage2_delta():
    """Only the camera's MARGINAL contribution: fused logit = v10 logit + alpha * (stage-2 with camera - stage-2
    without camera). Both stage-2 models learn the same re-sharpening and the same training-label convention, so the
    difference keeps what the camera adds. Trained on phase labels, scored on phase AND pp (convention transfer)."""
    warnings.filterwarnings("ignore")
    feats = block_features()
    pr = ec.prep(SRC)
    meta = pr["meta"]
    tdd = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in pr["labs"]["phase"]]
    res = {}
    for seed in (0, 1, 2):
        raw = bench._load(ec.RAW[seed])[SRC]
        phs = [smooth(np.clip(p, 1e-4, 1 - 1e-4)) for p in raw]
        for train_conv in ("phase", "pp"):
            pc = stage2_probs(feats, phs, pr["labs"][train_conv], meta, True, "s%d_%s" % (seed, train_conv))
            pn = stage2_probs(feats, phs, pr["labs"][train_conv], meta, False, "s%d_%s" % (seed, train_conv))
            for alpha in (0.5, 1.0, 2.0):
                fused = [smooth(1 / (1 + np.exp(-(camera.logit(ph) + alpha * (camera.logit(a) - camera.logit(b))))))
                         for ph, a, b in zip(phs, pc, pn)]
                for dname, pds in (("learned D", pr["pdl"]), ("true D", tdd)):
                    base = e2.decode_all(ec.DEC[SRC], pds, phs)
                    preds = e2.decode_all(ec.DEC[SRC], pds, fused)
                    for c in ("phase", "pp"):
                        b_ = ec.group_scores(meta, pr["labs"][c], base)
                        s = ec.group_scores(meta, pr["labs"][c], preds)
                        res.setdefault((train_conv, alpha, dname, c), []).append((b_.mean(), s.mean(), (s > b_).sum()))
            print("seed %d trained on %s done" % (seed, train_conv), flush=True)
    print("camera's marginal stage-2 contribution on SkillCorner (LOMO; mean over 3 seeds; wins per seed):")
    for (tc, alpha, dname, c), v in sorted(res.items()):
        b, f = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print("  trained on %-5s alpha %.1f | %-9s scored %-5s v10 %.4f -> %.4f (%+.4f) wins %s" % (
            tc, alpha, dname, c, b, f, f - b, "/".join("%d" % x[2] for x in v)))


def eval_delta_robust():
    """Production-robust variants of the camera residual, trained on pp labels: fset core (no zoom / pan) and norm
    (core with the lead room scaled by the half's median |lead room|), alpha 0.5 / 1.0, 3 seeds."""
    warnings.filterwarnings("ignore")
    feats = block_features()
    pr = ec.prep(SRC)
    meta = pr["meta"]
    tdd = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in pr["labs"]["phase"]]
    res = {}
    for seed in (0, 1, 2):
        raw = bench._load(ec.RAW[seed])[SRC]
        phs = [smooth(np.clip(p, 1e-4, 1 - 1e-4)) for p in raw]
        pn = stage2_probs(feats, phs, pr["labs"]["pp"], meta, False, "s%d_pp" % seed)
        for fset in ("core", "norm"):
            pc = stage2_probs(feats, phs, pr["labs"]["pp"], meta, True, "s%d_pp" % seed, fset)
            for alpha in (0.5, 1.0):
                fused = [smooth(1 / (1 + np.exp(-(camera.logit(ph) + alpha * (camera.logit(a) - camera.logit(b))))))
                         for ph, a, b in zip(phs, pc, pn)]
                for dname, pds in (("learned D", pr["pdl"]), ("true D", tdd)):
                    base = e2.decode_all(ec.DEC[SRC], pds, phs)
                    preds = e2.decode_all(ec.DEC[SRC], pds, fused)
                    for c in ("phase", "pp"):
                        b_ = ec.group_scores(meta, pr["labs"][c], base)
                        s = ec.group_scores(meta, pr["labs"][c], preds)
                        res.setdefault((fset, alpha, dname, c), []).append((b_.mean(), s.mean(), (s > b_).sum()))
            print("seed %d %s done" % (seed, fset), flush=True)
    print("camera residual, robust feature sets, trained on pp (LOMO; mean over 3 seeds; wins per seed):")
    for (fset, alpha, dname, c), v in sorted(res.items()):
        b, f = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print("  %-4s alpha %.1f | %-9s scored %-5s v10 %.4f -> %.4f (%+.4f) wins %s" % (
            fset, alpha, dname, c, b, f, f - b, "/".join("%d" % x[2] for x in v)))


if __name__ == "__main__":
    {"truth": truth, "eval": eval_, "veto": eval_veto, "stage2": eval_stage2, "delta": eval_stage2_delta,
     "robust": eval_delta_robust}[sys.argv[1] if len(sys.argv) > 1 else "truth"]()
