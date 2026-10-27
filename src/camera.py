"""Round 11, idea E6 (docs/RAPOR.md): broadcast camera framing as team-possession evidence on BAR-ATH.

TV camera operators frame ahead of the ball in the direction of play ("lead room"): the pitch point under the image
centre (broadcast homography) sits ahead of the ball towards the goal the possessing team attacks. src/exp_framing.py:
on v10's confident frames this lead room separates home from away with AUC 0.95 (0.94-0.97 in every ball-position
bin, 0.97-0.99 in every ball-velocity bin), on 88% of v10's live frames, and it flips ~1.7 s before v10's own changes.

This module turns it into a per-frame log-likelihood ratio and fuses it with v10's team probabilities:
  camera model   small LightGBM on camera-view features (lead room, ball position, zoom, pan, ball velocity) with
                 v10's confident labels at t + delta as target, cross-fitted by half (no frame predicts itself)
  fusion         logit p(t) = logit p_v10(t) + w * clip(logit p_cam(t - delta) - logit prior, +-cap)
  decoding       v10's decoder unchanged (model_v7.predict with v10's DEFAULT)
Label-free: the camera model learns from v10's own confident frames, never from the ground truth. AGGREGATE output.

    python -I src/camera.py fit            # camera-model quality (cross-fitted AUC on v10-confident frames)
    python -I src/camera.py run W DELTA CAP  # fused decode: label-free diagnostics (+ public window, info only)
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import commentary as cm  # noqa: E402
import evaluate as ev  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "cache", "camera")
FPS = 25
L, W = 105.3, 68.0
N = atd.N_FRAMES["tactical"]
FEATS = ["dx", "dy", "bxn", "by", "zoom", "pan", "bvx", "since_valid"]


def _win_diff(x, k):
    """(x[t+k] - x[t-k]) / (2k / FPS), NaN at the edges."""
    out = np.full(len(x), np.nan)
    out[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k / FPS)
    return out


def cue_features(refresh=False):
    """Per tactical frame (NaN where the broadcast has no valid view or no ball)."""
    path = os.path.join(OUT, "cue_features.pkl")
    if os.path.isfile(path) and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import pandas as pd
    hom = pd.read_csv(atd.paths("broadcast")["homography"])
    Hm = hom[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].values.reshape(-1, 3, 3)
    valid = hom["valid"].values.astype(bool)
    nb = len(valid)
    aim = np.full((nb, 2), np.nan)
    zoom = np.full(nb, np.nan)
    idx = np.flatnonzero(valid)
    q = np.linalg.inv(Hm[idx]) @ np.array([960.0, 540.0, 1.0])
    aim[idx] = q[:, :2] / q[:, 2:3]
    # zoom: image pixels per metre along the pitch length at the aimed point
    p1 = np.stack([aim[idx, 0] + 1.0 / L, aim[idx, 1], np.ones(len(idx))], 1)
    r = np.einsum("nij,nj->ni", Hm[idx], p1)
    zoom[idx] = np.hypot(r[:, 0] / r[:, 2] - 960.0, r[:, 1] / r[:, 2] - 540.0)
    ball_b = atd.load("broadcast")["ball"][:nb, :2].astype(float)
    # frames since the view (re)became valid (cuts back from replays / close-ups)
    since = np.zeros(nb)
    run = 0
    for i in range(nb):
        run = run + 1 if valid[i] else 0
        since[i] = run
    tac = atd.load("tactical")
    team, slots = tac["team"], tac["slots"]
    gk = slots["position_type"].values == "Goalkeeper"
    sgn = np.zeros(N)
    for _, a, b in atd.HALVES:
        x = tac["players"][a:b + 1, :, 0]
        sgn[a:b + 1] = 1.0 if np.nanmean(x[:, (team == 0) & gk]) < 0.5 else -1.0
    t_of_b = np.where(np.arange(nb) <= cm.CUT_B, np.arange(nb) + 4014, np.arange(nb) + 26920)
    ok = valid & np.isfinite(ball_b[:, 0]) & (t_of_b < N)
    s_b = sgn[np.clip(t_of_b, 0, N - 1)]
    pan_b = _win_diff(aim[:, 0] * L, 12) * s_b
    bvx_b = _win_diff(ball_b[:, 0] * L, 12) * s_b
    F = {k: np.full(N, np.nan, np.float32) for k in FEATS}
    t = t_of_b[ok]
    F["dx"][t] = (aim[ok, 0] - ball_b[ok, 0]) * L * s_b[ok]
    F["dy"][t] = (aim[ok, 1] - ball_b[ok, 1]) * W
    F["bxn"][t] = np.where(s_b[ok] > 0, ball_b[ok, 0], 1 - ball_b[ok, 0])
    F["by"][t] = ball_b[ok, 1]
    F["zoom"][t] = zoom[ok]
    F["pan"][t] = pan_b[ok]
    F["bvx"][t] = bvx_b[ok]
    F["since_valid"][t] = np.minimum(since[ok], 250)
    os.makedirs(OUT, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(F, fh, protocol=4)
    return F


def sc_style_features(refresh=False):
    """Camera features with the definitions of src/camera_sc.py (SkillCorner), per tactical frame: the horizontal aim
    is where the image's centre column, mapped to the pitch, crosses the ball's depth; zoom is 1 / the width (m) of the
    image's bottom edge on the pitch; dy is 0 by construction (the aim is taken at the ball's depth)."""
    path = os.path.join(OUT, "sc_style_features.pkl")
    if os.path.isfile(path) and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import pandas as pd
    hom = pd.read_csv(atd.paths("broadcast")["homography"])
    Hm = hom[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].values.reshape(-1, 3, 3)
    valid = hom["valid"].values.astype(bool)
    nb = len(valid)
    idx = np.flatnonzero(valid)
    inv = np.linalg.inv(Hm[idx])

    def g(px, py):
        q = inv @ np.array([px, py, 1.0])
        return q[:, :2] / q[:, 2:3]
    c0, c1 = g(960.0, 540.0), g(960.0, 1080.0)
    bl, br = g(0.0, 1080.0), g(1920.0, 1080.0)
    ball_b = atd.load("broadcast")["ball"][:nb, :2].astype(float)
    by_ = ball_b[idx, 1]
    with np.errstate(all="ignore"):
        s_ = (by_ - c0[:, 1]) / (c1[:, 1] - c0[:, 1])
        ax = c0[:, 0] + s_ * (c1[:, 0] - c0[:, 0])
        width = np.hypot((br[:, 0] - bl[:, 0]) * L, (br[:, 1] - bl[:, 1]) * W)
    aim_x = np.full(nb, np.nan)
    zoom = np.full(nb, np.nan)
    aim_x[idx] = ax
    zoom[idx] = 1.0 / width
    since = np.zeros(nb)
    run = 0
    for i in range(nb):
        run = run + 1 if valid[i] else 0
        since[i] = run
    tac = atd.load("tactical")
    team, slots = tac["team"], tac["slots"]
    gk = slots["position_type"].values == "Goalkeeper"
    sgn = np.zeros(N)
    for _, a, b in atd.HALVES:
        x = tac["players"][a:b + 1, :, 0]
        sgn[a:b + 1] = 1.0 if np.nanmean(x[:, (team == 0) & gk]) < 0.5 else -1.0
    t_of_b = np.where(np.arange(nb) <= cm.CUT_B, np.arange(nb) + 4014, np.arange(nb) + 26920)
    s_b = sgn[np.clip(t_of_b, 0, N - 1)]
    ok = valid & np.isfinite(ball_b[:, 0]) & np.isfinite(aim_x) & (t_of_b < N)
    pan_b = _win_diff(aim_x * L, 12) * s_b
    bvx_b = _win_diff(ball_b[:, 0] * L, 12) * s_b
    F = {k: np.full(N, np.nan, np.float32) for k in FEATS}
    t = t_of_b[ok]
    F["dx"][t] = (aim_x[ok] - ball_b[ok, 0]) * L * s_b[ok]
    F["dy"][t] = 0.0
    F["bxn"][t] = np.where(s_b[ok] > 0, ball_b[ok, 0], 1 - ball_b[ok, 0])
    F["by"][t] = ball_b[ok, 1]
    F["zoom"][t] = zoom[ok]
    F["pan"][t] = pan_b[ok]
    F["bvx"][t] = bvx_b[ok]
    F["since_valid"][t] = np.minimum(since[ok], 250)
    os.makedirs(OUT, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(F, fh, protocol=4)
    return F


def camera_residual_probs(alpha=1.0, fset="core", train_conv="pp"):
    """v10's smoothed probabilities plus the camera's marginal contribution learned on SkillCorner with TRUE labels
    (src/camera_sc.py: stage 2 with camera minus stage 2 without camera, both trained on all 20 matches with v10's
    seed-0 logits), +-25 smoothing as in the SkillCorner check. Written in v10's cache format."""
    import lightgbm as lgb
    import bench
    import camera_sc as cs
    import exp_contest as ec
    sc_feats = cs.block_features()
    pr = ec.prep(cs.SRC)
    labs = pr["labs"][train_conv]
    raw = bench._load(ec.RAW[0])[cs.SRC]
    phs = [cs.smooth(np.clip(p, 1e-4, 1 - 1e-4)) for p in raw]
    models = {}
    for cam in (True, False):
        X = np.concatenate([cs.stage2_rows(F, ph, cam, fset)[::4] for F, ph in zip(sc_feats, phs)])
        y = np.concatenate([lb[::4] for lb in labs])
        ok = y < 2
        models[cam] = lgb.train(cs.S2_PARAMS, lgb.Dataset(X[ok], (y[ok] == 0).astype(np.float32)), 200)
    F = sc_style_features()
    with open(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"), "rb") as fh:
        probs = pickle.load(fh)
    out = {}
    for a, (pd_, p) in probs.items():
        Fb = {k: v[a:a + len(p)] for k, v in F.items()}
        pc = models[True].predict(cs.stage2_rows(Fb, p, True, fset))
        pn = models[False].predict(cs.stage2_rows(Fb, p, False, fset))
        z = logit(p) + alpha * (logit(pc) - logit(pn))
        out[a] = (pd_, cs.smooth(1 / (1 + np.exp(-z))))
    path = os.path.join(OUT, "probs_resid_a%g_%s_%s.pkl" % (alpha, fset, train_conv))
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return path


def reference(codes, ph):
    """v10's team per frame (-1 outside spells) and its confident frames (spells >= 6 s, 1 s from the ends,
    |p - 0.5| > 0.4)."""
    team_at = np.full(N, -1, np.int8)
    conf = np.zeros(N, bool)
    for a, e, k in cm.spells(codes):
        team_at[a:e] = k
        if e - a >= 6 * FPS:
            conf[a + FPS:e - FPS] = True
    conf &= np.abs(np.nan_to_num(ph, nan=0.5) - 0.5) > 0.4
    return team_at, conf


def camera_probs(delta, refresh=False):
    """P(home in possession at t + delta | camera view at t), cross-fitted by half; NaN without a cue."""
    path = os.path.join(OUT, "pcam_d%d.pkl" % delta)
    if os.path.isfile(path) and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import lightgbm as lgb
    F = cue_features()
    X = np.stack([F[k] for k in FEATS], 1).astype(np.float32)
    have = np.isfinite(X[:, 0])
    codes, ph = cm.v10_state()
    team_at, conf = reference(codes, ph)
    tgt = np.full(N, -1, np.int8)
    tgt[:N - delta] = np.where(conf[delta:], team_at[delta:], -1) if delta > 0 else np.where(conf, team_at, -1)
    half = atd.half_of_frame(N)
    p = np.full(N, np.nan)
    stats = []
    for h in (1, 2):
        tr = have & (tgt >= 0) & (half != h) & (half > 0)
        te = have & (half == h)
        params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
                      feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True,
                      num_threads=8, verbose=-1)
        mdl = lgb.train(params, lgb.Dataset(X[tr], (tgt[tr] == 0).astype(np.float32)), 300)
        p[te] = mdl.predict(X[te])
        ev_m = te & (tgt >= 0)
        from exp_framing import auc
        stats.append((h, ev_m.sum(), auc(p[ev_m], tgt[ev_m] == 0)))
    prior = float((tgt[have & (tgt >= 0)] == 0).mean())
    out = {"p": p, "prior": prior, "stats": stats}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fused_probs(w, delta, cap):
    """v10's smoothed probabilities with the camera log-likelihood ratio added; written in v10's cache format."""
    cam = camera_probs(delta)
    llr = np.clip(logit(cam["p"]) - logit(cam["prior"]), -cap, cap)
    ev_t = np.zeros(N)
    # camera at t - delta speaks about the label at t
    ev_t[delta:] = np.nan_to_num(llr[:N - delta] if delta > 0 else llr, nan=0.0)
    with open(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"), "rb") as fh:
        probs = pickle.load(fh)
    out = {}
    for a, (pd_, p) in probs.items():
        z = logit(p) + w * ev_t[a:a + len(p)]
        out[a] = (pd_, 1 / (1 + np.exp(-z)))
    path = os.path.join(OUT, "probs_w%g_d%d_c%g.pkl" % (w, delta, cap))
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return path


def decode(probs_path):
    import make_submission
    import model_v10
    import model_v7
    f = make_submission.merged_features()
    return model_v7.predict(f, dict(model_v10.DEFAULT), probs_path=probs_path)


def describe(name, labels, base=None):
    d = ev.diagnostics(labels)
    s = ev.score(labels)
    sp = cm.spells(np.asarray(ev.to_codes(labels)))
    lens = np.array([e - a for a, e, _ in sp]) / FPS
    extra = ""
    if base is not None:
        g = atd.graded_mask(N)
        extra = " | frames changed vs v10 %.2f%%" % (100 * np.mean(np.asarray(labels)[g] != np.asarray(base)[g]))
    print("%-24s full match: %d poss (<2 s %d, 2-4 s %d), %d D seg, D share %.3f%s | public (info) spellF1 %.3f "
          "macroF1 %.3f team %.3f n_pred %d/%d" % (
              name, d["n_poss"], (lens < 2).sum(), ((lens >= 2) & (lens < 4)).sum(), d["n_dead_segs"],
              d["dead_share"], extra, s["spellF1"], s["macroF1"], s["team_acc"], s["n_pred"], s["n_gt"]))


def cmd_fit():
    for delta in (0, 25, 50):
        cam = camera_probs(delta)
        print("delta %d frames: cross-fitted AUC on v10-confident frames %s; prior home %.2f" % (
            delta, ", ".join("half %d %.3f (n %d)" % (h, a, nn) for h, nn, a in cam["stats"]), cam["prior"]))


def cmd_run(w, delta, cap):
    codes, _ = cm.v10_state()
    base = decode(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"))
    describe("v10", base)
    lab = decode(fused_probs(w, delta, cap))
    describe("camera w%g d%d c%g" % (w, delta, cap), lab, base)


def cmd_grid():
    base = decode(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"))
    describe("v10", base)
    for w, delta, cap in ((0.5, 0, 4), (1.0, 0, 4), (0.5, 25, 4), (1.0, 25, 4), (0.5, 50, 4), (1.0, 50, 4),
                          (1.0, 25, 2), (1.0, 25, 8)):
        lab = decode(fused_probs(w, delta, cap))
        describe("camera w%g d%d c%g" % (w, delta, cap), lab, base)
        sys.stdout.flush()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        cmd_fit()
    elif cmd == "grid":
        cmd_grid()
    elif cmd == "resid":
        base = decode(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"))
        describe("v10", base)
        for alpha, fset in ((0.5, "core"), (1.0, "core"), (0.5, "norm"), (1.0, "norm"), (1.0, "all")):
            lab = decode(camera_residual_probs(alpha, fset))
            describe("camera residual a%g %s" % (alpha, fset), lab, base)
            sys.stdout.flush()
    else:
        cmd_run(float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
