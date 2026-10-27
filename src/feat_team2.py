"""Extra team-perspective features for the H/A model (docs/RAPOR.md group C), bench plugins:
bench.run("base+feat_team2:c1", ...). Every function takes ONE half -- P (n, slots, 2) normalised 0-1,
NaN = missing; B (n, 2) normalised, NaN = missing; team (slots,) 0/1; gk (slots,) bool; pitch (L, W) m --
and returns (F_extra, T_home, T_away). All features are team-perspective (T dicts: the same feature seen
from each side, for the side-swap training); F_extra stays empty so the dead-ball model's inputs do not
change.

  c1  set-piece formation (needs no ball): own player beyond a touch / goal line, at a corner flag, own
      goalkeeper in its goal area with no opponent in the penalty box (goal kick), kick-off shape, a tight
      cluster of slow own players (wall), windowed means and frames since / until each shape
  c3  long context (+-10 s, +-20 s, one-sided 20 s) of the control share and the distance margin
  c5  team movement: own players' velocity towards the opponent goal, share running to / away from the
      ball, own - opponent tempo
  c2  ball-carrier persistence: hand-overs within the team and gains from the opponent per window, frames
      since the last own touch
  c4  team shape: width / depth (p90 - p10) and their own - opponent differences
"""
import numpy as np

from mlfeat import CLIP, FPS, since_last, until_next, velocity, win_mean

GA_DEPTH, GA_HALF_W = 5.5, 9.16      # goal area
PA_DEPTH, PA_HALF_W = 16.5, 20.16    # penalty area
CIRCLE = 9.15


def _prep(P, B, team, gk, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    x, y = pxy[:, :, 0], pxy[:, :, 1]
    th, ta = team == 0, team == 1
    gk = np.asarray(gk, bool)
    with np.errstate(invalid="ignore"):
        cx_h = np.nanmean(np.where(th, x, np.nan), 1)
        cx_a = np.nanmean(np.where(ta, x, np.nan), 1)
        gkx_h = np.nanmean(np.where((th & gk)[None, :], x, np.nan)) if (th & gk).any() else np.nan
    # attacking direction: the rule of mlfeat.compute (home goalkeeper's side, else the team centroids)
    home_left = (gkx_h < L / 2) if np.isfinite(gkx_h) else (np.nanmean(cx_h) < np.nanmean(cx_a))
    sgn_h = 1.0 if home_left else -1.0
    return {"L": L, "W": W, "pxy": pxy, "bxy": bxy, "x": x, "y": y, "n": len(P), "th": th, "ta": ta,
            "gk": gk, "seen": np.isfinite(bxy[:, 0]), "sgn": {0: sgn_h, 1: -sgn_h}}


def _f32(d):
    return {k: np.asarray(v, np.float32) for k, v in d.items()}


def _both(S, side_fn):
    th = side_fn(S, 0)
    ta = side_fn(S, 1)
    return {}, _f32(th), _f32(ta)


def _mask_team(S, k):
    own = S["th"] if k == 0 else S["ta"]
    opp = S["ta"] if k == 0 else S["th"]
    return own, opp


def _count(m):
    return m.sum(1).astype(np.float32)


# -- C1: set-piece formation ---------------------------------------------------------------------------

def _wall(x, y, own, slow, r=1.2, chunk=8000):
    """Largest group of slow own players standing within r of one player (a defensive wall), per frame."""
    idx = np.flatnonzero(own)
    n = len(x)
    out = np.zeros(n, np.float32)
    if len(idx) < 2:
        return out
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        xx, yy = x[sl][:, idx], y[sl][:, idx]
        ok = np.isfinite(xx) & slow[sl][:, idx]
        dd = np.hypot(xx[:, :, None] - xx[:, None, :], yy[:, :, None] - yy[:, None, :])
        close = (dd < r) & ok[:, :, None] & ok[:, None, :]
        out[sl] = close.sum(2).max(1)          # includes the player itself; 1 = nobody close
    return out


def _side_c1(S, k):
    L, W, x, y = S["L"], S["W"], S["x"], S["y"]
    own, opp = _mask_team(S, k)
    sgn = S["sgn"][k]
    ox, oy = np.where(own[None, :], x, np.nan), np.where(own[None, :], y, np.nan)
    px, py = np.where(opp[None, :], x, np.nan), np.where(opp[None, :], y, np.nan)
    goal_x = 0.0 if sgn > 0 else L           # own goal line
    T = {}
    with np.errstate(invalid="ignore"):
        out_side = np.nanmax(np.fmax(-oy, oy - W), 1)
        out_goal = np.nanmax(np.fmax(-ox, ox - L), 1)
        corners = np.array([[0, 0], [0, W], [L, 0], [L, W]], np.float32)
        dcor = np.nanmin(np.min(np.hypot(ox[:, :, None] - corners[None, None, :, 0],
                                         oy[:, :, None] - corners[None, None, :, 1]), axis=2), 1)
        in_ga = lambda xx, yy, gx: (np.abs(xx - gx) < GA_DEPTH) & (np.abs(yy - W / 2) < GA_HALF_W)
        in_pa = lambda xx, yy, gx: (np.abs(xx - gx) < PA_DEPTH) & (np.abs(yy - W / 2) < PA_HALF_W)
        gk_own = own & S["gk"]
        gk_present = np.isfinite(x[:, gk_own]).any(1) if gk_own.any() else np.zeros(S["n"], bool)
        gk_in_ga = (in_ga(x[:, gk_own], y[:, gk_own], goal_x).any(1) if gk_own.any()
                    else np.zeros(S["n"], bool))
        opp_in_own_pa = _count(in_pa(px, py, goal_x))
        own_in_opp_pa = _count(in_pa(ox, oy, L - goal_x))
        rel_half = sgn * (ox - L / 2)          # < 0: own half
        n_own = np.isfinite(ox).sum(1)
        own_half_share = np.where(n_own > 0, _count(rel_half < 0.5) / np.maximum(n_own, 1), np.nan)
        own_centre_d = np.nanmin(np.hypot(ox - L / 2, oy - W / 2), 1)
        opp_in_circle = _count(np.hypot(px - L / 2, py - W / 2) < CIRCLE)
    v = velocity(S["pxy"])
    spd = np.linalg.norm(v, axis=2)
    slow = np.isfinite(spd) & (spd < 1.0)
    wall = _wall(x, y, own, slow)
    goal_kick = gk_in_ga & gk_present & (opp_in_own_pa == 0)
    kick_off = (own_centre_d < 2.0) & (own_half_share > 0.85) & (opp_in_circle == 0)
    shapes = {"beyond_side": out_side > -0.3, "beyond_goal": out_goal > -0.3, "at_corner": dcor < 2.0,
              "goal_kick": goal_kick, "kick_off": kick_off, "wall3": wall >= 3}
    cont = {"out_side": out_side, "out_goal": out_goal, "corner_d": np.fmin(dcor, 30.0),
            "opp_in_own_pa": opp_in_own_pa, "own_in_opp_pa": own_in_opp_pa, "own_half_share": own_half_share,
            "centre_d": np.fmin(own_centre_d, 30.0), "opp_in_circle": opp_in_circle, "wall": wall}
    for nm, val in cont.items():
        T["x1_%s_c12" % nm] = win_mean(val, -12, 12)
        T["x1_%s_c50" % nm] = win_mean(val, -50, 50)
    for nm, m in shapes.items():
        T["x1_%s_c50" % nm] = win_mean(m.astype(np.float32), -50, 50)
        T["x1_%s_since" % nm] = since_last(m)
        T["x1_%s_until" % nm] = until_next(m)
    return T


def c1(P, B, team, gk, pitch):
    return _both(_prep(P, B, team, gk, pitch), _side_c1)


# -- ball-player distances shared by C2 / C3 / C5 --------------------------------------------------------

def _dist(S):
    if "d" not in S:
        d = np.linalg.norm(S["pxy"] - S["bxy"][:, None, :], axis=2)
        S["d"] = np.where(np.isfinite(d), d, np.inf)
        dh = np.where(S["th"][None, :], S["d"], np.inf).min(1)
        da = np.where(S["ta"][None, :], S["d"], np.inf).min(1)
        S["dk"] = {0: np.where(np.isinf(dh), np.nan, dh), 1: np.where(np.isinf(da), np.nan, da)}
    return S


# -- C3: long context ---------------------------------------------------------------------------------

def _side_c3(S, k):
    _dist(S)
    d_own, d_opp = S["dk"][k], S["dk"][1 - k]
    seen = S["seen"]
    with np.errstate(invalid="ignore"):
        ctrl = seen & (d_own <= 1.5) & ~(d_opp < d_own + 0.5)
        nearest = np.where(seen & np.isfinite(d_own) & np.isfinite(d_opp), (d_own < d_opp).astype(np.float32),
                           np.nan)
        margin = np.clip(d_opp - d_own, -10, 10)
    c = np.where(seen, ctrl.astype(np.float32), np.nan)
    T = {}
    for lo, hi in ((-250, 250), (-500, 500), (-500, 0), (0, 500)):
        T["x3_ctrl_w%d_%d" % (lo, hi)] = win_mean(c, lo, hi)
        T["x3_near_w%d_%d" % (lo, hi)] = win_mean(nearest, lo, hi)
    for lo, hi in ((-250, 250), (-500, 500)):
        T["x3_margin_w%d_%d" % (lo, hi)] = win_mean(margin, lo, hi)
    return T


def c3(P, B, team, gk, pitch):
    return _both(_prep(P, B, team, gk, pitch), _side_c3)


# -- C5: team movement -----------------------------------------------------------------------------------

def _side_c5(S, k):
    _dist(S)
    if "v" not in S:
        v = velocity(S["pxy"])
        sp = np.linalg.norm(v, axis=2)
        v[sp > 12] = np.nan
        S["v"], S["sp"] = v, np.where(sp > 12, np.nan, sp)
    v, sp = S["v"], S["sp"]
    own, opp = _mask_team(S, k)
    sgn = S["sgn"][k]
    to = S["bxy"][:, None, :] - S["pxy"]
    dist = np.linalg.norm(to, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v * to, axis=2) / np.maximum(sp * dist, 1e-6)
        moving = sp > 1.0
        valid = np.isfinite(sp) & np.isfinite(cos) & own[None, :]
        n_valid = valid.sum(1)
        toward = np.where(n_valid > 0, (valid & moving & (cos > 0.7)).sum(1) / np.maximum(n_valid, 1), np.nan)
        away = np.where(n_valid > 0, (valid & moving & (cos < -0.7)).sum(1) / np.maximum(n_valid, 1), np.nan)
        fwd = np.nanmean(np.where(own[None, :], sgn * v[:, :, 0], np.nan), 1)
        s_own = np.nanmean(np.where(own[None, :], sp, np.nan), 1)
        s_opp = np.nanmean(np.where(opp[None, :], sp, np.nan), 1)
        fwd_opp = np.nanmean(np.where(opp[None, :], -sgn * v[:, :, 0], np.nan), 1)
    T = {}
    for lo, hi in ((-12, 12), (-50, 50)):
        T["x5_fwd_w%d_%d" % (lo, hi)] = win_mean(fwd, lo, hi)
        T["x5_fwd_diff_w%d_%d" % (lo, hi)] = win_mean(fwd - fwd_opp, lo, hi)
        T["x5_toward_w%d_%d" % (lo, hi)] = win_mean(toward, lo, hi)
        T["x5_away_w%d_%d" % (lo, hi)] = win_mean(away, lo, hi)
    for lo, hi in ((-25, 25), (-100, 100), (0, 50), (-50, 0)):
        T["x5_spd_diff_w%d_%d" % (lo, hi)] = win_mean(s_own - s_opp, lo, hi)
    return T


def c5(P, B, team, gk, pitch):
    return _both(_prep(P, B, team, gk, pitch), _side_c5)


# -- C2: ball-carrier persistence ------------------------------------------------------------------------

def _owners(S, r=1.5, carry=50):
    _dist(S)
    if "owner" not in S:
        d = S["d"]
        j = np.argmin(d, 1)
        near = d[np.arange(S["n"]), j]
        team = np.where(S["th"], 0, np.where(S["ta"], 1, -1))
        owner = np.where(near <= r, j, -1)
        # previous distinct owner (carried over gaps of up to `carry` frames)
        n = S["n"]
        has = owner >= 0
        idx = np.where(has, np.arange(n), -1)
        last = np.maximum.accumulate(idx)
        change = np.zeros(n, bool)
        prev_team = np.full(n, -1)
        t_has = np.flatnonzero(has)
        if len(t_has) > 1:
            o = owner[t_has]
            gap = np.diff(t_has)
            ch = (o[1:] != o[:-1]) & (gap <= carry)
            change[t_has[1:][ch]] = True
            prev_team[t_has[1:][ch]] = team[o[:-1][ch]]
        S["owner"], S["owner_team"] = owner, np.where(has, team[np.maximum(owner, 0)], -1)
        S["change"], S["prev_team"], S["last_owner_frame"] = change, prev_team, last
    return S


def _side_c2(S, k):
    _owners(S)
    ot, ch, pt = S["owner_team"], S["change"], S["prev_team"]
    hand = (ch & (ot == k) & (pt == k)).astype(np.float32)        # own -> own (pass / hand-over)
    gain = (ch & (ot == k) & (pt == 1 - k)).astype(np.float32)    # opponent -> own
    touch = ot == k
    T = {}
    for lo, hi in ((-125, 125), (-250, 250), (-250, 0), (0, 250)):
        T["x2_hand_w%d_%d" % (lo, hi)] = win_mean(hand, lo, hi) * (hi - lo + 1)
        T["x2_gain_w%d_%d" % (lo, hi)] = win_mean(gain, lo, hi) * (hi - lo + 1)
    T["x2_touch_since"] = since_last(touch)
    T["x2_touch_until"] = until_next(touch)
    T["x2_hand_since"] = since_last(hand > 0)
    T["x2_gain_since"] = since_last(gain > 0)
    T["x2_gain_until"] = until_next(gain > 0)
    return T


def c2(P, B, team, gk, pitch):
    return _both(_prep(P, B, team, gk, pitch), _side_c2)


# -- C4: team shape ------------------------------------------------------------------------------------

def _side_c4(S, k):
    if "shape" not in S:
        shape = {}
        for kk, own in ((0, S["th"]), (1, S["ta"])):
            m = own & ~S["gk"]
            xs = np.where(m[None, :], S["x"], np.nan)
            ys = np.where(m[None, :], S["y"], np.nan)
            with np.errstate(invalid="ignore"):
                depth = np.nanpercentile(xs, 90, axis=1) - np.nanpercentile(xs, 10, axis=1)
                width = np.nanpercentile(ys, 90, axis=1) - np.nanpercentile(ys, 10, axis=1)
            shape[kk] = (depth, width)
        S["shape"] = shape
    (d_own, w_own), (d_opp, w_opp) = S["shape"][k], S["shape"][1 - k]
    T = {}
    for lo, hi in ((-25, 25), (-100, 100)):
        T["x4_width_w%d_%d" % (lo, hi)] = win_mean(w_own, lo, hi)
        T["x4_depth_w%d_%d" % (lo, hi)] = win_mean(d_own, lo, hi)
        T["x4_width_diff_w%d_%d" % (lo, hi)] = win_mean(w_own - w_opp, lo, hi)
        T["x4_depth_diff_w%d_%d" % (lo, hi)] = win_mean(d_own - d_opp, lo, hi)
    return T


def c4(P, B, team, gk, pitch):
    return _both(_prep(P, B, team, gk, pitch), _side_c4)


# -- combinations ----------------------------------------------------------------------------------------

def _combo(P, B, team, gk, pitch, fns):
    S = _prep(P, B, team, gk, pitch)
    th, ta = {}, {}
    for fn in fns:
        th.update(fn(S, 0))
        ta.update(fn(S, 1))
    return {}, _f32(th), _f32(ta)


def c13(P, B, team, gk, pitch):
    return _combo(P, B, team, gk, pitch, (_side_c1, _side_c3))


def c135(P, B, team, gk, pitch):
    return _combo(P, B, team, gk, pitch, (_side_c1, _side_c3, _side_c5))


def c35(P, B, team, gk, pitch):
    return _combo(P, B, team, gk, pitch, (_side_c3, _side_c5))


def c45(P, B, team, gk, pitch):
    return _combo(P, B, team, gk, pitch, (_side_c4, _side_c5))


def c145(P, B, team, gk, pitch):
    return _combo(P, B, team, gk, pitch, (_side_c1, _side_c4, _side_c5))


def c12345(P, B, team, gk, pitch):
    return _combo(P, B, team, gk, pitch, (_side_c1, _side_c2, _side_c3, _side_c4, _side_c5))
