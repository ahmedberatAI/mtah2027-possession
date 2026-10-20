"""Dataset-agnostic per-frame features for learned H/A/D models (BAR-ATH ATD and degraded DFL).

compute(P, B, team, gk, pitch) works on ONE continuous block (a half): P (n, slots, >=2)
normalised positions, B (n, >=2) normalised ball (NaN = missing), team (slots,) 0/1, gk (slots,)
bool. Returns (alive_feats: dict, team_feats_home: dict, team_feats_away: dict); the team dicts
are the same features seen from each side, so the H/A model can be trained with a side swap.
"""
import numpy as np

FPS = 25
CLIP = 500.0


def _cs(x):
    return np.concatenate(([0.0], np.cumsum(x)))


def win_mean(x, lo_off, hi_off):
    """Mean of x over [t+lo_off, t+hi_off] (inclusive), ignoring NaN; NaN if empty."""
    x = np.asarray(x, np.float64)
    n = len(x)
    ok = np.isfinite(x)
    cs, cn = _cs(np.where(ok, x, 0.0)), _cs(ok.astype(np.float64))
    t = np.arange(n)
    lo = np.clip(t + lo_off, 0, n)
    hi = np.clip(t + hi_off + 1, 0, n)
    num, den = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def since_last(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(n) - last, CLIP).astype(np.float32)


def until_next(mask):
    return since_last(mask[::-1])[::-1]


def velocity(xy):
    v = np.full(xy.shape, np.nan, np.float32)
    v[1:-1] = (xy[2:] - xy[:-2]) * FPS / 2.0
    return v


def compute(P, B, team, gk, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    n = len(P)
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    present = np.isfinite(pxy[:, :, 0])
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    seen = np.isfinite(bxy[:, 0])
    th, ta = team == 0, team == 1

    pv = velocity(pxy)
    spd = np.linalg.norm(pv, axis=2)
    spd[spd > 12] = np.nan
    spd[~present] = np.nan
    F = {}
    with np.errstate(invalid="ignore"):
        s_all = np.nanmean(spd, 1)
        s_h = np.nanmean(spd[:, th], 1)
        s_a = np.nanmean(spd[:, ta], 1)
        fast2 = np.nanmean(np.where(np.isnan(spd), np.nan, spd > 2), 1)
        fast4 = np.nanmean(np.where(np.isnan(spd), np.nan, spd > 4), 1)
        p90 = np.nanpercentile(spd, 90, axis=1)
    for name, x in (("spd", s_all), ("fast2", fast2), ("fast4", fast4), ("p90", p90)):
        for h in (12, 25, 50, 100, 250):
            F["%s_c%d" % (name, h)] = win_mean(x, -h, h)
    for lo, hi in ((0, 25), (0, 50), (0, 100), (25, 75), (-50, 0), (-100, 0), (-75, -25)):
        F["spd_w%d_%d" % (lo, hi)] = win_mean(s_all, lo, hi)
    F["spd_fwd_minus_bwd"] = F["spd_w0_50"] - F["spd_w-50_0"]
    F["spd_team_min"] = np.fmin(win_mean(s_h, -25, 25), win_mean(s_a, -25, 25))
    F["spd_team_max"] = np.fmax(win_mean(s_h, -25, 25), win_mean(s_a, -25, 25))
    F["n_players"] = present.sum(1).astype(np.float32)

    # formation triggers (Lang et al. 2022 style), on raw player positions
    x, y = pxy[:, :, 0], pxy[:, :, 1]
    with np.errstate(invalid="ignore"):
        out_side = np.nanmax(np.fmax(-y, y - W), 1)
        out_goal = np.nanmax(np.fmax(-x, x - L), 1)
        corners = np.array([[0, 0], [0, W], [L, 0], [L, W]], np.float32)
        dc = np.min(np.linalg.norm(pxy[:, :, None, :] - corners[None, None], axis=-1), axis=2)
        n_corner = np.sum(dc < 3.0, 1).astype(np.float32)
        dcen = np.nanmin(np.linalg.norm(pxy - np.array([L / 2, W / 2], np.float32), axis=2), 1)
        # goal areas (5.5 m deep, 18.32 m wide)
        in_ga = ((x < 5.5) | (x > L - 5.5)) & (np.abs(y - W / 2) < 9.16)
        n_ga = np.sum(in_ga, 1).astype(np.float32)
    F["out_side_c12"] = win_mean(out_side, -12, 12)
    F["out_goal_c12"] = win_mean(out_goal, -12, 12)
    F["n_corner_c12"] = win_mean(n_corner, -12, 12)
    F["dcen_c12"] = win_mean(dcen, -12, 12)
    F["n_goalarea_c12"] = win_mean(n_ga, -12, 12)

    # attacking direction per team from the goalkeeper (or team centroid) over the block
    with np.errstate(invalid="ignore"):
        cx_h, cx_a = np.nanmean(np.where(th, x, np.nan), 1), np.nanmean(np.where(ta, x, np.nan), 1)
        cy_h, cy_a = np.nanmean(np.where(th, y, np.nan), 1), np.nanmean(np.where(ta, y, np.nan), 1)
        sx_h, sx_a = np.nanstd(np.where(th, x, np.nan), 1), np.nanstd(np.where(ta, x, np.nan), 1)
        sy_h, sy_a = np.nanstd(np.where(th, y, np.nan), 1), np.nanstd(np.where(ta, y, np.nan), 1)
        gkx_h = np.nanmean(np.where((th & gk)[None, :], x, np.nan)) if (th & gk).any() else np.nan
    home_left = (gkx_h < L / 2) if np.isfinite(gkx_h) else (np.nanmean(cx_h) < np.nanmean(cx_a))
    sgn_h = 1.0 if home_left else -1.0          # +1: home attacks towards x = L
    F["own_half_frac_c12"] = win_mean(np.nanmean(np.where(
        th, (x - L / 2) * sgn_h < 0.5, np.where(ta, (x - L / 2) * -sgn_h < 0.5, np.nan)), 1), -12, 12)
    F["centroid_gap"] = win_mean(np.hypot(cx_h - cx_a, cy_h - cy_a), -25, 25)
    F["spread_max"] = win_mean(np.fmax(sx_h, sx_a), -25, 25)
    F["spread_min"] = win_mean(np.fmin(sx_h, sx_a), -25, 25)

    # ball
    bv = velocity(bxy)
    bspd = np.linalg.norm(bv, axis=1)
    bspd[bspd > 45] = np.nan
    F["ball_seen"] = seen.astype(np.float32)
    F["ball_seen_c25"] = win_mean(seen.astype(np.float32), -25, 25)
    F["ball_seen_c100"] = win_mean(seen.astype(np.float32), -100, 100)
    F["since_seen"] = since_last(seen)
    F["until_seen"] = until_next(seen)
    bx, by = bxy[:, 0], bxy[:, 1]
    ball_out = np.fmax(np.fmax(-bx, bx - L), np.fmax(-by, by - W))
    edge = np.fmin(np.fmin(bx, L - bx), np.fmin(by, W - by))
    F["ball_out"] = ball_out
    F["ball_out_c12"] = win_mean(ball_out, -12, 12)
    F["ball_edge"] = edge
    # last/next seen edge distance carried across gaps
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    nxt = np.where(seen, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    F["last_edge"] = np.where(last >= 0, edge[np.maximum(last, 0)], np.nan)
    F["next_edge"] = np.where(nxt < n, edge[np.minimum(nxt, n - 1)], np.nan)
    F["bspd_c6"] = win_mean(bspd, -6, 6)
    F["bspd_c25"] = win_mean(bspd, -25, 25)
    F["bspd_max_w"] = np.fmax(win_mean(bspd, 0, 25), win_mean(bspd, -25, 0))
    lo = np.clip(np.arange(n) - 12, 0, n - 1)
    hi = np.clip(np.arange(n) + 12, 0, n - 1)
    disp = np.where(seen & seen[lo] & seen[hi], np.linalg.norm(bxy[hi] - bxy[lo], axis=1), np.nan)
    F["ball_disp24"] = disp
    F["ball_still_c50"] = win_mean(np.where(np.isfinite(disp), disp < 0.6, np.nan), -50, 50)

    # ball - player relations
    d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dh = np.where(th[None, :], d, np.inf).min(1)
    da = np.where(ta[None, :], d, np.inf).min(1)
    dh = np.where(np.isinf(dh), np.nan, dh).astype(np.float32)
    da = np.where(np.isinf(da), np.nan, da).astype(np.float32)
    d1 = np.fmin(dh, da)
    F["d1"] = d1
    F["d1_c25"] = win_mean(d1, -25, 25)
    near_idx = np.argmin(d, axis=1)
    rel = np.linalg.norm(bv - pv[np.arange(n), near_idx], axis=1)
    F["rel_speed"] = np.where(seen, rel, np.nan)
    gk_d = np.where(gk[None, :], d, np.inf).min(1)
    F["gk_ball_d"] = np.where(np.isinf(gk_d), np.nan, gk_d)

    # team-perspective features (home view; away view = swapped)
    def side(d_own, d_opp, s_own, s_opp, sx_own, sx_opp, sy_own, sy_opp, cx_own, sgn):
        T = {}
        ctrl = seen & (d_own <= 1.5) & ~(d_opp < d_own + 0.5)
        ctrl2 = seen & (d_own <= 1.0) & ~(d_opp < d_own + 0.3)
        T["d_own"] = d_own
        T["d_diff"] = d_opp - d_own
        T["d_diff_c12"] = win_mean(d_opp - d_own, -12, 12)
        for nm, m in (("ctrl", ctrl), ("ctrl2", ctrl2)):
            mm = m.astype(np.float32)
            T[nm] = mm
            for lo_, hi_ in ((-25, 0), (0, 25), (-75, 0), (0, 75), (-200, 0), (0, 200)):
                T["%s_w%d_%d" % (nm, lo_, hi_)] = win_mean(np.where(seen, mm, np.nan), lo_, hi_)
            T[nm + "_since"] = since_last(m)
            T[nm + "_until"] = until_next(m)
        T["spd_own_c25"] = win_mean(s_own, -25, 25)
        T["spread_x_diff"] = win_mean(sx_own - sx_opp, -25, 25)
        T["spread_y_diff"] = win_mean(sy_own - sy_opp, -25, 25)
        att = (cx_own - L / 2) * sgn
        T["centroid_att"] = win_mean(att, -25, 25)
        vel = np.full(n, np.nan, np.float32)
        vel[25:-25] = (att[50:] - att[:-50]) / 2.0
        T["centroid_att_vel"] = win_mean(vel, -12, 12)
        T["ball_att"] = (bx - L / 2) * sgn
        return T

    T_h = side(dh, da, s_h, s_a, sx_h, sx_a, sy_h, sy_a, cx_h, sgn_h)
    T_a = side(da, dh, s_a, s_h, sx_a, sx_h, sy_a, sy_h, cx_a, -sgn_h)
    return F, T_h, T_a
