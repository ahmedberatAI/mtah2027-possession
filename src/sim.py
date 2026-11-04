"""Degrade clean DFL tracking so that it looks like Metrica's automatic single-camera ATD.

Calibrated to BAR-ATH aggregates: ball missing ~41% of graded frames in long runs (visible runs
median ~14 s, missing runs median ~11 s), gaps starting more often while the ball is dead or in
the air; aerial balls projected to the ground through a high side camera (parallax); player
tracks fragmented (~7% of player-frames missing) with small position noise.
"""
import numpy as np

FPS = 25
DEFAULT = dict(alive_missing=0.12, dead_missing=0.70, miss_median_s=10.7, alive_gap_median_s=5.0,
               miss_sigma=0.9,
               air_mult=2.5, air_z=1.8, cam=(0.5, -0.45, 25.0), ball_noise=0.35,
               p_noise=0.15, p_present_median_s=35.0, p_missing_median_s=2.0, p_sigma=1.2,
               p_target_missing=0.07)
# ball_model "v2" (ball_dropout_v2): pass as degrade(..., params=V2). Targets from BAR-ATH aggregates
# (src/ball_gaps.py, docs/RAPOR.md Ek C; merged ball = what the team model sees): ~6.5%
# missing in play, ~70% while dead; the ball is already gone on the last alive frame before ~24% of
# stoppages (median ~2.7 s earlier) and still gone on the first alive frame after ~22% of restarts
# (median ~2 s more). V2_TACTICAL: tactical-ball rates (alive/dead = the v1 targets 12% / 70%).
V2 = dict(ball_model="v2", v2_alive_missing=0.065, v2_dead_missing=0.70, restart_missing=0.22,
          carry_median_s=2.0, carry_sigma=0.8, entry_missing=0.24, lead_median_s=2.7, lead_sigma=0.75)
V2_TACTICAL = dict(V2, v2_alive_missing=0.12, restart_missing=0.30, entry_missing=0.30)
# noise_model "smooth" (V3): the default i.i.d. per-frame position noise (players 0.15 m, ball 0.35 m) turns
# into ~2.6 / ~6 m/s of velocity noise after differencing -- degraded DFL had median player speed 3.8 m/s
# (clean DFL 1.9-2.1, BAR-ATH ATD 1.95 in play) and a perfectly separable domain (src/exp_advval.py). BAR-ATH
# tracks are smooth (median |2nd difference| 0.002 m), so V3 low-passes the same random draws over
# p_noise_s / ball_noise_s seconds: same masks and random stream as V2, positional error kept, velocity
# error ~0.1-0.5 m/s.
V3 = dict(V2, noise_model="smooth", p_noise_s=1.0, ball_noise_s=0.5)
# V4 = V3 on pre-smoothed source tracks: open DFL is ~5x rougher frame to frame than BAR-ATH's filtered ATD
# (median |2nd difference| in play: players 0.0105 vs 0.0020 m, ball 0.06 vs 0.011 m; SkillCorner and Metrica are
# as smooth as BAR-ATH). Gaussian low-pass of the clean positions within finite runs (players 1.5 frames, ball
# 4 frames) before the degradation; no random draws, so masks / noise equal V3's.
V4 = dict(V3, presmooth_p=1.5, presmooth_b=4.0)
# ball_every k (src/exp_ballrate.py, bench variants "v3b<k>"): the ball is observed only every k-th frame -- a ball
# detector run at a stride -- and linearly interpolated between two consecutive observed samples; no random draws,
# so the masks and noise equal the base variant's.
# V5 (src/exp_fab.py; research a01 / a04 / a18): BAR-ATH's tactical ATD fills part of the ball track by prediction --
# over-smooth runs, 3.1% of the seen frames, ~1.7 m off (src/ballclean.py). V5 = V3 plus short seen live-play
# segments (lognormal, median fab_med frames, fab_min..fab_max) replaced by a straight line between their ends -- a
# tracker that briefly lost the ball and predicted it -- until fab_share of the live frames are covered (own random
# stream, so V3's masks / noise are unchanged). clean_fab: afterwards drop the over-smooth runs again
# (ballclean.fabricated), as model_v7's clean_ball does on BAR-ATH (there is no broadcast ball to substitute).
V5 = dict(V3, fab_share=0.03, fab_med=20, fab_sigma=0.6, fab_min=8, fab_max=75)
V5C = dict(V5, clean_fab=True)
V3C = dict(V3, clean_fab=True)
# V6 (round 49, src/exp_edges.py): V2 / V3 cut every dead-ball gap exactly at the restart, so the ball reappears on the true restart frame at about half of the
# restarts (DFL 0.55, Metrica 0.72); raw PFF tracking does so at 0.03, with the ball visible through the last 2 s before 0.65 of the restarts. V6 replaces the
# first and last seconds of each dead stretch by a draw of three states: visible on, gone, or mixed. Shares: label-free statistics of the target's ball around
# the model's own stops and restarts, whole match, no public window (all restarts: visible 0.427, mixed 0.126; all stops: visible 0.385, mixed 0.24), divided
# by the share of edges where the ball is seen on the adjacent live frame. Own random stream; the masks in live play are V3's except for the +-edge_jitter frames
# around a restart. Where the source has no ball on a revealed frame (Metrica's dead ball) the nearest live position is held.
V6 = dict(V3, edge_model=True, after_vis=0.507, after_mix=0.316, before_vis=0.547, before_mix=0.16, edge_vis_median_s=4.0, edge_sigma=0.8,
          edge_gone_median_s=10.7, edge_gone_sigma=0.9, edge_mix_frames=50, edge_jitter=6)


def _smooth_runs(x, sigma):
    """Gaussian low-pass of a 1-D track within its finite runs (NaN gaps stay)."""
    from scipy.ndimage import gaussian_filter1d
    out = np.array(x, np.float64, copy=True)
    ok = np.isfinite(out)
    e = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a > 3:
            out[a:b] = gaussian_filter1d(out[a:b], sigma, mode="nearest")
    return out


def _smooth_unit(z, sigma_frames):
    """Gaussian low-pass of i.i.d. draws along time, rescaled to unit standard deviation."""
    from scipy.ndimage import gaussian_filter1d
    y = gaussian_filter1d(np.asarray(z, np.float64), sigma_frames, axis=0, mode="nearest")
    return y / max(float(y.std()), 1e-12)


def _ball_every(Bn, k):
    """Keep the ball on frames t % k == 0 only; linear in between where both neighbouring samples are seen."""
    t = np.arange(len(Bn))
    idx = t[::k]
    keep = idx[np.isfinite(Bn[idx, 0])]
    out = np.full_like(Bn, np.nan)
    if len(keep) < 2:
        out[keep] = Bn[keep]
        return out
    pos = np.searchsorted(keep, t, side="right")
    left = keep[np.clip(pos - 1, 0, len(keep) - 1)]
    right = keep[np.clip(pos, 0, len(keep) - 1)]
    ok = ((pos > 0) & (pos < len(keep)) & (right - left <= k)) | np.isin(t, keep)
    for c in range(Bn.shape[1]):
        out[:, c] = np.interp(t, keep, Bn[keep, c])
    out[~ok] = np.nan
    return out


def _fabricate(Bn, lab, p, rng):
    """Replace short seen live-play segments by a straight line between their ends (a tracker predicting a briefly
    lost ball) until fab_share of the live frames are covered."""
    out = Bn.copy()
    n = len(Bn)
    ok = np.isfinite(Bn[:, 0]) & (lab < 2)
    cand = np.flatnonzero(ok)
    target = p["fab_share"] * int((lab < 2).sum())
    done, tries = 0, 0
    used = np.zeros(n, bool)
    while done < target and tries < 20 * max(1, int(target / max(p["fab_med"], 1))) and len(cand):
        tries += 1
        a = int(cand[rng.integers(len(cand))])
        ln = int(np.clip(round(rng.lognormal(np.log(p["fab_med"]), p["fab_sigma"])), p["fab_min"], p["fab_max"]))
        b = a + ln
        if a < 1 or b >= n or not ok[a - 1:b + 1].all() or used[a - 1:b + 1].any():
            continue
        w = np.arange(1, ln + 1) / float(ln + 1)
        for c in range(Bn.shape[1]):
            if np.isfinite(Bn[a - 1, c]) and np.isfinite(Bn[b, c]):
                out[a:b, c] = Bn[a - 1, c] + w * (Bn[b, c] - Bn[a - 1, c])
        used[a:b] = True
        done += ln
    return out


def _lognormal_frames(rng, median_s, sigma, size=None):
    return np.maximum(1, (rng.lognormal(np.log(median_s * FPS), sigma, size))).astype(int)


def _renewal(hazard, gap_median_s, gap_sigma, rng):
    n = len(hazard)
    seen = np.ones(n, bool)
    u = rng.random(n)
    t = 0
    while t < n:
        if u[t] < hazard[t]:
            g = _lognormal_frames(rng, gap_median_s, gap_sigma)
            seen[t:t + g] = False
            t += g
        else:
            t += 1
    return seen


def ball_dropout(lab, z, rng, p):
    """'seen' mask with state-specific missing rates (calibrated to BAR-ATH: ~12% missing in
    play, ~70% while dead), gaps more likely for aerial balls; hazards are scaled iteratively
    so the realised rates hit the targets."""
    dead = lab == 2
    alive = ~dead
    air = np.nan_to_num(z) > p["air_z"]
    h_a, h_d = 1.0 / (40 * FPS), 1.0 / (6 * FPS)
    for _ in range(4):
        # two renewal draws: dead-state gaps and alive-state gaps
        s_d = _renewal(np.where(dead, h_d, 0.0), p["miss_median_s"], p["miss_sigma"], rng)
        s_a = _renewal(np.where(alive, h_a * np.where(air, p["air_mult"], 1.0), 0.0),
                       p["alive_gap_median_s"], p["miss_sigma"], rng)
        seen = s_d & s_a
        ma = 1 - seen[alive].mean() if alive.any() else p["alive_missing"]
        md = 1 - seen[dead].mean() if dead.any() else p["dead_missing"]
        h_a *= np.clip(p["alive_missing"] / max(ma, 1e-3), 0.3, 3.0)
        h_d *= np.clip(p["dead_missing"] / max(md, 1e-3), 0.3, 3.0)
    return seen


def _gap_union(starts, glen, n, stop=None):
    """Missing mask: union of the gaps [t, t + glen[t]) over the start frames t, each cut at stop[t]."""
    starts = np.asarray(starts, int)
    e = starts + glen[starts]
    if stop is not None:
        e = np.minimum(e, stop[starts])
    cover = np.zeros(n + 1, np.int64)
    np.add.at(cover, starts, 1)
    np.add.at(cover, np.minimum(e, n), -1)
    return np.cumsum(cover[:n]) > 0


def _fit_rate(miss_at, target, lo=1e-7, hi=1.0, iters=36):
    """Hazard in [lo, hi] whose realised missing share miss_at(h) (non-decreasing in h, common random
    numbers) is closest to target; bisection on log h."""
    best = None
    for _ in range(iters):
        h = np.sqrt(lo * hi)
        m = miss_at(h)
        if best is None or abs(m - target) < abs(best[1] - target):
            best = (h, m)
        lo, hi = (h, hi) if m < target else (lo, h)
    return best[0]


def _pick(already, u, target):
    """Events to make missing: besides those already missing, the eligible ones (u <= 1) with the
    smallest u until a share target of all events is missing."""
    sel = np.zeros(len(already), bool)
    k = int(round(target * len(already))) - int(already.sum())
    cand = np.flatnonzero(~already & (u <= 1.0))
    if k > 0 and len(cand):
        sel[cand[np.argsort(u[cand], kind="stable")[:k]]] = True
    return sel


def ball_dropout_v2(lab, z, rng, p, present=None):
    """'seen' mask built around the stoppages, as in BAR-ATH: in ball_dropout, gaps that start while
    the ball is dead ran on for their full ~11 s median and covered ~20-25% of the alive frames (the
    hazard loop could only shrink alive-started gaps). Here
      - dead gaps (lognormal, miss_median_s) are cut at the restart;
      - after a share restart_missing of restarts the ball stays missing for a short lognormal time
        (carry_median_s; only where it was missing on the last dead frame);
      - before a share entry_missing of stoppages the ball is gone already (lead_median_s), and the
        dead gap then starts with the stoppage;
      - random gaps in play (alive_gap_median_s, air_mult for aerial balls) fill up v2_alive_missing.
    Rates are fitted per half with common random numbers (monotone in each parameter), so the realised
    shares, the source's own gaps (`present`) included, hit the targets."""
    n = len(lab)
    dead = lab == 2
    alive = ~dead
    absent = np.zeros(n, bool) if present is None else ~np.asarray(present, bool)
    air = np.nan_to_num(z) > p["air_z"]
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    d_runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    stop = np.full(n, n)        # dead frame -> its restart
    nxt = np.full(n, n)         # alive frame -> start of the next stoppage
    prv = np.zeros(n, int)      # alive frame -> end of the previous stoppage
    last = 0
    for a, b in d_runs:
        stop[a:b] = b
        nxt[last:a] = a
        prv[last:a] = last
        last = b
    prv[last:] = last
    restarts = np.array([b for _, b in d_runs if b < n], int)
    entries = np.array([a for a, _ in d_runs if a > 0], int)
    u_d, g_d = rng.random(n), _lognormal_frames(rng, p["miss_median_s"], p["miss_sigma"], n)
    u_a, g_a = rng.random(n), _lognormal_frames(rng, p["alive_gap_median_s"], p["miss_sigma"], n)
    u_c, g_c = rng.random(len(restarts)), _lognormal_frames(rng, p["carry_median_s"], p["carry_sigma"], len(restarts))
    u_l, g_l = rng.random(len(entries)), _lognormal_frames(rng, p["lead_median_s"], p["lead_sigma"], len(entries))
    w_a = np.where(alive, np.where(air, p["air_mult"], 1.0), 0.0)
    miss_a = lead = carry = miss_d = np.zeros(n, bool)
    for _ in range(3):
        in_play = miss_a | carry | absent
        led = _pick(in_play[entries - 1], u_l, p["entry_missing"])
        lead = np.zeros(n, bool)
        for a, ln in zip(entries[led], g_l[led]):
            lead[max(a - ln, prv[a - 1]):a] = True
        forced = entries[led]

        def dead_gaps(h):
            return _gap_union(np.union1d(np.flatnonzero(dead & (u_d < h)), forced), g_d, n, stop)
        other = miss_a | absent
        h_d = _fit_rate(lambda h: (dead_gaps(h) | other)[dead].mean(), p["v2_dead_missing"])
        miss_d = dead_gaps(h_d)
        pre = (miss_d | miss_a | absent)[restarts - 1]
        car = _pick((miss_a | absent)[restarts], np.where(pre, u_c, 2.0), p["restart_missing"])
        carry = np.zeros(n, bool)
        for r, c in zip(restarts[car], g_c[car]):
            carry[r:min(r + c, nxt[r])] = True
        fixed = lead | carry | absent

        def play_gaps(h):
            return _gap_union(np.flatnonzero(u_a < h * w_a), g_a, n)
        h_a = _fit_rate(lambda h: (play_gaps(h) | fixed)[alive].mean(), p["v2_alive_missing"])
        miss_a = play_gaps(h_a)
    return ~(miss_a | miss_d | lead | carry)


def project_aerial(bxy_norm, z, pitch, cam):
    """Ground-plane projection of a ball at height z as seen from a camera at cam (normalised
    x, y, height in metres): displaced away from the camera by z / (H - z)."""
    L, W = pitch
    cx, cy, H = cam[0] * L, cam[1] * W, cam[2]
    bx, by = bxy_norm[:, 0] * L, bxy_norm[:, 1] * W
    zz = np.clip(np.nan_to_num(z), 0, H * 0.8)
    k = zz / (H - zz)
    gx = bx + (bx - cx) * k
    gy = by + (by - cy) * k
    return np.stack([gx / L, gy / W], 1)


def player_dropout(P, rng, p):
    n, s, _ = P.shape
    out = P.copy()
    for j in range(s):
        alive = np.isfinite(P[:, j, 0])
        t = 0
        while t < n:
            run = _lognormal_frames(rng, p["p_present_median_s"], p["p_sigma"])
            t += run
            gap = _lognormal_frames(rng, p["p_missing_median_s"], p["p_sigma"])
            out[t:t + gap, j, :] = np.nan
            t += gap
        out[~alive, j, :] = np.nan
    return out


def _edges(seen, finite, lab, rng, p):
    """V6: the ball around each true dead stretch [a, b). Returns the new mask and fill: for a revealed frame without a source position the frame whose position is held, else -1."""
    n = len(lab)
    seen = seen.copy()
    fill = np.full(n, -1, np.int64)
    dead = lab == 2
    e = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        m = b - a
        ua, ub = rng.random(), rng.random()
        vis_a, vis_b = _lognormal_frames(rng, p["edge_vis_median_s"], p["edge_sigma"]), _lognormal_frames(rng, p["edge_vis_median_s"], p["edge_sigma"])
        gone_a, gone_b = _lognormal_frames(rng, p["edge_gone_median_s"], p["edge_gone_sigma"]), _lognormal_frames(rng, p["edge_gone_median_s"], p["edge_gone_sigma"])
        mix_a, mix_b = int(rng.integers(1, p["edge_mix_frames"])), int(rng.integers(1, p["edge_mix_frames"]))
        jit = int(rng.integers(-p["edge_jitter"], p["edge_jitter"] + 1))
        if a > 0 and seen[a - 1]:
            src = a - 1
            if ua < p["after_vis"]:
                e1 = min(a + max(vis_a, 50), b)
                seen[a:e1] = True
                nofin = np.flatnonzero(~finite[a:e1]) + a
                fill[nofin] = src
                if e1 < b:
                    seen[e1:min(e1 + max(gone_a, 50), b)] = False
            elif ua < p["after_vis"] + p["after_mix"]:
                e1 = min(a + mix_a, b)
                seen[a:e1] = True
                nofin = np.flatnonzero(~finite[a:e1]) + a
                fill[nofin] = src
                seen[e1:min(e1 + max(gone_a, 50), b)] = False
            else:
                seen[a:min(a + max(gone_a, 50), b)] = False
        if b < n and seen[b]:          # not one of the restarts where the ball stays missing
            nxt = b
            while nxt < n and not finite[nxt]:
                nxt += 1
            if nxt < n:
                if ub < p["before_vis"]:
                    s1 = max(b - max(vis_b, 50), a)
                    seen[s1:b] = True
                    nofin = np.flatnonzero(~finite[s1:b]) + s1
                    fill[nofin] = nxt
                    if s1 > a:
                        seen[max(s1 - max(gone_b, 50), a):s1] = False
                elif ub < p["before_vis"] + p["before_mix"]:
                    s1 = max(b - mix_b, a)
                    seen[s1:b] = True
                    nofin = np.flatnonzero(~finite[s1:b]) + s1
                    fill[nofin] = nxt
                    seen[max(s1 - max(gone_b, 50), a):s1] = False
                else:
                    seen[max(b - max(gone_b, 50), a):b] = False
                    if jit < 0:             # the tracker picks the ball up just before the kick
                        s1 = max(b + jit, a)
                        seen[s1:b] = True
                        nofin = np.flatnonzero(~finite[s1:b]) + s1
                        fill[nofin] = nxt
                    elif jit > 0:           # or just after it
                        stop = b
                        while stop < min(n, b + jit) and not dead[stop]:
                            stop += 1
                        seen[b:stop] = False
    return seen & (finite | (fill >= 0)), fill


def degrade(P, B, lab, pitch, seed=0, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)
    L, W = pitch
    if p.get("presmooth_p") or p.get("presmooth_b"):
        P = P.copy()
        B = B.copy()
        if p.get("presmooth_p"):
            for j in range(P.shape[1]):
                for c in (0, 1):
                    P[:, j, c] = _smooth_runs(P[:, j, c], p["presmooth_p"])
        if p.get("presmooth_b"):
            for c in [0, 1] + ([3] if B.shape[1] > 3 else []):
                B[:, c] = _smooth_runs(B[:, c], p["presmooth_b"])
    Bn = B.copy()
    z = B[:, 3] if B.shape[1] > 3 else np.zeros(len(B))
    Bn[:, :2] = project_aerial(B[:, :2], z, pitch, p["cam"])
    smooth = p.get("noise_model", "iid") == "smooth"
    if smooth:
        s_b = p["ball_noise_s"] * FPS
        Bn[:, 0] += _smooth_unit(rng.normal(0, 1, len(B)), s_b) * p["ball_noise"] / L
        Bn[:, 1] += _smooth_unit(rng.normal(0, 1, len(B)), s_b) * p["ball_noise"] / W
    else:
        Bn[:, 0] += rng.normal(0, p["ball_noise"] / L, len(B))
        Bn[:, 1] += rng.normal(0, p["ball_noise"] / W, len(B))
    seen = ball_dropout(lab, z, rng, p) & np.isfinite(B[:, 0])
    if p.get("ball_model", "v1") == "v2":
        # the v1 draw above still runs, so the player dropout and noise below get exactly the v1
        # random stream (paired v1/v2 comparisons); the v2 ball mask has a stream of its own
        present = np.isfinite(B[:, 0])
        seen = ball_dropout_v2(lab, z, np.random.default_rng([seed, 2]), p, present) & present
    if p.get("edge_model"):
        seen, fill = _edges(seen, np.isfinite(Bn[:, 0]), np.asarray(lab), np.random.default_rng([seed, 6]), p)
        held = fill >= 0
        Bn[held] = Bn[fill[held]]
    Bn[~seen, :] = np.nan
    if p.get("fab_share", 0) > 0:
        Bn = _fabricate(Bn, lab, p, np.random.default_rng([seed, 3]))
    if p.get("clean_fab"):
        import ballclean
        Bn[ballclean.fabricated(Bn[:, :2] * np.array([L, W]))] = np.nan
    if p.get("ball_every", 1) > 1:
        Bn = _ball_every(Bn, int(p["ball_every"]))
    Pn = player_dropout(P, rng, p)
    if smooth:
        s_p = p["p_noise_s"] * FPS
        Pn[:, :, 0] += (_smooth_unit(rng.normal(0, 1, Pn.shape[:2]), s_p) * p["p_noise"] / L).astype(np.float32)
        Pn[:, :, 1] += (_smooth_unit(rng.normal(0, 1, Pn.shape[:2]), s_p) * p["p_noise"] / W).astype(np.float32)
    else:
        Pn[:, :, 0] += rng.normal(0, p["p_noise"] / L, Pn.shape[:2]).astype(np.float32)
        Pn[:, :, 1] += rng.normal(0, p["p_noise"] / W, Pn.shape[:2]).astype(np.float32)
    return Pn, Bn
