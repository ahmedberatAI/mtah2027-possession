"""Duel-outcome features for contested-ball moments (round 10, user idea; src/exp_contest.py): who comes away with the
ball after players of both teams were near it. v10's features look at most 1 s ahead; the outcome of a duel is often
clear only 1-3 s later. Per team view k (against o), per frame t, from the degraded tracking only:

  dd_h      d_o - d_k at t + h: nearest-player distance to the ball (m, clipped to +-10; NaN when the ball is unseen),
            h = -1, 0, +0.5, +1, +2, +3 s
  near_h    sign of dd_h (+1: a k player is nearest), same horizons
  n5_h      k players minus o players within 5 m of the ball at t + h, h = 0, +1, +2 s
  prog_h    ball displacement towards o's goal over [t, t + h] (m), h = +1, +2, +3 s
  ep_*      for the contest episode containing t (both teams <= 3 m of the ball; gaps up to 0.5 s merged): frames since
            its start and until its end (s), near_h 0.5 / 1 / 2 s after its end and 0.5 s before its start, the ball's
            progress towards o's goal from its start to 1 s after its end; NaN outside episodes

duel_feats(P, B, team, gk, pitch) -> (home view, away view), each (n, 25) float32.
"""
import numpy as np

FPS = 25
H_DD = (-25, 0, 12, 25, 50, 75)
H_N5 = (0, 25, 50)
H_PROG = (25, 50, 75)


def _shift(x, h):
    """y[t] = x[t + h] (NaN outside)."""
    y = np.full(len(x), np.nan, np.float32)
    if h >= 0:
        y[:len(x) - h] = x[h:]
    else:
        y[-h:] = x[:h]
    return y


def _episodes(mask, gap=12):
    """[(a, b)] runs of mask with gaps of up to `gap` frames merged."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.concatenate(([idx[0]], idx[br + 1]))
    ends = np.concatenate((idx[br], [idx[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def duel_feats(P, B, team, gk, pitch):
    L, W = pitch
    pxy = np.asarray(P, np.float32)[:, :, :2] * np.array([L, W], np.float32)
    bxy = np.asarray(B, np.float32)[:, :2] * np.array([L, W], np.float32)
    team = np.asarray(team)
    gk = np.asarray(gk, bool)
    n = len(bxy)
    seen = np.isfinite(bxy[:, 0])
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d = [np.min(np.where(team[None, :] == k, dist, np.inf), 1) for k in (0, 1)]
    n5 = [np.sum((team[None, :] == k) & (dist <= 5.0), 1).astype(np.float32) for k in (0, 1)]
    plus = []
    for k in (0, 1):
        m = gk & (team == k)
        with np.errstate(all="ignore"):
            x = np.nanmedian(pxy[:, m, 0]) if m.any() else np.nan
        plus.append(bool(x < L / 2) if np.isfinite(x) else k == 0)
    contest = (d[0] <= 3.0) & (d[1] <= 3.0) & seen
    eps = _episodes(contest)
    views = []
    for k in (0, 1):
        o = 1 - k
        with np.errstate(invalid="ignore"):
            dd = np.where(seen, np.clip(d[o] - d[k], -10, 10), np.nan).astype(np.float32)
        nn = np.where(seen, n5[k] - n5[o], np.nan).astype(np.float32)
        bx = bxy[:, 0] if plus[k] else L - bxy[:, 0]          # towards o's goal
        cols = []
        for h in H_DD:
            cols.append(_shift(dd, h))
        for h in H_DD:
            cols.append(np.sign(cols[H_DD.index(h)]))
        for h in H_N5:
            cols.append(_shift(nn, h))
        for h in H_PROG:
            cols.append(_shift(bx, h) - bx)
        since = np.full(n, np.nan, np.float32)
        until = np.full(n, np.nan, np.float32)
        after = [np.full(n, np.nan, np.float32) for _ in range(3)]
        before = np.full(n, np.nan, np.float32)
        prog = np.full(n, np.nan, np.float32)
        sgn = np.sign(dd)
        for a, b in eps:
            t = np.arange(a, b)
            since[a:b] = (t - a) / FPS
            until[a:b] = (b - t) / FPS
            for j, h in enumerate((12, 25, 50)):
                if b - 1 + h < n:
                    after[j][a:b] = sgn[b - 1 + h]
            if a - 12 >= 0:
                before[a:b] = sgn[a - 12]
            e = min(n - 1, b - 1 + 25)
            prog[a:b] = bx[e] - bx[a]
        cols += [since, until] + after + [before, prog]
        views.append(np.column_stack(cols).astype(np.float32))
    return views[0], views[1]
