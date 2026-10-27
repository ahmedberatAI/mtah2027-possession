"""Ball-touch events per team (idea C7, docs/RAPOR.md), as a bench feature plugin.

The remaining main error everywhere is a short spell after a turnover given to the wrong team (SkillCorner
with true dead balls: 20% of turnover spells). Nearest-player cues confuse the player in control with the
presser. A touch changes the ball's velocity: touches are local maxima of |v(t + k) - v(t - k)| above a
threshold with a player within reach; the touching team is the team of the nearest player. Possession in
the "controlled" sense is a run of touches by one team (several touches / a pass between teammates), so
per team:
  since / until   frames since the team's last touch and until its next one (capped)
  last            +1 if the most recent touch was this team's, -1 if the opponent's, 0 if none yet
  streak          consecutive touches by this team since the opponent's last touch (and the next run)
  passes          teammate-to-teammate touch pairs (different players, < 4 s apart) in +-3 s / +-6 s
  touches         touch counts in the last / next 3 s
Needs realistic ball kinematics (sim V3; V2's per-frame noise makes every frame look like a touch); the
detection threshold adapts so that every half yields the same touch rate (see RATE).

  bench.run("base+...+feat_touch:touch", "native", var="v3")
"""
import numpy as np

from mlfeat import CLIP, since_last, until_next, velocity, win_mean

K = 3            # velocity change measured over +-K frames
DV = 1.0         # m/s: minimum change for a candidate
RATE = 10.0      # touches kept per minute of the half: the strongest velocity changes. A fixed m/s threshold
                 # is domain dependent -- at 3 m/s DFL gives 38 touches per alive minute, SkillCorner 24 and the
                 # (heavily filtered) BAR-ATH ball 14 -- so the threshold adapts to a common touch rate.
REACH = 2.0      # m, a player this close to the ball can have touched it
NMS = 6          # frames of non-maximum suppression


def touches(P, B, team, pitch):
    """(frame, slot, team) of detected touches in one half."""
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = B[:, :2].astype(np.float32) * np.array([L, W], np.float32)
    n = len(bxy)
    v = velocity(bxy)
    dv = np.full(n, np.nan, np.float32)
    dv[K:-K] = np.linalg.norm(v[2 * K:] - v[:-2 * K], axis=1)
    dv = np.nan_to_num(dv, nan=0.0)
    d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, 1)
    near = d[np.arange(n), j]
    cand = (dv > DV) & (near < REACH)
    # local maxima of dv among candidates
    pad = np.concatenate([np.zeros(NMS), dv, np.zeros(NMS)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * NMS + 1)
    peak = cand & (dv >= win.max(1))
    t = np.flatnonzero(peak)
    keep = int(round(RATE * n / (60 * 25)))
    if len(t) > keep:
        t = np.sort(t[np.argsort(-dv[t], kind="stable")[:keep]])
    return t, j[t], team[j[t]]


def touch(P, B, team, gk, pitch):
    n = len(P)
    t, slot, tt = touches(P, B, team, pitch)
    out = {}
    for k in (0, 1):
        own = np.zeros(n, bool)
        opp = np.zeros(n, bool)
        own[t[tt == k]] = True
        opp[t[tt != k]] = True
        f = {}
        f["since_own"] = since_last(own)
        f["since_opp"] = since_last(opp)
        f["until_own"] = until_next(own)
        f["until_opp"] = until_next(opp)
        f["last"] = np.where(f["since_own"] < f["since_opp"], 1.0,
                             np.where(f["since_opp"] < f["since_own"], -1.0, 0.0)).astype(np.float32)
        f["next"] = np.where(f["until_own"] < f["until_opp"], 1.0,
                             np.where(f["until_opp"] < f["until_own"], -1.0, 0.0)).astype(np.float32)
        # streaks: own touches since the last opponent touch (backward) / before the next one (forward)
        streak_b = np.zeros(n, np.float32)
        streak_f = np.zeros(n, np.float32)
        run = 0
        for i, (tk, kk) in enumerate(zip(t, tt)):
            run = run + 1 if kk == k else 0
            streak_b[tk] = run
        streak_b = _carry(own | opp, streak_b)
        run = 0
        for tk, kk in zip(t[::-1], tt[::-1]):
            run = run + 1 if kk == k else 0
            streak_f[tk] = run
        streak_f = _carry(own[::-1] | opp[::-1], streak_f[::-1])[::-1]
        f["streak_bwd"] = streak_b
        f["streak_fwd"] = streak_f
        # passes: consecutive touches by two different players of this team < 100 frames apart
        pas = np.zeros(n, np.float32)
        for a in range(1, len(t)):
            if tt[a] == k and tt[a - 1] == k and slot[a] != slot[a - 1] and t[a] - t[a - 1] < 100:
                pas[t[a]] = 1.0
        f["passes_c75"] = win_mean(pas, -75, 75) * 151
        f["passes_c150"] = win_mean(pas, -150, 150) * 301
        tc = own.astype(np.float32)
        f["touch_bwd75"] = win_mean(tc, -75, 0) * 76
        f["touch_fwd75"] = win_mean(tc, 0, 75) * 76
        out[k] = {kk: np.clip(np.asarray(v, np.float32), -CLIP, CLIP) for kk, v in f.items()}
    T = {}
    for own_k, opp_k in ((0, 1), (1, 0)):
        T[own_k] = {}
        for name in out[own_k]:
            T[own_k]["tch_own_" + name] = out[own_k][name]
            T[own_k]["tch_opp_" + name] = out[opp_k][name]
    F = {"tch_rate_c250": win_mean(np.isin(np.arange(n), t).astype(np.float32), -250, 250) * 25}
    return F, T[0], T[1]


def _carry(event, value):
    """Value at the most recent event frame, carried forward (0 before the first event)."""
    idx = np.where(event, np.arange(len(event)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, value[np.maximum(last, 0)], 0.0).astype(np.float32)
