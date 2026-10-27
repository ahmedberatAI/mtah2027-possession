"""Round 11, idea E6 (docs/RAPOR.md): does the broadcast camera's framing reveal the team in possession? TV camera
operators leave "lead room" in the direction of play, so the point the camera aims at (image centre mapped to the
pitch through the broadcast homography) should sit ahead of the ball towards the goal the possessing team attacks.
Label-free check against v10's own confident labels; ball velocity is controlled for (a lagging camera puts the
ball ahead of the centre whenever the ball moves). AGGREGATE output only.

    python -I src/exp_framing.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import commentary as cm  # noqa: E402
import evaluate as ev  # noqa: E402

FPS = 25
L, W = 105.3, 68.0


def auc(score, y):
    """P(score of a positive > score of a negative), ties half."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def main():
    import pandas as pd
    hom = pd.read_csv(atd.paths("broadcast")["homography"])
    Hm = hom[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].values.reshape(-1, 3, 3)
    valid = hom["valid"].values.astype(bool)
    nb = len(valid)
    aim = np.full((nb, 2), np.nan)
    c = np.array([960.0, 540.0, 1.0])
    idx = np.flatnonzero(valid)
    inv = np.linalg.inv(Hm[idx])
    q = inv @ c
    aim[idx] = q[:, :2] / q[:, 2:3]
    bro = atd.load("broadcast")
    ball_b = bro["ball"][:nb, :2].astype(float)
    # tactical timeline
    tac = atd.load("tactical")
    team, slots = tac["team"], tac["slots"]
    gk = slots["position_type"].values == "Goalkeeper" if "position_type" in slots else np.zeros(len(team), bool)
    codes, ph = cm.v10_state()
    n = len(codes)
    sgn = np.zeros(n)
    for _, a, b in atd.HALVES:
        x = tac["players"][a:b + 1, :, 0]
        hg = (team == 0) & gk
        gx = np.nanmean(x[:, hg]) if hg.any() else np.nanmean(x[:, team == 0])
        sgn[a:b + 1] = 1.0 if gx < 0.5 else -1.0     # +1: home attacks towards x = 1
    t_of_b = np.where(np.arange(nb) <= cm.CUT_B, np.arange(nb) + 4014, np.arange(nb) + 26920)
    ok = valid & np.isfinite(ball_b[:, 0]) & (t_of_b < n)
    b_idx = np.flatnonzero(ok)
    t = t_of_b[b_idx]
    s = sgn[t]
    dx = (aim[b_idx, 0] - ball_b[b_idx, 0]) * L * s            # + : camera aims ahead of the ball, home direction
    # ball velocity along the home attacking direction (m/s), +-12 broadcast frames
    bx = ball_b[:, 0] * L
    lo, hi = np.clip(b_idx - 12, 0, nb - 1), np.clip(b_idx + 12, 0, nb - 1)
    vx = (bx[hi] - bx[lo]) / ((hi - lo) / FPS) * s
    pan = np.full(len(b_idx), np.nan)
    ax = aim[:, 0] * L
    pan = (ax[hi] - ax[lo]) / ((hi - lo) / FPS) * s
    spells = cm.spells(codes)
    team_at = np.full(n, -1, np.int8)
    conf = np.zeros(n, bool)
    for a, e, k in spells:
        team_at[a:e] = k
        if e - a >= 6 * FPS:
            conf[a + FPS:e - FPS] = True
    conf &= np.abs(np.nan_to_num(ph, nan=0.5) - 0.5) > 0.4
    graded = atd.graded_mask(n)
    m = conf[t] & graded[t] & np.isfinite(dx) & np.isfinite(vx)
    y = team_at[t[m]] == 0
    print("broadcast frames with a valid view + ball, mapped to v10-confident live frames: %d (%.0f%% home)" % (
        m.sum(), 100 * y.mean()))
    print("AUC(home) of lead room dx %.3f | of camera pan velocity %.3f | of ball velocity %.3f" % (
        auc(dx[m], y), auc(pan[m], y), auc(vx[m], y)))
    print("median lead room dx (m) when v10 says home %.2f / away %.2f" % (np.median(dx[m][y]), np.median(dx[m][~y])))
    for lo_v, hi_v in ((-0.5, 0.5), (-2, -0.5), (0.5, 2), (-99, -2), (2, 99)):
        mm = (vx[m] >= lo_v) & (vx[m] < hi_v)
        print("  ball velocity in [%g, %g) m/s: n %d, AUC(dx) %.3f, home share %.2f" % (
            lo_v, hi_v, mm.sum(), auc(dx[m][mm], y[mm]), y[mm].mean()))
    # field-position control: ball x along the home attacking direction (0 = home goal, 1 = away goal)
    bxn = np.where(s > 0, ball_b[b_idx, 0], 1 - ball_b[b_idx, 0])
    for lo_x, hi_x in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        mm = (bxn[m] >= lo_x) & (bxn[m] < hi_x)
        print("  ball x (home direction) in [%.2f, %.2f): n %d, AUC(dx) %.3f, home share %.2f, median dx home %.1f "
              "away %.1f" % (lo_x, hi_x, mm.sum(), auc(dx[m][mm], y[mm]), y[mm].mean(),
                            np.median(dx[m][mm][y[mm]]), np.median(dx[m][mm][~y[mm]])))
    for h, (_, a, b) in enumerate(atd.HALVES):
        mm = (t[m] >= a) & (t[m] <= b)
        print("  half %d: n %d AUC(dx) %.3f" % (h + 1, mm.sum(), auc(dx[m][mm], y[mm])))
    # coverage and strength where v10 is unsure
    alive = np.isin(codes, (ev.pm.H, ev.pm.A)) & graded
    cov = np.zeros(n, bool)
    cov[t[np.isfinite(dx)]] = True
    print("coverage: valid view + ball on %.2f of v10-alive frames" % cov[alive].mean())
    pu = ph[t]
    unsure = (pu > 0.2) & (pu < 0.8) & alive[t] & np.isfinite(dx)
    print("v10-unsure frames (0.2 < p < 0.8) with the cue: %d; |dx| > 6 m on %.2f of them (confident frames %.2f)" % (
        unsure.sum(), (np.abs(dx[unsure]) > 6).mean(), (np.abs(dx[m]) > 6).mean()))
    # best shift between the camera's direction and v10's labels
    tl = team_at[np.clip(t, 0, n - 1)]
    for sh in (-50, -37, -25, -12, 0, 12, 25):
        tt = np.clip(t + sh, 0, n - 1)
        k = (team_at[tt] >= 0) & np.isfinite(dx) & (np.abs(dx) > 3) & graded[tt]
        print("  shift %+d frames (label at t%+d): sign agreement %.3f (n %d)" % (
            sh, sh, ((dx[k] > 0) == (team_at[tt][k] == 0)).mean(), k.sum()), end="\n" if sh == 25 else ";")
    # strong disagreement runs inside v10 spells: camera says the other team (|dx| > 6 m) for >= 1 s
    dis = np.zeros(n, bool)
    strong = np.isfinite(dx) & (np.abs(dx) > 6)
    other = (dx > 0) != (tl == 0)
    dis[t[strong & other & (tl >= 0)]] = True
    e = np.diff(np.concatenate(([0], dis.astype(np.int8), [0])))
    runs = [(a0, b0) for a0, b0 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    lens = np.array([b0 - a0 for a0, b0 in runs]) / FPS
    live_min = alive.sum() / FPS / 60
    for lo_s, hi_s in ((1, 2), (2, 4), (4, 8), (8, 999)):
        sel = (lens >= lo_s) & (lens < hi_s)
        print("  disagreement runs %g-%g s: %d (%.2f / live min)" % (lo_s, hi_s, sel.sum(), sel.sum() / live_min))
    # around v10's live changes: sign-aligned lead room (+ = towards the NEW team's attacking goal)
    pos = {tt: i for i, tt in enumerate(t)}
    curve = {k: [] for k in range(-75, 76, 25)}
    nchg = 0
    for (s0, e0, k0), (s1, e1, k1) in zip(spells[:-1], spells[1:]):
        if e0 != s1 or k0 == k1 or e0 - s0 < 2 * FPS or e1 - s1 < 2 * FPS:
            continue
        nchg += 1
        sign_new = 1.0 if k1 == 0 else -1.0
        for k in curve:
            i = pos.get(s1 + k)
            if i is not None and np.isfinite(dx[i]):
                curve[k].append(dx[i] * sign_new)
    print("v10 changes (both spells >= 2 s): %d; mean lead room towards the NEW team's goal (m) at offset:" % nchg)
    print("  " + " | ".join("%+.0fs %.2f (n %d)" % (k / FPS, np.mean(v), len(v)) for k, v in curve.items() if v))


def camera_cue():
    """Per tactical frame: lead room dx (m, + = camera aims ahead of the ball towards the goal the HOME team
    attacks), ball x along the home direction, NaN where the broadcast has no valid view or no ball."""
    import pandas as pd
    hom = pd.read_csv(atd.paths("broadcast")["homography"])
    Hm = hom[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].values.reshape(-1, 3, 3)
    valid = hom["valid"].values.astype(bool)
    nb = len(valid)
    aim = np.full((nb, 2), np.nan)
    idx = np.flatnonzero(valid)
    q = np.linalg.inv(Hm[idx]) @ np.array([960.0, 540.0, 1.0])
    aim[idx] = q[:, :2] / q[:, 2:3]
    ball_b = atd.load("broadcast")["ball"][:nb, :2].astype(float)
    tac = atd.load("tactical")
    team, slots = tac["team"], tac["slots"]
    gk = slots["position_type"].values == "Goalkeeper"
    n = atd.N_FRAMES["tactical"]
    sgn = np.zeros(n)
    for _, a, b in atd.HALVES:
        x = tac["players"][a:b + 1, :, 0]
        hg = (team == 0) & gk
        sgn[a:b + 1] = 1.0 if np.nanmean(x[:, hg]) < 0.5 else -1.0
    t_of_b = np.where(np.arange(nb) <= cm.CUT_B, np.arange(nb) + 4014, np.arange(nb) + 26920)
    ok = valid & np.isfinite(ball_b[:, 0]) & (t_of_b < n)
    dx, bxn = np.full(n, np.nan), np.full(n, np.nan)
    t = t_of_b[ok]
    dx[t] = (aim[ok, 0] - ball_b[ok, 0]) * L * sgn[t]
    bxn[t] = np.where(sgn[t] > 0, ball_b[ok, 0], 1 - ball_b[ok, 0])
    return dx, bxn, sgn


def timing():
    """Event-triggered averages around v10's live changes: camera lead room towards the new team, tracking
    proximity (old team's nearest distance - new team's), v10's smoothed probability of the new team."""
    import pickle
    dx, _, _ = camera_cue()
    codes, ph = cm.v10_state()
    with open(os.path.join(os.path.dirname(HERE), "cache", "features_merged.pkl"), "rb") as fh:
        f = pickle.load(fh)
    dh, da = np.asarray(f["dh"], float), np.asarray(f["da"], float)
    sp = cm.spells(codes)
    offs = np.arange(-100, 101, 5)
    acc = {k: {o: [] for o in offs} for k in ("cam", "prox", "p")}
    n_chg = 0
    for (s0, e0, k0), (s1, e1, k1) in zip(sp[:-1], sp[1:]):
        if e0 != s1 or k0 == k1 or e0 - s0 < 2 * FPS or e1 - s1 < 2 * FPS:
            continue
        n_chg += 1
        sg = 1.0 if k1 == 0 else -1.0
        for o in offs:
            i = s1 + o
            if np.isfinite(dx[i]):
                acc["cam"][o].append(dx[i] * sg)
            d_new, d_old = (dh[i], da[i]) if k1 == 0 else (da[i], dh[i])
            if np.isfinite(d_new) and np.isfinite(d_old):
                acc["prox"][o].append(np.clip(d_old - d_new, -20, 20))
            pn = ph[i] if k1 == 0 else 1 - ph[i]
            if np.isfinite(pn):
                acc["p"][o].append(pn)
    print("v10 live changes (both spells >= 2 s): %d; means by offset from v10's change (s):" % n_chg)
    for k, name in (("cam", "camera lead room towards new team (m)"), ("prox", "old - new nearest distance (m)"),
                    ("p", "v10 smoothed p(new team)")):
        print("  %s:" % name)
        print("   " + " ".join("%+.1f:%.2f" % (o / FPS, np.mean(v)) for o, v in acc[k].items() if v and o % 10 == 0))
    # where does the label at t+s agree best with the camera at t?
    team_at = np.full(len(codes), -1, np.int8)
    for a, e, k in sp:
        team_at[a:e] = k
    best = []
    for s in range(-50, 151, 10):
        tt = np.arange(len(codes))
        src = np.isfinite(dx) & (np.abs(dx) > 3)
        g = tt[src] + s
        g = g[(g >= 0) & (g < len(codes))]
        srcs = tt[src][:len(g)]
        k = team_at[g] >= 0
        best.append((s, ((dx[srcs][k] > 0) == (team_at[g][k] == 0)).mean()))
    print("  camera at t vs v10 label at t+s: " + " ".join("%+d:%.3f" % b for b in best))


if __name__ == "__main__":
    {"main": main, "timing": timing}[sys.argv[1] if len(sys.argv) > 1 else "main"]()
