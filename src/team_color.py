"""Team labels of the automatic tracking, checked against the shirts in the tactical video (round 84, docs/RAPOR.md).

The ATD gives each track slot one fixed team, but slots "swap identity as the tracker loses and re-acquires people" (official DATA.md), so a slot can follow a player of the other team for a while. This module reads the shirt colour under every tracked object at 5 Hz, models the two kits per half (from the labels themselves, robustly; no human, no ground truth), decodes per slot which team it is following over time, and rewrites the tracking so that the rest of the pipeline stays as it is: the frames of a slot that follow the other team move to a spare slot of that team.

Everything is local and aggregate: nothing here prints a frame, a pixel or a position.

    python -I src/team_color.py extract     # about ten minutes on the CPU
    python -I src/team_color.py fit
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "cache", "team_color")
STEP = 5
N_F = 8                # mean Lab, median Lab, share of non-grass pixels, somebody overlapping the crop
CLIP = 5.0             # the most one colour reading can say, in log-odds
W_OVER = 0.25          # a crop that somebody else may overlap counts this much
S_RUN = 12.0           # cost of changing team inside one continuous track
S_NEW = 2.0            # after a gap of a second or more the slot may be somebody else
PRIOR = 0.05           # per reading, in favour of the slot's own team


def _project(H, x, y):
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w


def _read_frame(cv2, img, H, x, y):
    """(n, N_F) for the objects at normalised x, y; NaN where nothing can be read."""
    n = len(x)
    out = np.full((n, N_F), np.nan, np.float32)
    u, v = _project(H, x, y)
    u1, v1 = _project(H, x + 1 / atd.PITCH[0], y)
    h = 1.7 * np.hypot(u1 - u, v1 - v)
    for i in range(n):
        if not (np.isfinite(u[i]) and np.isfinite(h[i]) and 6 <= h[i] <= 200):
            continue
        x0, x1 = int(u[i] - .24 * h[i]), int(u[i] + .24 * h[i]) + 1
        y0, y1 = int(v[i] - .85 * h[i]), int(v[i] - .42 * h[i]) + 1
        if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0] or x1 - x0 < 2 or y1 - y0 < 2:
            continue
        crop = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grass = (hsv[:, 0] >= 28) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 45)
        if (~grass).sum() < 4:
            continue
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass].astype(np.float32)
        near = (np.abs(u - u[i]) < .45 * h[i]) & (v >= v[i] - .2 * h[i]) & (v - h[i] < v[i] - .42 * h[i])
        near[i] = False
        out[i, :3] = lab.mean(0)
        out[i, 3:6] = np.median(lab, axis=0)
        out[i, 6] = (~grass).mean()
        out[i, 7] = float(np.any(near))
    return out


def _segment(job):
    """Colour readings of the frames idx (sorted), in one pass through the video."""
    import cv2
    import pandas as pd
    cv2.setNumThreads(1)
    idx, path = job
    if os.path.isfile(path):
        return path
    d = atd.load("tactical")
    P = d["players"][:, :, :2]
    hom = pd.read_csv(atd.paths("tactical")["homography"], usecols=["h%d%d" % (r, c) for r in range(3) for c in range(3)]).values.reshape(-1, 3, 3)
    cap = cv2.VideoCapture(atd.paths("tactical")["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[0]))
    pos = int(idx[0])
    out = np.full((len(idx), P.shape[1], N_F), np.nan, np.float32)
    for k, f in enumerate(idx):
        while pos < f:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            continue
        s = np.flatnonzero(np.isfinite(P[f, :, 0]))
        if len(s):
            out[k, s] = _read_frame(cv2, img, hom[f], P[f, s, 0].astype(np.float64), P[f, s, 1].astype(np.float64))
    cap.release()
    np.savez(path, idx=idx, feat=out)
    return path


def grid():
    """The frames read: every STEP-th graded frame, per half."""
    return {h: np.arange(a, b + 1, STEP) for h, a, b in atd.HALVES}


def extract(workers=6):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for h, idx in grid().items():
        for i, part in enumerate(np.array_split(idx, 3)):
            jobs.append((part, os.path.join(OUT, "part_h%d_%d.npz" % (h, i))))
    t = time.time()
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_segment, jobs)):
            print("  part %d / %d, %.0f s" % (k + 1, len(jobs), time.time() - t), flush=True)


def readings(h):
    idx = grid()[h]
    parts = [np.load(os.path.join(OUT, "part_h%d_%d.npz" % (h, i))) for i in range(3)]
    feat = np.concatenate([p["feat"] for p in parts])
    assert np.array_equal(np.concatenate([p["idx"] for p in parts]), idx)
    return idx, feat


def _gauss(x):
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826 + 1.0
    x = x[(np.abs(x - med) / mad).max(1) < 3.5]       # away from referees and overlaps
    mu = x.mean(0)
    cov = np.cov(x.T) + np.eye(x.shape[1]) * 1.0
    return mu, np.linalg.inv(cov), np.linalg.slogdet(cov)[1]


def kit_models(feat, team, gk, rounds=3):
    """Two Gaussians over the colour readings, from the clear readings of the field players, labels refined by the colour itself."""
    T, S, _ = feat.shape
    ok = np.isfinite(feat[:, :, 0]) & (np.nan_to_num(feat[:, :, 7]) == 0) & ~gk[None, :]
    X = feat[ok][:, :6].astype(np.float64)
    y = np.broadcast_to(team, (T, S))[ok].copy()
    for _ in range(rounds):
        models = [_gauss(X[y == k]) for k in (0, 1)]
        ll = llr(X, models)
        y = np.where(ll > 5, 0, np.where(ll < -5, 1, y))
    return models


def llr(X, models):
    """Log-odds of team 0 over team 1, per reading."""
    lp = []
    for mu, ic, ld in models:
        z = X - mu
        lp.append(-0.5 * (np.einsum("ij,jk,ik->i", z, ic, z) + ld))
    return lp[0] - lp[1]


def decode_slot(e, gap_after, own):
    """Viterbi over this slot's readings. e: log-odds of team 0 per reading (0 where nothing was read); gap_after: reading is the first after a gap; own: the slot's own team."""
    T = len(e)
    pri = np.where(own == 0, PRIOR, -PRIOR)
    em = np.stack([(e + pri) / 2, -(e + pri) / 2], 1)
    s = np.where(gap_after, S_NEW, S_RUN)
    dp = np.zeros((T, 2))
    bk = np.zeros((T, 2), np.int8)
    dp[0] = em[0] + np.array([0.0 if own == 0 else -S_NEW, 0.0 if own == 1 else -S_NEW])
    for t in range(1, T):
        for k in (0, 1):
            stay, move = dp[t - 1, k], dp[t - 1, 1 - k] - s[t]
            bk[t, k] = k if stay >= move else 1 - k
            dp[t, k] = max(stay, move) + em[t, k]
    out = np.zeros(T, np.int8)
    out[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1):
        out[t - 1] = bk[t, out[t]]
    return out


def fit():
    """Per half: per reading the slot's decoded team, -1 where the slot is absent. Saved as cache/team_color/team_fix.npz."""
    import bench
    d = atd.load("tactical")
    team = d["team"]
    f = bench._load(os.path.join(ROOT, "cache", "features_merged.pkl"))
    gk = np.zeros(len(team), bool)
    gk[f["gk_slots"]] = True
    res = {}
    for h, idx in grid().items():
        _, feat = readings(h)
        models = kit_models(feat, team, gk)
        T, S, _ = feat.shape
        fix = np.full((T, S), -1, np.int8)
        present = np.isfinite(d["players"][idx, :, 0])
        X = np.nan_to_num(feat[:, :, :6], nan=0.0).astype(np.float64).reshape(T * S, 6)
        e = llr(X, models).reshape(T, S)
        have = np.isfinite(feat[:, :, 0])
        e = np.clip(np.where(have, e, 0.0), -CLIP, CLIP)
        e = np.where(np.nan_to_num(feat[:, :, 7]) > 0, W_OVER * e, e)
        for s in range(S):
            on = np.flatnonzero(present[:, s])
            if not len(on):
                continue
            if gk[s]:
                fix[on, s] = team[s]
                continue
            gap = np.r_[True, np.diff(on) >= 5]       # >= 1 s between readings of this slot
            fix[on, s] = decode_slot(e[on, s], gap, int(team[s]))
        res["fix_h%d" % h] = fix
        res["llr_h%d" % h] = np.where(have, llr(X, models).reshape(T, S), np.nan).astype(np.float32)
        wrong = (fix >= 0) & (fix != team[None, :])
        print("half %d: readings %d, colour read on %.3f of the present objects; slots' frames that follow the other team %.4f" % (h, T, have[present].mean(), wrong.sum() / max((fix >= 0).sum(), 1)), flush=True)
    np.savez(os.path.join(OUT, "team_fix.npz"), **res)


def repaired(P, team, gk, a, h):
    """The half's tracking with the repair applied. P (T, S, 2+), frames a, a + 1, ...; returns P2 (T, 2S, 2), team2, gk2: the frames of slot s that follow the other team sit in the spare slot S + s, of that team."""
    z = np.load(os.path.join(OUT, "team_fix.npz"))
    fix = z["fix_h%d" % h]
    idx = grid()[h]
    T, S = P.shape[0], P.shape[1]
    k = np.clip(np.round((np.arange(T) + a - idx[0]) / STEP).astype(np.int64), 0, len(idx) - 1)
    f = fix[k]                                   # (T, S), nearest reading
    # a slot's state between readings where it was absent at the reading: fill from its nearest present reading
    other = (f >= 0) & (f != np.asarray(team)[None, :])
    P2 = np.full((T, 2 * S, 2), np.nan, np.float32)
    keep = np.asarray(P[:, :, :2], np.float32).copy()
    keep[other] = np.nan
    P2[:, :S] = keep
    moved = np.full((T, S, 2), np.nan, np.float32)
    moved[other] = np.asarray(P[:, :, :2], np.float32)[other]
    P2[:, S:] = moved
    return P2, np.r_[np.asarray(team), 1 - np.asarray(team)].astype(np.asarray(team).dtype), np.r_[np.asarray(gk), np.zeros(S, bool)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if cmd == "extract":
        extract()
    elif cmd == "fit":
        fit()
