"""Per-frame features from the tactical ATD, homography and Smart Tagging. Vectorised numpy.

Every array is indexed by video frame (0..199318). Distances in metres, speeds in m/s (the ATD
speed channel is m/s despite the 'kph' label in the XML -- it matches position-derived speed with
ratio 1.006). The ATD ball speed channel is NOT used: it is sparse and uncorrelated with motion.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import atd

L, W = atd.PITCH
FPS = atd.FPS
TEAM_TAG = {"barcelona": 0, "atletic club": 1, "athletic club": 1}


def rolling_mean(x, half, valid=None):
    """Centred mean over +-half frames, ignoring NaN (or ~valid). NaN where no sample."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x) if valid is None else (valid & np.isfinite(x))
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    num, den = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def rolling_frac(mask, half):
    m = np.asarray(mask, dtype=np.float64)
    cs = np.concatenate(([0.0], np.cumsum(m)))
    i = np.arange(len(m))
    lo, hi = np.maximum(0, i - half), np.minimum(len(m), i + half + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def runs_of(mask):
    m = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def gap_context(seen):
    """For each frame: frames since the ball was last seen / until it is next seen (0 if seen)."""
    n = len(seen)
    idx = np.arange(n)
    last = np.where(seen, idx, -1)
    last = np.maximum.accumulate(last)
    since = np.where(last >= 0, idx - last, n)
    nxt = np.where(seen, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    until = np.where(nxt < n, nxt - idx, n)
    return since, until, last, nxt


def ball_kinematics(ball_xy, seen):
    """Velocity (m/s) by central differences over observed neighbours; NaN where unknown."""
    n = len(ball_xy)
    fwd = np.full((n, 2), np.nan)
    bwd = np.full((n, 2), np.nan)
    ok = seen[1:] & seen[:-1]
    step = (ball_xy[1:] - ball_xy[:-1]) * FPS
    fwd[:-1][ok] = step[ok]
    bwd[1:][ok] = step[ok]
    with np.errstate(invalid="ignore"):
        v = np.nanmean(np.stack([fwd, bwd]), axis=0)
    return v


def player_velocity(pxy):
    """(n, slots, 2) velocity in m/s from position differences, NaN at track edges."""
    v = np.full(pxy.shape, np.nan, dtype=np.float32)
    v[1:-1] = (pxy[2:] - pxy[:-2]) * FPS / 2.0
    # identity switches make a slot jump between people; nobody runs faster than ~11 m/s
    fast = np.linalg.norm(v, axis=-1) > 12.0
    v[fast] = np.nan
    return v


def load_tags(path=None):
    path = path or atd.paths()["smart_tagging"]
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        rows.append({"code": (inst.findtext("code") or "").strip().upper(),
                     "start": float(inst.findtext("start")), "end": float(inst.findtext("end")),
                     "team": TEAM_TAG.get((tags.get("Team") or "").strip().lower(), -1),
                     "type": (tags.get("Type") or "").strip().lower(), "half": tags.get("Half")})
    df = pd.DataFrame(rows)
    df["sf"] = (df["start"] * FPS).round().astype(int)
    df["ef"] = (df["end"] * FPS).round().astype(int)
    return df


# Codes whose Team tag is the team IN possession, and those whose Team is the DEFENDING team.
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
                "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
DEFEND_CODES = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")


def tag_features(n, tags=None):
    tags = load_tags() if tags is None else tags
    out = {}
    home = np.zeros(n)
    away = np.zeros(n)
    for _, r in tags.iterrows():
        a, b = max(0, r.sf), min(n, r.ef)
        if r.code in ATTACK_CODES and r.team >= 0:
            (home if r.team == 0 else away)[a:b] += 1
        elif r.code in DEFEND_CODES and r.team >= 0:
            (away if r.team == 0 else home)[a:b] += 1
    out["tag_home"] = home
    out["tag_away"] = away
    sp = np.zeros(n)
    for _, r in tags[tags.code == "SET PIECES"].iterrows():
        sp[max(0, r.sf):min(n, r.ef)] = 1
    out["tag_setpiece"] = sp
    return out


def homography_features(n):
    table = pd.read_csv(atd.paths()["homography"])
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    H = table[cols].values.reshape(-1, 3, 3)
    Hinv = np.linalg.inv(H)

    def unproj(px, py):
        v = np.array([px, py, 1.0])
        o = Hinv @ v
        return atd.to_metres(np.stack([o[:, 0] / o[:, 2], o[:, 1] / o[:, 2]], 1))

    centre = unproj(960, 540)
    left, right = unproj(0, 540), unproj(1919, 540)
    width = np.linalg.norm(right - left, axis=1)
    return {"cam_x": centre[:, 0], "cam_y": centre[:, 1], "cam_w": width, "H": H}


def build(source="tactical", ball_override=None):
    """ball_override: optional (n, 2) metres array used wherever the tactical ball is missing
    (e.g. the aligned broadcast ball); NaN rows stay missing."""
    d = atd.load(source)
    P = d["players"]
    B = d["ball"]
    team = d["team"]
    n = len(B)
    pxy = atd.to_metres(P[:, :, :2])
    present = ~np.isnan(P[:, :, 0])
    bxy = atd.to_metres(B[:, :2])
    seen = ~np.isnan(B[:, 0])
    tac_seen = seen.copy()
    if ball_override is not None:
        fill = ~seen & np.isfinite(ball_override[:, 0])
        bxy = bxy.copy()
        bxy[fill] = ball_override[fill]
        seen = seen | fill

    f = {"n": n, "team": team, "present": present, "pxy": pxy, "bxy": bxy, "seen": seen,
         "tac_seen": tac_seen}
    # ball
    f["bvel"] = ball_kinematics(bxy, seen)
    f["bspeed"] = np.linalg.norm(f["bvel"], axis=1)
    out_x = np.maximum(-bxy[:, 0], bxy[:, 0] - L)
    out_y = np.maximum(-bxy[:, 1], bxy[:, 1] - W)
    f["ball_out"] = np.where(seen, np.maximum(out_x, out_y), np.nan)  # >0 = metres outside
    f["edge_dist"] = np.where(seen, np.minimum.reduce([bxy[:, 0], L - bxy[:, 0], bxy[:, 1],
                                                       W - bxy[:, 1]]), np.nan)
    since, until, last, nxt = gap_context(seen)
    f["since_seen"], f["until_seen"], f["last_seen"], f["next_seen"] = since, until, last, nxt

    # players vs ball
    diff = pxy - bxy[:, None, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    f["dist"] = dist
    dh = np.where(team[None, :] == 0, dist, np.inf)
    da = np.where(team[None, :] == 1, dist, np.inf)
    f["dh"], f["da"] = dh.min(1), da.min(1)
    f["ih"], f["ia"] = dh.argmin(1), da.argmin(1)
    order = np.argsort(dist, axis=1)
    f["near1"] = order[:, 0]
    f["d1"] = np.take_along_axis(dist, order[:, :1], 1)[:, 0]
    f["d2"] = np.take_along_axis(dist, order[:, 1:2], 1)[:, 0]
    f["t1"] = team[order[:, 0]]
    for r in (2.0, 5.0):
        f["nh_%g" % r] = (dh <= r).sum(1)
        f["na_%g" % r] = (da <= r).sum(1)

    # relative speed ball vs nearest player (control indicator)
    pvel = player_velocity(pxy)
    f["pvel"] = pvel
    rel = f["bvel"] - pvel[np.arange(n), f["near1"]]
    f["rel_speed"] = np.linalg.norm(rel, axis=1)

    # player kinematics
    spd = np.linalg.norm(pvel, axis=2)
    spd = np.where(present, spd, np.nan)
    with np.errstate(invalid="ignore"):
        f["spd_all"] = np.nanmean(spd, axis=1)
        f["spd_h"] = np.nanmean(np.where(team[None, :] == 0, spd, np.nan), axis=1)
        f["spd_a"] = np.nanmean(np.where(team[None, :] == 1, spd, np.nan), axis=1)
        f["spd_max"] = np.nanmax(np.where(np.isnan(spd), -1, spd), axis=1)
    for h in (12, 25, 50, 100):
        f["spd_all_s%d" % h] = rolling_mean(f["spd_all"], h)
    f["n_present"] = present.sum(1)
    with np.errstate(invalid="ignore"):
        for k, name in ((0, "h"), (1, "a")):
            m = present & (team[None, :] == k)
            x = np.where(m, pxy[:, :, 0], np.nan)
            y = np.where(m, pxy[:, :, 1], np.nan)
            f["cx_" + name] = np.nanmean(x, 1)
            f["cy_" + name] = np.nanmean(y, 1)
            f["sx_" + name] = np.nanstd(x, 1)
            f["sy_" + name] = np.nanstd(y, 1)
            f["n_" + name] = m.sum(1)

    # goalkeepers (one slot per team carries position_type == Goalkeeper)
    gk = np.flatnonzero(d["slots"]["position_type"].values == "Goalkeeper")
    f["gk_slots"] = gk
    for s in gk:
        name = "h" if team[s] == 0 else "a"
        f["gk_x_" + name] = pxy[:, s, 0]
        f["gk_y_" + name] = pxy[:, s, 1]
        f["gk_d_" + name] = np.where(np.isinf(dist[:, s]), np.nan, dist[:, s])

    f.update(tag_features(n))
    f.update(homography_features(n))
    f["graded"] = atd.graded_mask(n)
    f["half"] = atd.half_of_frame(n)
    return f
