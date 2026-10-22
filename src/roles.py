"""Role awareness without shirt numbers (round 7, user idea): each player's role from where he has been over the last
minutes and his rank inside his team -- goalkeeper, centre back, full back, midfielder, forward.

Per half: the attacking direction of each team (feat_team2's rule: home goalkeeper's side, else the centroids);
per player the mean depth towards the opponent goal and the mean distance from the pitch's long axis over a
centred window (+-ROLE_WIN frames, missing frames ignored); per frame the depth rank among the outfield team-mates
seen (0 = deepest, 1 = most advanced). Groups by rank (4-4-2-like): the 4 deepest outfield players are defenders
(the 2 widest of them full backs), the 2 most advanced forwards, the rest midfielders.

Validated against the metadata roles of DFL (PlayingPosition) and SkillCorner (player_role.position_group) in
src/exp_role.py; team-perspective features of the players nearest to the ball in role_feats().
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

import dfl
import feat_team2
import skillcorner as scr
from mlfeat import win_mean

ROLE_WIN = 3750                 # +-2.5 min
GROUPS = ("GK", "CB", "FB", "MID", "FWD")
DFL_GROUP = {"TW": "GK", "IVL": "CB", "IVR": "CB", "IVZ": "CB", "LV": "FB", "RV": "FB",
             "STZ": "FWD", "STL": "FWD", "STR": "FWD", "LA": "FWD", "RA": "FWD"}   # every other code: midfield
SC_GROUP = {"Goalkeeper": "GK", "Central Defender": "CB", "Full Back": "FB", "Midfield": "MID",
            "Wide Attacker": "FWD", "Center Forward": "FWD"}
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "roles")


def dfl_slot_roles(match):
    """{half: [group per slot]} in dfl.parse's slot order (players in order of appearance per half)."""
    path = os.path.join(CACHE, "dfl_%s.json" % match)
    if os.path.isfile(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    teams, players, _ = dfl.match_info(match)
    order = {}
    for ev, el in ET.iterparse(dfl._file("positions_raw_observed", match), events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            sec, tid, pid = el.get("GameSection"), el.get("TeamId"), el.get("PersonId")
            if tid and tid.upper() != "BALL" and tid in teams:
                order.setdefault(sec, [])
                if pid not in order[sec]:
                    order[sec].append(pid)
        elif ev == "end" and el.tag in ("Frame", "FrameSet"):
            el.clear()
    out = {}
    for si, sec in enumerate(sorted(order, key=lambda s: 0 if s == "firstHalf" else 1)):
        out[si + 1] = [DFL_GROUP.get(players.get(p, (None, None))[1] or "", "MID")
                       if (players.get(p, (None, None))[1]) else None for p in order[sec]]
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def sc_slot_roles(match):
    """[group per slot] in skillcorner.parse's slot order; None for substitutes ("Other"/SUB)."""
    meta = json.load(open(os.path.join(scr.DATA, match, "%s_match.json" % match), encoding="utf-8"))
    out = []
    for p in meta["players"]:
        r = p["player_role"]
        g = "GK" if r.get("acronym") == "GK" else SC_GROUP.get(r.get("position_group"))
        out.append(g)
    return out


def infer(P, team, gk, pitch, B=None):
    """Per frame and slot: depth (m towards the opponent goal), width (m from the long axis), depth rank among the
    outfield team-mates (0 deepest .. 1 most advanced, NaN if unseen) and the inferred group index into GROUPS."""
    S = feat_team2._prep(P, B if B is not None else np.full((len(P), 2), np.nan, np.float32), team, gk, pitch)
    L, W = S["L"], S["W"]
    x, y = S["x"], S["y"]
    n, s = x.shape
    depth = np.full((n, s), np.nan)
    for j in range(s):
        k = int(team[j])
        xr = x[:, j] if S["sgn"][k] > 0 else L - x[:, j]
        depth[:, j] = win_mean(xr, -ROLE_WIN, ROLE_WIN)
    width = np.stack([win_mean(np.abs(y[:, j] - W / 2), -ROLE_WIN, ROLE_WIN) for j in range(s)], 1)
    seen = np.isfinite(x)
    rank = np.full((n, s), np.nan)
    group = np.full((n, s), -1, np.int8)
    gk = np.asarray(gk, bool)
    for k in (0, 1):
        cols = np.flatnonzero((team == k) & ~gk)
        d = np.where(seen[:, cols] & np.isfinite(depth[:, cols]), depth[:, cols], np.inf)
        order = np.argsort(d, axis=1)
        rk = np.empty_like(order)
        np.put_along_axis(rk, order, np.arange(len(cols))[None, :].repeat(n, 0), axis=1)
        cnt = np.isfinite(d).sum(1)
        r = rk / np.maximum(cnt - 1, 1)[:, None]
        r[~np.isfinite(d)] = np.nan
        rank[:, cols] = r
        # groups: 4 deepest defenders (2 widest of them full backs), 2 most advanced forwards, midfield between
        g = np.full((n, len(cols)), GROUPS.index("MID"), np.int8)
        g[rk < 4] = GROUPS.index("CB")
        g[rk >= (cnt - 2)[:, None]] = GROUPS.index("FWD")
        wd = np.where(rk < 4, width[:, cols], -np.inf)
        worder = np.argsort(-wd, axis=1)[:, :2]
        fb = np.zeros_like(g, bool)
        np.put_along_axis(fb, worder, True, axis=1)
        g[fb & (rk < 4)] = GROUPS.index("FB")
        g[~np.isfinite(d)] = -1
        group[:, cols] = g
        gcols = np.flatnonzero((team == k) & gk)
        group[:, gcols] = np.where(seen[:, gcols], GROUPS.index("GK"), -1)
        rank[:, gcols] = np.where(seen[:, gcols], -0.25, np.nan)
    return dict(depth=depth, width=width, rank=rank, group=group, S=S)


def role_feats(P, B, team, gk, pitch, R=None):
    """Team-perspective features (home view, away view), each (n, F): roles of the players nearest to the ball."""
    R = R or infer(P, team, gk, pitch, B)
    S = R["S"]
    L, W = S["L"], S["W"]
    pxy, bxy = S["pxy"], S["bxy"]
    n = len(pxy)
    d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    rows = np.arange(n)
    near = {}
    for k in (0, 1):
        cols = np.flatnonzero(team == k)
        j = cols[np.argmin(d[:, cols], axis=1)]
        ok = np.isfinite(d[rows, j])
        opp = np.flatnonzero(team != k)
        dd = np.linalg.norm(pxy[rows, j][:, None, :] - pxy[:, opp], axis=2)
        press = np.nansum(dd < 3.0, axis=1).astype(np.float32)
        near[k] = dict(rank=np.where(ok, R["rank"][rows, j], np.nan), width=np.where(ok, R["width"][rows, j], np.nan),
                       group=np.where(ok, R["group"][rows, j], -1), dist=np.where(ok, d[rows, j], np.nan),
                       press=np.where(ok, press, np.nan))
    out = []
    for k in (0, 1):
        o = 1 - k
        bx_own = bxy[:, 0] if S["sgn"][k] > 0 else L - bxy[:, 0]       # ball distance from k's own goal line
        in_own_box = (bx_own < 16.5) & (np.abs(bxy[:, 1] - W / 2) < 20.16)
        own, opp = near[k], near[o]
        defender = np.isin(own["group"], (GROUPS.index("CB"), GROUPS.index("FB"), GROUPS.index("GK")))
        forward = own["group"] == GROUPS.index("FWD")
        cols = [own["rank"], own["width"], opp["rank"], opp["width"],
                (defender & in_own_box & (own["press"] >= 1)).astype(np.float32),
                (forward & (bx_own > L / 2)).astype(np.float32),
                (np.isin(opp["group"], (GROUPS.index("CB"), GROUPS.index("FB"), GROUPS.index("GK")))
                 & (bx_own > L - 16.5)).astype(np.float32),
                own["press"], opp["press"], own["rank"] - opp["rank"]]
        for lo, hi in ((-12, 12), (-50, 0), (0, 50)):
            cols += [win_mean(own["rank"], lo, hi), win_mean(opp["rank"], lo, hi)]
        out.append(np.stack(cols, 1).astype(np.float32))
    return out
