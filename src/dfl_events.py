"""Round 12, idea R12-3 (docs/RAPOR.md): the open DFL / IDSSE event files (never used before) synchronised to the
tracking frames. Manually coded event times lag the tracking (~1.5 s on average, Vidal-Codina et al. 2022), so every
event with a player is moved to the frame where that player is closest to the (clean) ball within [-3, +1] s of its
coded time (ETSY-style), if that distance is below 2 m.

Outputs per match (cache/dfl/<match>_events.pkl): list per half of dicts
  frame (index in dfl.parse()'s half arrays), raw_frame, team (0 home / 1 away / -1), kind, sub, player slot (-1),
  winner_team / poss_change (duels), evaluation (passes), phase (BallPossessionPhase), dist (player-ball at frame)
and two label conventions on dfl.parse()'s frames (H 0 / A 1 / D 2 from BallStatus):
  phase   team of the latest synced event's BallPossessionPhase owner (DFL's own possession phases)
  evctrl  native possession flag, but a native spell counts only if its team shows a controlled event inside it
          (pass, cross, shot, ball claiming, set piece, or an event of the same team >= 1 s after the spell start);
          otherwise it goes back to the previous team (the challenge's "controlled possession" read with DFL events)

    python -I src/dfl_events.py stats
"""
import datetime as dt
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2
CONTROL_KINDS = {"Play", "ShotAtGoal", "BallClaiming", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff",
                 "Penalty"}


def _t(s):
    return dt.datetime.fromisoformat(s).timestamp()


def meta(match):
    """PersonId order of dfl.parse()'s player slots and the time of the first frame, per half."""
    path = os.path.join(dfl.CACHE, "%s_meta.pkl" % match)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    teams, _, _ = dfl.match_info(match)
    order, t0, n0 = {}, {}, {}
    for _, el in ET.iterparse(dfl._file("positions_raw_observed", match), events=("end",)):
        if el.tag != "FrameSet":
            continue
        sec, tid, pid = el.get("GameSection"), el.get("TeamId"), el.get("PersonId")
        fr = el.find("Frame")
        if tid and tid.upper() == "BALL":
            t0[sec], n0[sec] = _t(fr.get("T")), int(fr.get("N"))
        elif tid in teams:
            order.setdefault(sec, []).append(pid)
        el.clear()
    secs = sorted(t0, key=lambda s: 0 if s == "firstHalf" else 1)
    out = {"secs": secs, "pids": [order[s] for s in secs], "t0": [t0[s] for s in secs], "n0": [n0[s] for s in secs]}
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def raw_events(match):
    teams, _, _ = dfl.match_info(match)
    role = {tid: (0 if r == "home" else 1) for tid, r in teams.items()}
    out = []
    for ev in ET.parse(dfl._file("events_raw", match)).getroot().iter("Event"):
        kids = list(ev)
        if not kids or not ev.get("EventTime"):
            continue
        k = kids[0]
        kind, sub = k.tag, (k[0].tag if len(k) else "")
        a = dict(k.attrib)
        if len(k):
            a.update({kk: vv for kk, vv in k[0].attrib.items() if kk not in a})
            if len(k[0]):
                a.update({kk: vv for kk, vv in k[0][0].attrib.items() if kk not in a})
        team = role.get(a.get("Team"), -1)
        if kind == "TacklingGame":
            team = role.get(a.get("WinnerTeam"), -1)
        out.append(dict(t=_t(ev.get("EventTime")), kind=kind, sub=sub, team=team, player=a.get("Player") or a.get("Winner"),
                        winner_team=role.get(a.get("WinnerTeam"), -1), poss_change=a.get("PossessionChange"),
                        winner_role=a.get("WinnerRole"), evaluation=a.get("Evaluation"),
                        phase=a.get("BallPossessionPhase")))
    return out


def synced(match, refresh=False):
    path = os.path.join(dfl.CACHE, "%s_events.pkl" % match)
    if os.path.isfile(path) and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    d = dfl.parse(match)
    m = meta(match)
    evs = raw_events(match)
    res = []
    for hi in range(2):
        tag = "h%d" % (hi + 1)
        P, B = d[tag + "_players"], d[tag + "_ball"]
        L, W = d["pitch"]
        n = len(B)
        slot = {p: j for j, p in enumerate(m["pids"][hi])}
        t0 = m["t0"][hi]
        t_end = t0 + n / FPS
        half = []
        for e in evs:
            if not (t0 - 5 <= e["t"] <= t_end + 5):
                continue
            raw = int(round((e["t"] - t0) * FPS))
            j = slot.get(e["player"], -1)
            frame, dist = raw, np.nan
            if j >= 0:
                lo, hi_ = max(0, raw - 3 * FPS), min(n, raw + FPS + 1)
                if hi_ > lo:
                    dx = (P[lo:hi_, j, 0] - B[lo:hi_, 0]) * L
                    dy = (P[lo:hi_, j, 1] - B[lo:hi_, 1]) * W
                    dd = np.hypot(dx, dy)
                    if np.isfinite(dd).any():
                        k = int(np.nanargmin(dd))
                        if dd[k] < 2.0:
                            frame, dist = lo + k, float(dd[k])
            if 0 <= frame < n:
                half.append(dict(e, frame=frame, raw_frame=raw, slot=j, dist=dist))
        half.sort(key=lambda e: e["frame"])
        res.append(half)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return res


def phase_labels(lab, evs):
    """DFL possession phases: alive frames take the owner team of the latest synced event's phase."""
    out = lab.copy()
    owner = {}
    for e in evs:
        if e["phase"] is not None and e["team"] >= 0 and e["kind"] in CONTROL_KINDS | {"OtherBallAction", "TacklingGame"}:
            owner.setdefault(e["phase"], e["team"])
    cur = np.full(len(lab), -1, np.int8)
    for e in evs:
        if e["phase"] in owner:
            cur[e["frame"]] = owner[e["phase"]]
    last = -1
    for i in range(len(lab)):
        if cur[i] >= 0:
            last = cur[i]
        if lab[i] < 2 and last >= 0:
            out[i] = last
    return out


def evctrl_labels(lab, evs, keep_s=1.0):
    """Native flag, but a spell of team k after the other team's spell stays only if k shows control inside it."""
    out = lab.copy()
    n = len(lab)
    ctrl = np.zeros((2, n), bool)
    anyev = np.full((2, n), -1, np.int64)
    for e in evs:
        if e["team"] < 0:
            continue
        f = e["frame"]
        if e["kind"] in CONTROL_KINDS:
            ctrl[e["team"], f] = True
        anyev[e["team"], f] = f
    e_ = np.flatnonzero(np.diff(lab) != 0) + 1
    st, en = np.concatenate(([0], e_)), np.concatenate((e_, [n]))
    keep = int(keep_s * FPS)
    for i in range(1, len(st)):
        a, b, k = st[i], en[i], out[st[i]]
        prev = out[st[i - 1]]
        if k > 1 or prev > 1 or prev == k or st[i - 1] + (en[i - 1] - st[i - 1]) != a:
            continue
        if ctrl[k, a:b].any() or (anyev[k, min(n - 1, a + keep):b] >= 0).any():
            continue
        out[a:b] = prev
    return out


def stats():
    for m in dfl.MATCHES:
        d = dfl.parse(m)
        ev = synced(m)
        parts = []
        for hi in range(2):
            lab = d["h%d_labels" % (hi + 1)]
            e = ev[hi]
            moved = np.array([x["frame"] - x["raw_frame"] for x in e if np.isfinite(x["dist"])])
            ph, ec = phase_labels(lab, e), evctrl_labels(lab, e)

            def nsp(x):
                alive = x < 2
                c = np.flatnonzero(np.diff(x) != 0) + 1
                return int(sum(1 for s in np.concatenate(([0], c)) if alive[s]))
            parts.append("h%d events %d synced %.2f median shift %+.2f s | spells native %d phase %d evctrl %d | "
                         "phase = native on %.3f of alive" % (hi + 1, len(e), np.mean([np.isfinite(x["dist"]) for x in e]),
                                                              np.median(moved) / FPS if len(moved) else np.nan,
                                                              nsp(lab), nsp(ph), nsp(ec),
                                                              np.mean((ph == lab)[lab < 2])))
        print(m, " || ".join(parts), flush=True)


if __name__ == "__main__":
    {"stats": stats}[sys.argv[1] if len(sys.argv) > 1 else "stats"]()
