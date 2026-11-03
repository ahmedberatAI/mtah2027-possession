"""SoccerTrack v2 (huggingface.co/datasets/atomscott/soccertrack-v2, CC BY 4.0; fetched by src/download_soccertrack.py):
ten university matches filmed by a fixed panoramic camera, with the provider's raw exports in raw/<match>/:
tracker_box_data.xml (per frame: player positions on a 0.01 pitch grid, i.e. 1.05 m x 0.68 m, the ball and the
provider's ball status HOME / AWAY / NEUTRAL / BALLOUT) and player_nodes.csv (the provider's event stream, with pass
outcomes). The ball is reconstructed from the events (anchored at them, straight lines between), so ball features carry
the event timing. Two matches (NO_STATUS) carry no status ('INPLAY' everywhere) and are left out of matches().

  parse(match)          per half: provider status, ball, players on the 25 fps grid (cached in cache/soccertrack/)
  events(match)         per half: (frame, team, kind, outcome) in gradient.controlled_reading's vocabulary
  labels(match, conv)   'native' (provider status, NEUTRAL keeps the team that had the ball) or 'evctrl' (the
                        challenge's controlled reading of it, keep 0.5 s as for PFF and SkillCorner)
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.join(ROOT, "external_data", "soccertrack_v2")
CACHE = os.path.join(ROOT, "cache", "soccertrack")
FPS, MS = 25, 40
H, A, D, NEUTRAL, INPLAY = 0, 1, 2, 3, 4
PITCH = (105.0, 68.0)
PERIODS = ("FIRST_HALF", "SECOND_HALF")
STATUS = {"HOME": H, "AWAY": A, "BALLOUT": D, "NEUTRAL": NEUTRAL, "INPLAY": INPLAY}
NO_STATUS = ("132831", "132877")    # M9, M10: every frame 'INPLAY' (no team, no dead ball), so no labels
OFF_BALL = {"changeIn", "changeOut", "pause", "foulCommitted", "foulConceded", "offside", "handballFoul"}
_FRAME = re.compile(r'<frame matchTime="(\d+)" frameNumber="\d+" eventPeriod="(\w+)" ballStatus="(\w+)"')
_OBJ = re.compile(r'<(?:player|ball) playerId="(\w+)" loc="\[([^,\]]+), ([^\]]+)\]"')


def matches():
    """The eight matches whose export carries the provider's ball status."""
    return sorted(d for d in os.listdir(os.path.join(DATA, "raw")) if d.isdigit() and d not in NO_STATUS)


def _raw(match, suffix):
    return os.path.join(DATA, "raw", match, "%s_%s" % (match, suffix))


def teams(match):
    """({player id: 0 home / 1 away}, home team id)."""
    meta = open(_raw(match, "tracker_box_metadata.xml"), encoding="utf-8").read()
    side = {t: s for t, s in re.findall(r'<team id="(\d+)"[^>]*side="(\w+)"', meta)}
    home = next(t for t, s in side.items() if s == "home")
    return {int(p): int(side[t] == "away") for p, t in re.findall(r'<player id="(\d+)"[^>]*teamId="(\d+)"', meta)}, home


def goalkeepers(match):
    meta = open(_raw(match, "tracker_box_metadata.xml"), encoding="utf-8").read()
    return {int(p) for p in re.findall(r'<player id="(\d+)"[^>]*position="GK"', meta)}


def halves(match):
    """bench-style halves: players (n, P, 2) and ball (n, 2) as pitch fractions, team / gk per slot, native labels."""
    d = parse(match)
    gk = goalkeepers(match)
    out = []
    for h in (1, 2):
        out.append({"players": d["h%d_players" % h], "ball": d["h%d_ball" % h], "team": d["team"],
                    "gk": np.array([int(p) in gk for p in d["pids"]]), "pitch": PITCH,
                    "labels": native(d["h%d_status" % h])})
    return out


def parse(match):
    """{'h1_status', 'h1_t' (ms on the provider's clock), 'h1_ball' (n, 2), 'h1_players' (n, P, 2), 'pids', 'team'}:
    coordinates as fractions of the 105 x 68 m pitch, NaN where a player is absent; frames missing from the export
    get status NEUTRAL and NaN positions."""
    path = os.path.join(CACHE, "%s.npz" % match)
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    side, _ = teams(match)
    pids = np.array(sorted(side), np.int64)
    slot = {int(p): j for j, p in enumerate(pids)}
    rows = {p: [] for p in PERIODS}
    cur = None
    with open(_raw(match, "tracker_box_data.xml"), encoding="utf-8") as fh:
        for line in fh:
            m = _FRAME.search(line)
            if m:
                cur = rows.get(m.group(2))
                if cur is not None:
                    cur.append([int(m.group(1)), STATUS[m.group(3)], np.nan, np.nan, []])
                continue
            if cur is None:
                continue
            m = _OBJ.search(line)
            if m:
                x, y = float(m.group(2)), float(m.group(3))
                if m.group(1) == "ball":
                    cur[-1][2:4] = x, y
                elif int(m.group(1)) in slot:
                    cur[-1][4].append((slot[int(m.group(1))], x, y))
    out = {"pids": pids, "team": np.array([side[int(p)] for p in pids], np.int8)}
    for h, per in enumerate(PERIODS, 1):
        r = rows[per]
        t = np.array([x[0] for x in r], np.int64)
        idx = np.round((t - t[0]) / MS).astype(np.int64)
        n = int(idx[-1]) + 1
        status = np.full(n, NEUTRAL, np.int8)
        ball = np.full((n, 2), np.nan, np.float32)
        players = np.full((n, len(pids), 2), np.nan, np.float32)
        status[idx] = [x[1] for x in r]
        ball[idx] = [x[2:4] for x in r]
        for i, x in zip(idx, r):
            for j, px, py in x[4]:
                players[i, j] = px, py
        out["h%d_t" % h] = t[0] + MS * np.arange(n)
        out["h%d_status" % h] = status
        out["h%d_ball" % h] = ball
        out["h%d_players" % h] = players
        out["h%d_missing" % h] = np.array([n - len(r)])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def events(match, d=None):
    """[h1, h2] lists of (frame, team, kind, outcome, player id) for the provider's on-ball events: a completed pass is
    ('PA', 'C'), a completed cross ('CR', 'C'), a shot 'SH'; substitutions, pauses, fouls and offsides are left out."""
    d = parse(match) if d is None else d
    side, _ = teams(match)
    ev = pd.read_csv(_raw(match, "player_nodes.csv"), dtype=str, keep_default_na=False)
    out = []
    for h, per in enumerate(PERIODS, 1):
        t = d["h%d_t" % h]
        half = []
        for r in ev[ev.event_period == per].itertuples():
            toks = set(r.filtered_event_types.split())
            if not r.player_id or int(r.player_id) not in side or not toks - OFF_BALL:
                continue
            f = int(np.clip(np.round((int(r.event_time) - t[0]) / MS), 0, len(t) - 1))
            if "shot" in toks:
                kind, oc = "SH", None
            elif toks & {"crossSucceeded", "crossFailed"}:
                kind, oc = "CR", "C" if "crossSucceeded" in toks else None
            elif toks & {"passSucceeded", "passFailed"}:
                kind, oc = "PA", "C" if "passSucceeded" in toks else None
            else:
                kind, oc = sorted(toks - OFF_BALL)[0], None
            half.append((f, side[int(r.player_id)], kind, oc, int(r.player_id)))
        out.append(sorted(half))
    return out


def native(status):
    """Provider status as H / A / D; a NEUTRAL frame keeps the team that had the ball (the next team at the start of an
    alive stretch)."""
    out = np.asarray(status, np.int8).copy()
    for seq in (range(len(out)), range(len(out) - 1, -1, -1)):
        last = -1
        for i in seq:
            if out[i] in (H, A):
                last = out[i]
            elif out[i] == D:
                last = -1
            elif last >= 0:
                out[i] = last
    out[out == NEUTRAL] = D
    return out


def labels(match, conv="evctrl", keep_s=.5):
    """[h1, h2] label arrays (H / A / D) under conv."""
    import gradient
    d = parse(match)
    nat = [native(d["h%d_status" % h]) for h in (1, 2)]
    if conv == "native":
        return nat
    assert conv == "evctrl", conv
    return [gradient.controlled_reading(lab, [e[:4] for e in evs], keep_s) for lab, evs in zip(nat, events(match, d))]
