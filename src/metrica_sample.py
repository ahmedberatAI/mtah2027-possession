"""Metrica Sports public sample games (github.com/metrica-sports/sample-data) -> arrays in the same
layout as dfl.parse(), with per-frame H/A/D labels derived from the synchronised event data.

Licence: the repository has no licence file; its README asks users to "be responsible with the use
of this data" and to acknowledge Metrica Sports as the source when it is used for anything public.
The files stay in external_data/metrica (gitignored) and are never redistributed.

Labels (events are frame-synchronised with the tracking, 25 fps):
  D  from a dead-ball trigger until the next restart. Triggers: BALL OUT (end frame = ball crosses
     the line), SHOT ...-OUT / ...-GOAL (end frame), FAULT RECEIVED and CHALLENGE ...-FAULT-...
     (start frame), BALL LOST / PASS with OFFSIDE, FORCED, HAND BALL, END HALF (end frame).
     Restarts: SET PIECE (not RETAKEN) and any event with a GOAL KICK subtype (goal kicks are PASS /
     BALL LOST / BALL OUT events in this data, not SET PIECE). A restart without a preceding trigger
     closes a dead stretch that starts at the end of the previous action. Frames before the kick-off
     of a half and after its END HALF are D.
  H/A  team of the last controlling event (SET PIECE, PASS, CARRY, RECOVERY, SHOT, BALL LOST start
     frames); CHALLENGE, BALL OUT and FAULT RECEIVED do not change possession, so a loose ball stays
     with the last team that had it.
The D boundaries coincide with the ball-missing runs of Metrica's own clean tracking (median offset
+1 / 0 frames). The sample data carry no possession flag, so CONVENTIONS also offers stricter readings
of possession change: tau1-3 (short spells merged, merge_short) and ctrl (the challenge's "controlled
possession" applied to the events, merge_uncontrolled).

  .venv/Scripts/python.exe -I src/metrica_sample.py download   # ~185 MB, pinned commit
  .venv/Scripts/python.exe -I src/metrica_sample.py stats      # possession-spell statistics
"""
import io
import json
import os
import sys
import urllib.request
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external_data", "metrica")
DATA = os.path.join(SRC, "data")
CACHE = os.path.join(ROOT, "cache", "metrica")
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"   # master, 2021-04-15 "Added Sample Game 3"
RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/".format(COMMIT)
FILES = [
    "README.md",
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_3/Sample_Game_3_events.json",
    "data/Sample_Game_3/Sample_Game_3_metadata.xml",
    "data/Sample_Game_3/Sample_Game_3_tracking.txt",
    "documentation/events-definitions.pdf",
]
GAMES = [1, 2, 3]
PITCH = (105.0, 68.0)   # README and the game 3 metadata
FPS = 25
H, A, D, X = 0, 1, 2, 3

CONTROL = {"SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST"}
STOP_LOST = {"OFFSIDE", "FORCED", "HAND BALL", "END HALF"}


def download():
    for rel in FILES:
        path = os.path.join(SRC, *rel.split("/"))
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", rel)
        req = urllib.request.Request(RAW + rel, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(path + ".part", path)
    print("done")


# -- reading ---------------------------------------------------------------------------------------

def _game_dir(g):
    return os.path.join(DATA, "Sample_Game_%d" % g)


def _read_csv_tracking(path):
    """Metrica CSV tracking (games 1, 2): 3 header rows, then Period, Frame, Time, x/y pairs."""
    with open(path, encoding="utf-8") as fh:
        for _ in range(2):
            fh.readline()
        cols = fh.readline().strip().split(",")
    names = [cols[i] for i in range(3, len(cols), 2)]
    arr = pd.read_csv(path, skiprows=3, header=None).to_numpy(np.float64)
    xy = arr[:, 3:3 + 2 * len(names)].reshape(len(arr), len(names), 2)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), names, xy


def _tracking_csv(g):
    d = _game_dir(g)
    per, frame, names_h, xy_h = _read_csv_tracking(
        os.path.join(d, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % g))
    per_a, frame_a, names_a, xy_a = _read_csv_tracking(
        os.path.join(d, "Sample_Game_%d_RawTrackingData_Away_Team.csv" % g))
    if not np.array_equal(frame, frame_a):
        raise ValueError("home/away tracking frames differ in game %d" % g)
    bh, ba = names_h.index("Ball"), names_a.index("Ball")
    ph = [i for i, nm in enumerate(names_h) if nm != "Ball"]
    pa = [i for i, nm in enumerate(names_a) if nm != "Ball"]
    P = np.concatenate([xy_h[:, ph], xy_a[:, pa]], axis=1)
    team = np.array([0] * len(ph) + [1] * len(pa), np.int8)
    return {"frame": frame, "period": per, "P": P, "ball": xy_h[:, bh], "team": team,
            "gk": None, "players": [names_h[i] for i in ph] + [names_a[i] for i in pa]}


def _tracking_epts(g):
    """EPTS (game 3): one DataFormatSpecification per substitution window, 22 player slots each."""
    d = _game_dir(g)
    meta = ET.parse(os.path.join(d, "Sample_Game_%d_metadata.xml" % g)).getroot()
    home_id = meta.find(".//Score").get("idLocalTeam")
    pinfo = {}
    for pl in meta.iter("Player"):
        params = {p.findtext("Name"): p.findtext("Value") for p in pl.iter("ProviderParameter")}
        pinfo[pl.get("id")] = (0 if pl.get("teamId") == home_id else 1,
                               params.get("position_type") == "Goalkeeper")
    chan = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    gp = {p.findtext("Name"): p.findtext("Value") for p in meta.find(".//ProviderGlobalParameters")}
    with open(os.path.join(d, "Sample_Game_%d_tracking.txt" % g), encoding="utf-8") as fh:
        txt = fh.read().replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(txt), header=None).to_numpy(np.float64)
    del txt
    frame = arr[:, 0].astype(int)
    pids = list(dict.fromkeys(chan[c.get("id")] for c in meta.iter("PlayerChannel")))
    col = {p: j for j, p in enumerate(pids)}
    P = np.full((len(arr), len(pids), 2), np.nan)
    for spec in meta.iter("DataFormatSpecification"):
        f0, f1 = int(spec.get("startFrame")), int(spec.get("endFrame"))
        rows = (frame >= f0) & (frame <= f1)
        refs = [chan[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")][::2]
        for k, pid in enumerate(refs):
            P[rows, col[pid]] = arr[rows][:, 1 + 2 * k:3 + 2 * k]
    nb = 1 + 2 * len(refs)
    period = np.where(frame <= int(gp["first_half_end"]), 1, 2)
    return {"frame": frame, "period": period, "P": P, "ball": arr[:, nb:nb + 2],
            "team": np.array([pinfo[p][0] for p in pids], np.int8),
            "gk": np.array([pinfo[p][1] for p in pids], bool), "players": pids}


def tracking(g):
    return _tracking_epts(g) if g == 3 else _tracking_csv(g)


def events(g):
    """Event table: team (0 home / 1 away), type, subtype tokens, period, start, end, order."""
    d = _game_dir(g)
    if g == 3:
        meta = ET.parse(os.path.join(d, "Sample_Game_3_metadata.xml")).getroot()
        home_id = meta.find(".//Score").get("idLocalTeam")
        raw = json.load(open(os.path.join(d, "Sample_Game_3_events.json"), encoding="utf-8"))["data"]
        rows = []
        for e in raw:
            st = e["subtypes"]
            st = [] if st is None else [st] if isinstance(st, dict) else st
            rows.append((0 if e["team"]["id"] == home_id else 1, e["type"]["name"],
                         "-".join(s["name"] for s in st), e["period"], e["start"]["frame"],
                         e["end"]["frame"] if e["end"]["frame"] is not None else e["start"]["frame"]))
        ev = pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])
    else:
        raw = pd.read_csv(os.path.join(d, "Sample_Game_%d_RawEventsData.csv" % g))
        ev = pd.DataFrame({"team": np.where(raw["Team"] == "Home", 0, 1), "type": raw["Type"],
                           "subtype": raw["Subtype"].fillna(""), "period": raw["Period"],
                           "start": raw["Start Frame"], "end": raw["End Frame"]})
    ev["end"] = np.maximum(ev["end"].astype(int), ev["start"].astype(int))
    ev["start"] = ev["start"].astype(int)
    ev["tokens"] = [frozenset(s.split("-")) if s else frozenset() for s in ev["subtype"]]
    ev["order"] = np.arange(len(ev))
    return ev.sort_values(["start", "order"], kind="stable").reset_index(drop=True)


# -- labels ----------------------------------------------------------------------------------------

def _restart(e):
    if e.type == "SET PIECE":
        return "RETAKEN" not in e.tokens
    return "GOAL KICK" in e.tokens


def _trigger(e):
    """Frame at which this event makes the ball dead, or None."""
    t = e.tokens
    if e.type == "BALL OUT":
        return e.end
    if e.type == "SHOT" and ("OUT" in t or "GOAL" in t):
        return e.end
    if e.type == "FAULT RECEIVED" or (e.type == "CHALLENGE" and "FAULT" in t):
        return e.start
    if e.type in ("BALL LOST", "PASS") and t & STOP_LOST:
        return e.end
    return None


def period_labels(ev, f0, f1, control=CONTROL):
    """H/A/D per frame f0..f1 (inclusive) of one period, from that period's events."""
    n = f1 - f0 + 1
    dead = []
    alive, dead_from, last_end = False, f0, f0
    info = {"restart_without_trigger": 0, "trigger_without_restart": 0}
    for e in ev.itertuples(index=False):
        if e.type == "CARD" or (e.type == "SET PIECE" and "RETAKEN" in e.tokens):
            continue
        if not alive and e.start < dead_from:
            continue                  # event recorded before a trigger it follows in the file
        if alive and _restart(e):
            info["restart_without_trigger"] += 1
            dead.append((min(last_end, e.start), e.start))
        elif not alive:
            trig = _trigger(e)
            if trig is not None and not _restart(e):
                continue              # second whistle-type event of the same stoppage
            if not _restart(e):
                key = "restart_by_" + e.type.replace(" ", "_").lower()
                info[key] = info.get(key, 0) + 1
            dead.append((dead_from, e.start))
        alive = True
        last_end = max(last_end, e.end)
        trig = _trigger(e)
        if trig is not None:
            alive, dead_from = False, max(trig, e.start)
    if alive:
        info["trigger_without_restart"] += 1   # no END HALF: dead after the last action
        dead_from = last_end
    dead.append((dead_from, f1 + 1))
    # team of the last controlling event
    ctl = ev[ev["type"].isin(control)]
    t = np.full(n, -1, np.int64)
    idx = np.clip(ctl["start"].to_numpy() - f0, 0, n - 1)
    np.maximum.at(t, idx, np.arange(len(ctl)))   # the later event wins within a frame
    t = np.maximum.accumulate(t)
    team = np.where(t >= 0, ctl["team"].to_numpy()[np.maximum(t, 0)], -1)
    lab = np.where(team == 0, H, np.where(team == 1, A, D)).astype(np.int8)
    for a, b in dead:
        lab[max(a, f0) - f0:max(min(b, f1 + 1), f0) - f0] = D
    info["n_dead"] = len(dead)
    return lab, info


def _find_gk(P, team):
    """Deepest player of each team (median x over the half) when the metadata has no positions."""
    gk = np.zeros(P.shape[1], bool)
    with np.errstate(invalid="ignore"):
        mx = np.nanmedian(P[:, :, 0], axis=0)
    present = np.isfinite(P[:, :, 0]).mean(0) > 0.2
    for k in (0, 1):
        m = (team == k) & present
        if not m.any():
            continue
        cand = np.flatnonzero(m)
        left = np.nanmean(mx[cand]) < 0.5
        gk[cand[np.argmin(mx[cand]) if left else np.argmax(mx[cand])]] = True
    return gk


def _speed(xy_m):
    v = np.full(xy_m.shape[:-1], np.nan, np.float32)
    v[1:-1] = np.linalg.norm(xy_m[2:] - xy_m[:-2], axis=-1) * FPS / 2.0
    return v


def parse(g, refresh=False, control=CONTROL):
    """dfl.parse()-style dict for sample game g: pitch, h{1,2}_players/_ball/_team/_gk/_labels/_n0."""
    os.makedirs(CACHE, exist_ok=True)
    out_path = os.path.join(CACHE, "game%d.npz" % g)
    if os.path.isfile(out_path) and not refresh and control is CONTROL:
        return dict(np.load(out_path, allow_pickle=True))
    tr, ev = tracking(g), events(g)
    L, W = PITCH
    result = {"pitch": np.array(PITCH, np.float32)}
    for si, per in enumerate((1, 2)):
        rows = np.flatnonzero(tr["period"] == per)
        fr = tr["frame"][rows]
        f0, f1 = int(fr[0]), int(fr[-1])
        if not np.array_equal(fr, np.arange(f0, f1 + 1)):
            raise ValueError("non-contiguous frames in game %d period %d" % (g, per))
        Pxy = tr["P"][rows]
        n, s = Pxy.shape[:2]
        P = np.full((n, s, 3), np.nan, np.float32)
        P[:, :, :2] = Pxy
        P[:, :, 2] = _speed(Pxy * np.array([L, W]))
        B = np.full((n, 4), np.nan, np.float32)
        B[:, :2] = tr["ball"][rows]
        B[:, 2] = _speed(tr["ball"][rows] * np.array([L, W]))
        lab, info = period_labels(ev[ev["period"] == per], f0, f1, control)
        tag = "h%d" % (si + 1)
        result[tag + "_players"] = P
        result[tag + "_ball"] = B
        result[tag + "_team"] = tr["team"]
        result[tag + "_gk"] = tr["gk"] if tr["gk"] is not None else _find_gk(Pxy, tr["team"])
        result[tag + "_labels"] = lab
        result[tag + "_n0"] = np.array([f0])
        result[tag + "_info"] = np.array(json.dumps(info))
    if control is CONTROL:
        np.savez(out_path, **result)
    return result


def merge_short(lab, tau_s):
    """Duration-based variant: a spell shorter than tau_s that directly follows a spell of the other
    team (no D between) is not a change of possession and is given to that team. Spells that start
    after a stoppage are kept (the restart names a clear possessor)."""
    if tau_s <= 0:
        return lab
    out = lab.copy()
    sp = spells(lab)
    for i in range(1, len(sp)):
        a, b, k = sp[i]
        pa, pb, _ = sp[i - 1]
        if b - a < tau_s * FPS and pb == a and out[pa] in (H, A) and out[pa] != k:
            out[a:b] = out[pa]
    return out


def merge_uncontrolled(lab, ev, f0, keep_s=1.0):
    """The challenge's reading of 'controlled' possession applied to the events: a spell that directly
    follows the other team's spell (no D between) is a change of possession only if the new team shows
    control inside it -- a completed PASS or a SHOT, or the ball kept for keep_s (a BALL LOST or CARRY
    starting keep_s after the spell starts, or a CARRY lasting keep_s). Otherwise (a lone RECOVERY
    touch, a header or block lost at once) the spell goes back to the team that had the ball. Time-based
    on purpose: only game 3 has CARRY events, and a short carry is no more control than none.
    ev holds this period's events, f0 its first frame."""
    out = lab.copy()
    st = ev["start"].to_numpy() - f0
    en = ev["end"].to_numpy() - f0
    tm = ev["team"].to_numpy()
    ty = ev["type"].to_numpy()
    keep = keep_s * FPS
    held = np.isin(ty, ("BALL LOST", "CARRY"))
    sp = spells(lab)
    for i in range(1, len(sp)):
        a, b, k = sp[i]
        pa, pb, _ = sp[i - 1]
        if pb != a or out[pa] not in (H, A) or out[pa] == k:
            continue
        inside = (st >= a) & (st < b) & (tm == k)
        ctrl = (np.isin(ty[inside], ("PASS", "SHOT")).any() or np.any(st[inside & held] >= a + keep)
                or np.any((en - st)[inside & (ty == "CARRY")] >= keep))
        if not ctrl:
            out[a:b] = out[pa]
    return out


CONVENTIONS = ("events", "tau1", "tau2", "tau3", "ctrl")


def convention(name, lab, ev=None, f0=None):
    """Ground truth of one half under a possession convention (see merge_short / merge_uncontrolled)."""
    if name == "events":
        return lab
    if name.startswith("tau"):
        return merge_short(lab, int(name[3:]))
    if name == "ctrl":
        return merge_uncontrolled(lab, ev, f0)
    raise ValueError(name)


def gt_halves(g, name):
    """Both halves of sample game g under convention name."""
    d = parse(g)
    ev = events(g) if name == "ctrl" else None
    return [convention(name, d["h%d_labels" % h], None if ev is None else ev[ev["period"] == h],
                       int(d["h%d_n0" % h][0])) for h in (1, 2)]


# -- statistics ------------------------------------------------------------------------------------

def spells(lab):
    """[(start, stop, code)] maximal H or A runs of one half (the official possession spells)."""
    breaks = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(lab)]))
    return [(int(a), int(b), int(lab[a])) for a, b in zip(bounds[:-1], bounds[1:]) if lab[a] in (H, A)]


def spell_stats(halves):
    """Possession-spell statistics over a list of per-half label arrays."""
    lens, dsegs, n_alive, n_dead = [], 0, 0, 0
    for lab in halves:
        lens += [(b - a) / FPS for a, b, _ in spells(lab)]
        breaks = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate(([0], breaks))
        dsegs += int(np.sum(lab[starts] == D))
        n_alive += int(np.sum((lab == H) | (lab == A)))
        n_dead += int(np.sum(lab == D))
    lens = np.array(lens)
    return {"n_spells": len(lens), "median_s": float(np.median(lens)), "mean_s": float(lens.mean()),
            "lt2s": float(np.mean(lens < 2)), "lt1s": float(np.mean(lens < 1)),
            "p25_s": float(np.percentile(lens, 25)), "p75_s": float(np.percentile(lens, 75)),
            "n_dead_segs": dsegs, "dead_share": n_dead / max(1, n_alive + n_dead),
            "alive_min": n_alive / FPS / 60}


def stats():
    fmt = lambda s: "  ".join("%s=%s" % (k, ("%.3f" % v) if isinstance(v, float) else v) for k, v in s.items())
    for g in GAMES:
        d = parse(g)
        seen = np.mean(np.concatenate([np.isfinite(d["h%d_ball" % h][:, 0]) for h in (1, 2)]))
        print("game %d events  %s  ball seen %.3f  info %s %s" % (
            g, fmt(spell_stats([d["h1_labels"], d["h2_labels"]])), seen, d["h1_info"], d["h2_info"]))
    for c in CONVENTIONS:
        print("3 games %-6s %s" % (c, fmt(spell_stats([lab for g in GAMES for lab in gt_halves(g, c)]))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmedian over absent substitutes
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        download()
    elif cmd == "stats":
        stats()
