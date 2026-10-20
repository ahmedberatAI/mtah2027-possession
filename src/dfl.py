"""Open DFL / IDSSE (Bassek et al., CC BY 4.0) positions -> arrays in the same layout as atd.load().

Per match and game section: players (n, slots, 3) normalised 0-1 + speed m/s, ball (n, 3)
normalised + z (m), team per slot (0 home / 1 guest), goalkeeper slots, and labels
H/A/D from BallStatus (0 -> D) and BallPossession (1 -> H, 2 -> A).
"""
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE = os.path.join(ROOT, "external_data", "idsse")
CACHE = os.path.join(ROOT, "cache", "dfl")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def _file(kind, match):
    hits = glob.glob(os.path.join(IDSSE, "DFL_*{}*{}.xml".format(kind, match)))
    if not hits:
        raise IOError("missing {} file for {}".format(kind, match))
    return hits[0]


def match_info(match):
    root = ET.parse(_file("matchinformation", match)).getroot()
    teams = {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}
    players = {}
    for team in root.iter("Team"):
        for pl in team.iter("Player"):
            players[pl.get("PersonId")] = (team.get("TeamId"), pl.get("PlayingPosition"))
    env = root.find(".//Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    return teams, players, pitch


def parse(match, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    out_path = os.path.join(CACHE, "{}.npz".format(match))
    if os.path.isfile(out_path) and not refresh:
        return dict(np.load(out_path, allow_pickle=True))
    teams, players, pitch = match_info(match)
    L, W = pitch
    sets = {}
    for _, el in ET.iterparse(_file("positions_raw_observed", match), events=("end",)):
        if el.tag != "FrameSet":
            continue
        sec = el.get("GameSection")
        tid = el.get("TeamId")
        pid = el.get("PersonId")
        frames = el.findall("Frame")
        N = np.array([int(fr.get("N")) for fr in frames], np.int64)
        X = np.array([float(fr.get("X")) for fr in frames], np.float32)
        Y = np.array([float(fr.get("Y")) for fr in frames], np.float32)
        S = np.array([float(fr.get("S", "nan")) for fr in frames], np.float32)
        rec = {"N": N, "X": X, "Y": Y, "S": S, "team": tid}
        if tid and tid.upper() == "BALL":
            rec["Z"] = np.array([float(fr.get("Z", "nan")) for fr in frames], np.float32)
            rec["poss"] = np.array([int(fr.get("BallPossession", "0")) for fr in frames], np.int8)
            rec["status"] = np.array([int(fr.get("BallStatus", "0")) for fr in frames], np.int8)
        sets.setdefault(sec, {})[pid if tid.upper() != "BALL" else "BALL"] = rec
        el.clear()
    result = {"pitch": np.array(pitch, np.float32)}
    for si, sec in enumerate(sorted(sets, key=lambda s: 0 if s == "firstHalf" else 1)):
        ss = sets[sec]
        ball = ss["BALL"]
        n0, n1 = int(ball["N"].min()), int(ball["N"].max())
        n = n1 - n0 + 1
        pids = [p for p in ss if p != "BALL" and ss[p]["team"] in teams]
        P = np.full((n, len(pids), 3), np.nan, np.float32)
        team = np.zeros(len(pids), np.int8)
        gk = np.zeros(len(pids), bool)
        for j, pid in enumerate(pids):
            r = ss[pid]
            i = r["N"] - n0
            P[i, j, 0] = (r["X"] + L / 2) / L
            P[i, j, 1] = (r["Y"] + W / 2) / W
            P[i, j, 2] = r["S"]
            team[j] = 0 if teams[r["team"]] == "home" else 1
            gk[j] = players.get(pid, (None, ""))[1] == "TW"
        B = np.full((n, 4), np.nan, np.float32)
        lab = np.full(n, 3, np.int8)
        i = ball["N"] - n0
        B[i, 0] = (ball["X"] + L / 2) / L
        B[i, 1] = (ball["Y"] + W / 2) / W
        B[i, 2] = ball["S"]
        B[i, 3] = ball["Z"]
        l = np.where(ball["status"] == 0, 2, np.where(ball["poss"] == 1, 0, np.where(ball["poss"] == 2, 1, 3)))
        lab[i] = l
        tag = "h%d" % (si + 1)
        result[tag + "_players"] = P
        result[tag + "_ball"] = B
        result[tag + "_team"] = team
        result[tag + "_gk"] = gk
        result[tag + "_labels"] = lab
        result[tag + "_n0"] = np.array([n0])
    np.savez(out_path, **result)
    return result
