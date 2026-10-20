"""Parse the Metrica FIFA/EPTS ATD export into numpy arrays and cache them locally.

Raw line: "<frame_id>:<slot>;<slot>;...;<slot>:<ball>" with every group "x,y,speed_kph" and NaN
for a missing observation. frame_id counts from 1, so video_frame = frame_id - 1.

Coordinates are normalised 0-1; ``to_metres`` scales them with the pitch size from games.py
(105.3 x 68.0 m). Speed is in km/h (the XML channel unit says kph).

The cache lives in ./cache (gitignored). Nothing here prints data values.
"""
import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache")
PITCH = (105.3, 68.0)
FPS = 25
N_FRAMES = {"tactical": 199319, "broadcast": 164488}
HALVES = [(1, 11185, 90651), (2, 110808, 190197)]  # graded video frames, inclusive
KICKOFF_2ND = 113742


def paths(source="tactical"):
    folder = os.path.join(DATA_DIR, source)
    stem = "BAR-ATH_{}".format(source)
    return {
        "meta": os.path.join(folder, stem + "_atd.xml"),
        "raw": os.path.join(folder, stem + "_atd.txt"),
        "homography": os.path.join(folder, stem + "_homography.csv"),
        "smart_tagging": os.path.join(folder, stem + "_smart_tagging.xml"),
        "video": os.path.join(folder, stem + ".mp4"),
    }


def read_meta(meta_path):
    """Slot table: team (0 home / 1 away) and the provider position parameters per slot."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_names = [t.findtext("Name") for t in root.iter("Team")]
    rows = []
    for player in root.iter("Player"):
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        rows.append({
            "slot_name": player.findtext("Name"),
            "team": 0 if player.get("teamId") == team_ids[0] else 1,
            "position_type": params.get("position_type"),
            "position_index": params.get("position_index"),
            "position_x": params.get("position_x"),
            "position_y": params.get("position_y"),
        })
    glob = {p.findtext("Name"): p.findtext("Value")
            for p in root.find(".//ProviderGlobalParameters")}
    return pd.DataFrame(rows), team_names, glob


def parse_raw(raw_path, n_slots):
    """(frame_id, players[n, slots, 3], ball[n, 3]) as float32, NaN where missing."""
    with open(raw_path, "r") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    n_cols = 1 + 3 * n_slots + 3
    table = pd.read_csv(io.StringIO(text), header=None, names=list(range(n_cols)),
                        dtype=np.float64, na_values=["NaN"], engine="c")
    values = table.values
    frame_id = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3).astype(np.float32)
    ball = values[:, 1 + 3 * n_slots:].astype(np.float32)
    return frame_id, players, ball


def load(source="tactical", refresh=False):
    """Dict of arrays for one source, indexed by video frame (row i = video frame i)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_atd.npz".format(source))
    p = paths(source)
    slots, team_names, glob = read_meta(p["meta"])
    if refresh or not os.path.isfile(cache):
        frame_id, players, ball = parse_raw(p["raw"], len(slots))
        n = N_FRAMES[source]
        if len(frame_id) != n or not np.array_equal(frame_id, np.arange(1, n + 1)):
            raise ValueError("unexpected frame ids in {}".format(p["raw"]))
        np.savez(cache, players=players, ball=ball)
    arrays = np.load(cache)
    return {
        "players": arrays["players"],      # (n, slots, 3): x, y normalised; speed km/h
        "ball": arrays["ball"],            # (n, 3)
        "team": slots["team"].values.astype(np.int8),
        "slots": slots,
        "team_names": team_names,
        "global": glob,
    }


def to_metres(xy):
    """Normalised (..., 2) -> metres, origin at a pitch corner."""
    out = np.asarray(xy, dtype=np.float32).copy()
    out[..., 0] *= PITCH[0]
    out[..., 1] *= PITCH[1]
    return out


def graded_mask(n=N_FRAMES["tactical"]):
    mask = np.zeros(n, dtype=bool)
    for _, a, b in HALVES:
        mask[a:b + 1] = True
    return mask


def half_of_frame(n=N_FRAMES["tactical"]):
    half = np.zeros(n, dtype=np.int8)
    for h, a, b in HALVES:
        half[a:b + 1] = h
    return half
