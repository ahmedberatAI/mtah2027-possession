"""SkillCorner open data (github.com/SkillCorner/opendata, MIT; 20 A-League 2024/25 matches of broadcast
tracking) -> arrays in the same layout as dfl.parse(), at 25 fps, with per-frame H/A/D labels.

Labels come from the "phases of play" file: phases are only defined while the ball is in play, so frames
outside every phase are D; inside a phase the label is its team_in_possession (home -> H, away -> A).
SkillCorner possessions are built from player possessions (a player controlling the ball), and the
resulting spell statistics are close to the public ground truth (median 7.6 s, 13% of spells < 2 s; GT
7.56 s, 2/17) -- a third possession convention next to DFL's flag and Metrica's events.

Also per frame: whether the broadcast view shows the pitch (image corners projected; valid in 94% of alive and
25% of dead frames, like BAR-ATH's broadcast homography validity) and how many players are detected on screen.
Tracking is 10 fps; positions are linearly interpolated to 25 fps (a gap stays NaN). Players come with
extrapolated positions when off screen (kept: the degradation in sim.py decides what is missing); the
ball is NaN where SkillCorner has no position. Coordinates: metres centred on the pitch -> normalised 0-1.

  .venv/Scripts/python.exe -I src/skillcorner.py stats      # parse + cache all matches, aggregate stats
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.join(ROOT, "external_data", "skillcorner", "data", "matches")
CACHE = os.path.join(ROOT, "cache", "skillcorner")
FPS_IN, FPS = 10, 25
H, A, D = 0, 1, 2


def matches():
    return sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))


def _interp(t_out, t_in, v):
    """Linear interpolation of v (n_in, ...) sampled at t_in onto t_out; NaN where a neighbour is NaN."""
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, len(t_in) - 2)
    w = (t_out - t_in[idx]) / (t_in[idx + 1] - t_in[idx])
    shape = (-1,) + (1,) * (v.ndim - 1)
    return v[idx] * (1 - w.reshape(shape)) + v[idx + 1] * w.reshape(shape)


def parse(match, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    out_path = os.path.join(CACHE, "%s.npz" % match)
    if os.path.isfile(out_path) and not refresh:
        d = dict(np.load(out_path, allow_pickle=True))
        if "h1_valid" in d:
            return d
    d = os.path.join(DATA, match)
    meta = json.load(open(os.path.join(d, "%s_match.json" % match), encoding="utf-8"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    players = meta["players"]
    pid = [p["id"] for p in players]
    slot = {p: i for i, p in enumerate(pid)}
    team = np.array([0 if p["team_id"] == home else 1 for p in players], np.int8)
    gk = np.array([p["player_role"]["name"] == "Goalkeeper" for p in players], bool)
    rows = {1: [], 2: []}
    with open(os.path.join(d, "%s_tracking_extrapolated.jsonl" % match), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["period"] in rows:
                rows[r["period"]].append(r)
    ph = pd.read_csv(os.path.join(d, "%s_phases_of_play.csv" % match))
    result = {"pitch": np.array([L, W], np.float32)}
    for per in (1, 2):
        rr = rows[per]
        f10 = np.array([r["frame"] for r in rr], np.int64)
        n10 = len(rr)
        P10 = np.full((n10, len(pid), 2), np.nan, np.float32)
        B10 = np.full((n10, 3), np.nan, np.float32)
        valid10 = np.array([r["image_corners_projection"]["x_top_left"] is not None for r in rr])
        ndet10 = np.array([sum(1 for q in r["player_data"] if q["is_detected"]) for r in rr], np.int16)
        for i, r in enumerate(rr):
            for q in r["player_data"]:
                j = slot.get(q["player_id"])
                if j is not None and q["x"] is not None:
                    P10[i, j] = ((q["x"] + L / 2) / L, (q["y"] + W / 2) / W)
            b = r["ball_data"]
            if b["x"] is not None:
                B10[i] = ((b["x"] + L / 2) / L, (b["y"] + W / 2) / W, b["z"] if b["z"] is not None else np.nan)
        lab10 = np.full(n10, D, np.int8)
        for p in ph[ph["period"] == per].itertuples():
            a, b = np.searchsorted(f10, p.frame_start), np.searchsorted(f10, p.frame_end, side="right")
            lab10[a:b] = H if p.team_in_possession_id == home else A
        t10 = (f10 - f10[0]) / FPS_IN
        n25 = int(np.floor(t10[-1] * FPS)) + 1
        t25 = np.arange(n25) / FPS
        P = np.full((n25, len(pid), 3), np.nan, np.float32)
        P[:, :, :2] = _interp(t25, t10, P10)
        B = np.full((n25, 4), np.nan, np.float32)
        B[:, :2] = _interp(t25, t10, B10[:, :2])
        B[:, 3] = _interp(t25, t10, B10[:, 2])
        near = np.clip(np.round(t25 * FPS_IN).astype(np.int64), 0, n10 - 1)
        tag = "h%d" % per
        result[tag + "_players"] = P
        result[tag + "_ball"] = B
        result[tag + "_team"] = team
        result[tag + "_gk"] = gk
        result[tag + "_labels"] = lab10[near]
        result[tag + "_f0"] = np.array([f10[0]])
        result[tag + "_valid"] = valid10[near]      # broadcast view shows the pitch (cf. BAR-ATH b_valid)
        result[tag + "_ndet"] = ndet10[near]        # players detected on screen
    np.savez(out_path, **result)
    return result


def _parse_job(m):
    d = parse(m)
    out = []
    for h in (1, 2):
        lab = d["h%d_labels" % h]
        seen = np.isfinite(d["h%d_ball" % h][:, 0])
        npl = np.isfinite(d["h%d_players" % h][:, :, 0]).sum(1)
        out.append((len(lab), np.mean(lab == D), np.mean(seen[lab < 2]), np.mean(seen[lab == D]), np.median(npl)))
    return m, out


def stats():
    from multiprocessing import Pool
    import metrica_sample as ms
    with Pool(6) as pool:
        res = pool.map(_parse_job, matches())
    labs = []
    for m in matches():
        d = parse(m)
        labs += [d["h1_labels"], d["h2_labels"]]
    s = ms.spell_stats(labs)
    n = len(labs) // 2
    a = np.array([x for _, o in res for x in o])
    print("SkillCorner %d matches at 25 fps: frames/half %.0f, D share %.3f, ball present alive %.3f / dead %.3f, "
          "players present median %.0f" % (n, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(),
                                           np.median(a[:, 4])))
    print("spells/match %.1f  per alive min %.2f  median %.2f s  mean %.2f s  <2s %.3f  <1s %.3f  dead segs/match %.0f" % (
        s["n_spells"] / n, s["n_spells"] / s["alive_min"], s["median_s"], s["mean_s"], s["lt2s"], s["lt1s"],
        s["n_dead_segs"] / n))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
