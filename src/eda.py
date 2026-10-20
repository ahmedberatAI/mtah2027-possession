"""Exploratory analysis of the provided assets. Prints AGGREGATE statistics only.

Sections
  A  tracking coverage and quality (ball, players, slots, speeds, positions)
  B  what the public 5-minute ground truth looks like against the tracking
  C  Smart Tagging: coverage, team tags, set pieces, agreement with the public GT
  D  homography: reprojection error and the camera as a proxy for the ball
  E  broadcast source

Run:  .venv/Scripts/python.exe src/eda.py > reports/eda_summary.txt
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "external", "more-than-a-hack-2027", "evaluation"))
import atd  # noqa: E402
import possession_metrics as pm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
GT_PATH = os.path.join(ROOT, "external", "more-than-a-hack-2027", "evaluation", "gt",
                       "BAR-ATH_possession_gt_public.csv")
L, W = atd.PITCH


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return "n=0"
    return "n={} mean={:.2f} ".format(len(x), x.mean()) + " ".join(
        "p{}={:.2f}".format(q, v) for q, v in zip(qs, np.percentile(x, qs)))


def runs_of(mask):
    """[(start, stop)] of True runs."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def nearest(players, ball, team):
    """Per frame: nearest distance (m), its team, nearest per team distances, 2nd nearest."""
    dx = (players[:, :, 0] - ball[:, None, 0]) * L
    dy = (players[:, :, 1] - ball[:, None, 1]) * W
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.where(np.isnan(dist), np.inf, dist)
    order = np.argsort(dist, axis=1)
    d1 = np.take_along_axis(dist, order[:, :1], 1)[:, 0]
    d2 = np.take_along_axis(dist, order[:, 1:2], 1)[:, 0]
    t1 = team[order[:, 0]]
    dh = np.where(team[None, :] == 0, dist, np.inf).min(1)
    da = np.where(team[None, :] == 1, dist, np.inf).min(1)
    return d1, t1, dh, da, d2


def section_a(d, graded, half):
    print("=" * 100)
    print("A. TRACKING COVERAGE AND QUALITY (tactical)")
    P, B, team = d["players"], d["ball"], d["team"]
    n = len(B)
    ball_seen = ~np.isnan(B[:, 0])
    print("frames {}  graded {}  slots {} (home {}, away {})".format(
        n, graded.sum(), P.shape[1], (team == 0).sum(), (team == 1).sum()))
    print("ball present: all {:.1%}  graded {:.1%}  half1 {:.1%}  half2 {:.1%}".format(
        ball_seen.mean(), ball_seen[graded].mean(), ball_seen[half == 1].mean(),
        ball_seen[half == 2].mean()))

    gaps = [(a, b) for a, b in runs_of(~ball_seen & graded)]
    lengths = np.array([b - a for a, b in gaps])
    print("\nball gaps inside graded play: {} gaps, total {} frames".format(len(gaps), lengths.sum()))
    for lo, hi in ((1, 2), (2, 5), (5, 13), (13, 25), (25, 50), (50, 125), (125, 250), (250, 750),
                   (750, 10 ** 9)):
        sel = (lengths >= lo) & (lengths < hi)
        print("  gap {:>4}-{:<10} frames: {:5d} gaps  {:6.1%} of missing frames".format(
            lo, hi if hi < 10 ** 9 else "inf", sel.sum(), lengths[sel].sum() / lengths.sum()))
    print("  gap length (frames): " + pct(lengths))

    present = ~np.isnan(P[:, :, 0])
    n_home = present[:, team == 0].sum(1)
    n_away = present[:, team == 1].sum(1)
    print("\ntracked players per graded frame: total " + pct((n_home + n_away)[graded]))
    print("  home " + pct(n_home[graded]))
    print("  away " + pct(n_away[graded]))
    print("  frames with >11 home: {:.2%}  >11 away: {:.2%}  <9 home: {:.2%}  <9 away: {:.2%}".format(
        (n_home[graded] > 11).mean(), (n_away[graded] > 11).mean(), (n_home[graded] < 9).mean(),
        (n_away[graded] < 9).mean()))

    seg_len = []
    n_segs = []
    for s in range(P.shape[1]):
        r = runs_of(present[:, s] & graded)
        n_segs.append(len(r))
        seg_len += [b - a for a, b in r]
    print("\nslot tracklets inside graded play: {} tracklets; per slot ".format(sum(n_segs)) +
          pct(n_segs))
    print("  tracklet length (s): " + pct(np.array(seg_len) / atd.FPS))
    occ = present[graded].mean(0)
    print("  slot occupancy over graded frames: " + pct(occ))

    pos_types = d["slots"]["position_type"].fillna("None")
    print("  position_type values per team: home {}  away {}".format(
        pos_types[team == 0].value_counts().to_dict(), pos_types[team == 1].value_counts().to_dict()))

    spd = P[:, :, 2][graded]
    print("\nplayer speed km/h (graded, observed): " + pct(spd[~np.isnan(spd)]))
    bs = B[graded, 2]
    print("ball speed km/h (graded, observed):   " + pct(bs[~np.isnan(bs)], (5, 25, 50, 75, 95, 99)))
    bx, by = B[graded, 0], B[graded, 1]
    ok = ~np.isnan(bx)
    out = (bx[ok] < 0) | (bx[ok] > 1) | (by[ok] < 0) | (by[ok] > 1)
    print("ball outside [0,1]^2 on {:.2%} of graded ball-present frames".format(out.mean()))
    print("ball x (m): " + pct(bx[ok] * L, (0.5, 5, 50, 95, 99.5)))
    print("ball y (m): " + pct(by[ok] * W, (0.5, 5, 50, 95, 99.5)))

    px = P[:, :, 0][graded]
    py = P[:, :, 1][graded]
    okp = ~np.isnan(px)
    outp = (px[okp] < -0.01) | (px[okp] > 1.01) | (py[okp] < -0.01) | (py[okp] > 1.01)
    print("player positions outside the pitch (>1% margin): {:.2%}".format(outp.mean()))

    # ball jumps: implausible displacement between consecutive observed frames
    bxy = atd.to_metres(B[:, :2])
    step = np.linalg.norm(np.diff(bxy, axis=0), axis=1)
    step = step[graded[1:] & graded[:-1]]
    step = step[np.isfinite(step)]
    print("ball step between consecutive observed frames (m): " + pct(step, (50, 90, 99, 99.9)))
    print("  steps > 2 m (> 180 km/h): {} ({:.3%})".format((step > 2).sum(), (step > 2).mean()))

    d1, t1, dh, da, d2 = nearest(P, B, team)
    print("\nnearest player to ball (m), graded ball-present: " +
          pct(d1[graded & ball_seen], (5, 25, 50, 75, 90, 95)))
    return d1, t1, dh, da


def load_gt():
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    return gt


def section_b(d, gt, d1, t1, dh, da):
    print("=" * 100)
    print("B. PUBLIC GROUND TRUTH (5 min) AGAINST THE TRACKING")
    P, B, team = d["players"], d["ball"], d["team"]
    win = gt != pm.X
    ball_seen = ~np.isnan(B[:, 0])
    for k, name in enumerate(("H", "A", "D")):
        m = win & (gt == k)
        print("GT {}: {:5d} frames  ball present {:.1%}".format(name, m.sum(), ball_seen[m].mean()))
    alive = win & (gt < 2) & ball_seen
    team_gt = gt[alive]
    agree = (t1[alive] == team_gt)
    print("\nnearest-player team == GT team on alive ball-present frames: {:.1%}".format(agree.mean()))
    for thr in (1, 1.5, 2, 3, 5, 10):
        m = d1[alive] <= thr
        print("  nearest dist <= {:>4} m: coverage {:5.1%}  agreement {:5.1%}".format(
            thr, m.mean(), agree[m].mean() if m.any() else float("nan")))
    print("nearest distance on GT-alive frames: " + pct(d1[alive], (10, 25, 50, 75, 90)))
    dead = win & (gt == 2) & ball_seen
    print("nearest distance on GT-dead frames:  " + pct(d1[dead], (10, 25, 50, 75, 90)))

    spd = P[:, :, 2]
    mean_spd = np.nanmean(spd, axis=1)
    for k, name in enumerate(("H", "A", "D")):
        m = win & (gt == k)
        print("mean player speed in GT {}: {}".format(name, pct(mean_spd[m], (10, 50, 90))))
    bs = B[:, 2]
    for k, name in enumerate(("H", "A", "D")):
        m = win & (gt == k) & ball_seen
        print("ball speed in GT {}: {}".format(name, pct(bs[m], (10, 50, 90))))
    out = (B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1)
    for k, name in enumerate(("H", "A", "D")):
        m = win & (gt == k) & ball_seen
        print("ball outside pitch in GT {}: {:.1%}".format(name, out[m].mean()))

    # transitions: what the tracking shows around each GT label change
    runs = pm.valid_runs(gt)
    trans = pm.transitions(gt, runs)
    print("\nGT transitions: {}".format(len(trans)))
    kinds = {}
    for f, a, b in trans:
        kinds.setdefault(pm.LABELS[a] + ">" + pm.LABELS[b], []).append(f)
    for kname, frames in sorted(kinds.items()):
        seen = [ball_seen[max(0, f - 12):f + 13].mean() for f in frames]
        print("  {}: n={}  ball visible in +-0.5 s window: mean {:.0%}".format(
            kname, len(frames), np.mean(seen)))

    # how often does the nearest team flip on alive frames (raw jitter)
    tt = np.where(ball_seen & win & (gt < 2), t1, -1)
    seq = tt[tt >= 0]
    print("raw nearest-team flips on GT-alive ball-present frames: {} over {} frames".format(
        int((seq[1:] != seq[:-1]).sum()), len(seq)))


def load_tags(path):
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        rows.append({"code": inst.findtext("code"), "start": float(inst.findtext("start")),
                     "end": float(inst.findtext("end")), **tags})
    df = pd.DataFrame(rows)
    df["sf"] = (df["start"] * atd.FPS).round().astype(int)
    df["ef"] = (df["end"] * atd.FPS).round().astype(int)
    return df


def section_c(gt, graded):
    print("=" * 100)
    print("C. SMART TAGGING")
    df = load_tags(atd.paths()["smart_tagging"])
    df["dur"] = (df["ef"] - df["sf"]) / atd.FPS
    print("instances {}, columns {}".format(len(df), [c for c in df.columns if c not in
                                                       ("start", "end", "sf", "ef")]))
    g = df.groupby("code")["dur"]
    print(pd.DataFrame({"n": g.size(), "mean_s": g.mean().round(1), "median_s": g.median().round(1),
                        "max_s": g.max().round(1)}).sort_values("n", ascending=False).to_string())
    print("\nTeam tag: {}".format(df["Team"].value_counts(dropna=False).to_dict()))
    sp = df[df["code"].str.upper() == "SET PIECES"]
    print("set pieces by Type: {}".format(sp["Type"].value_counts(dropna=False).to_dict()))
    print("set pieces Team by Type:\n{}".format(
        pd.crosstab(sp["Type"].fillna("?"), sp["Team"].fillna("?")).to_string()))

    n = len(gt)
    cover = np.zeros(n, dtype=np.int16)
    for _, r in df.iterrows():
        cover[max(0, r.sf):min(n, r.ef)] += 1
    print("\ngraded frames covered by >=1 tag: {:.1%}; by >=2: {:.1%}".format(
        (cover[graded] >= 1).mean(), (cover[graded] >= 2).mean()))
    print("tags starting in each half: {}".format(df["Half"].value_counts().to_dict()))

    # agreement with the public GT, per code (only intervals overlapping the public window)
    win = gt != pm.X
    team_map = {"Barcelona": 0, "Atletic Club": 1}
    print("\nagreement with public GT inside the 5-min window (frames of tag interval):")
    rows = []
    for code, sub in df.groupby("code"):
        tot = same = dead = 0
        for _, r in sub.iterrows():
            a, b = max(0, r.sf), min(n, r.ef)
            m = win[a:b]
            if not m.any():
                continue
            g = gt[a:b][m]
            tot += len(g)
            dead += int((g == 2).sum())
            if r.get("Team") in team_map:
                same += int((g == team_map[r["Team"]]).sum())
        if tot:
            rows.append((code, tot, same / tot, dead / tot))
    for code, tot, s, dd in sorted(rows, key=lambda x: -x[1]):
        print("  {:<32} frames {:5d}  GT==tag team {:5.1%}  GT dead {:5.1%}".format(code, tot, s, dd))

    # set-piece starts vs GT D->alive transitions in the public window
    runs = pm.valid_runs(gt)
    trans = [t for t in pm.transitions(gt, runs) if t[1] == pm.D]
    print("\nGT D->alive transitions in window: {}".format(len(trans)))
    starts, ends, hits, team_ok = [], [], 0, 0
    for f, a, b in trans:
        cand = sp[(sp["sf"] <= f + 125) & (sp["ef"] >= f - 125)]
        if len(cand):
            hits += 1
            r = cand.iloc[(cand["sf"] - f).abs().argmin()]
            starts.append(r.sf - f)
            ends.append(r.ef - f)
            team_ok += int(team_map.get(r.get("Team"), -1) == b)
    print("  restarts with a set-piece tag within +-5 s: {}/{}; tag team == restarting team: {}".format(
        hits, len(trans), team_ok))
    if starts:
        print("  tag start offset vs restart (frames): " + pct(starts, (0, 50, 100)))
        print("  tag end offset vs restart (frames):   " + pct(ends, (0, 50, 100)))
    # alive->D transitions vs tag boundaries
    return df


def section_d(d, gt, graded):
    print("=" * 100)
    print("D. HOMOGRAPHY (tactical)")
    table = pd.read_csv(atd.paths()["homography"])
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    Hm = table[cols].values.reshape(-1, 3, 3)
    err = table["reprojection_error"].values
    valid = table["valid"].values.astype(bool)
    print("rows {} valid {:.1%}  reprojection error graded: {}".format(
        len(table), valid.mean(), pct(err[graded], (50, 90, 99))))
    # pixel -> pitch for the image centre and the bottom-centre
    Hinv = np.linalg.inv(Hm[valid])
    def unproj(px, py):
        v = np.stack([np.full(len(Hinv), px), np.full(len(Hinv), py), np.ones(len(Hinv))], 1)
        o = np.einsum("nij,nj->ni", Hinv, v)
        out = np.full((len(table), 2), np.nan)
        out[valid] = o[:, :2] / o[:, 2:3]
        return out
    centre = unproj(960, 540)
    B = d["ball"]
    ball_seen = ~np.isnan(B[:, 0])
    m = graded & ball_seen & valid
    dist = np.linalg.norm(atd.to_metres(centre[m]) - atd.to_metres(B[m, :2]), axis=1)
    print("image centre projected on pitch vs ball (m): " + pct(dist, (25, 50, 75, 90, 95)))
    dxs = (centre[m, 0] - B[m, 0]) * L
    print("  along-length error (m): " + pct(np.abs(dxs), (50, 90)))
    # camera motion: speed of the projected centre
    cm = atd.to_metres(centre)
    cam_speed = np.linalg.norm(np.diff(cm, axis=0), axis=1) * atd.FPS
    cam_speed = np.concatenate(([np.nan], cam_speed))
    win = gt != pm.X
    for k, name in enumerate(("H", "A", "D")):
        mm = win & (gt == k)
        print("camera-centre speed (m/s) in GT {}: {}".format(name, pct(cam_speed[mm], (10, 50, 90))))
    # zoom proxy: pitch-space width of the view at the image centre row
    left, right = unproj(0, 540), unproj(1919, 540)
    view_w = np.linalg.norm(atd.to_metres(right) - atd.to_metres(left), axis=1)
    print("view width at centre row (m), graded: " + pct(view_w[graded], (5, 50, 95)))
    for k, name in enumerate(("H", "A", "D")):
        mm = win & (gt == k)
        print("  view width in GT {}: {}".format(name, pct(view_w[mm], (10, 50, 90))))
    return centre


def section_e():
    print("=" * 100)
    print("E. BROADCAST")
    b = atd.load("broadcast")
    table = pd.read_csv(atd.paths("broadcast")["homography"])
    valid = table["valid"].values.astype(bool)
    ball_seen = ~np.isnan(b["ball"][:, 0])
    print("frames {}  homography valid {:.1%}  ball present {:.1%}  ball present & valid {:.1%}".format(
        len(valid), valid.mean(), ball_seen.mean(), (ball_seen & valid).mean()))
    inv_runs = runs_of(~valid)
    lens = np.array([b_ - a for a, b_ in inv_runs]) / atd.FPS
    print("invalid stretches: {}  duration (s): {}".format(len(inv_runs), pct(lens, (10, 50, 90, 100))))
    present = ~np.isnan(b["players"][:, :, 0])
    print("tracked players per frame (valid frames): " + pct(present[valid].sum(1)))


if __name__ == "__main__":
    d = atd.load("tactical")
    graded = atd.graded_mask()
    half = atd.half_of_frame()
    gt = load_gt()
    d1, t1, dh, da = section_a(d, graded, half)
    section_b(d, gt, d1, t1, dh, da)
    section_c(gt, graded)
    section_d(d, gt, graded)
    section_e()
