"""Pilot evaluation (round 7, research R-6): PASS / DRIVE events of the 2023 SoccerNet ball-action-spotting winner on
the BAR-ATH broadcast video (colab/bas_pilot.py -> cache/video/bas_raw_fold<k>.npz), attributed to a team through
the player nearest to the ball at the event (tactical ATD players, merged ball; broadcast -> tactical frame lag of
src/broadcast.py). AGGREGATE output only.

Public window (17 spells, information only): attribution accuracy against the ground-truth team at the event, the
same for v7's labels at the event frames, and -- the decisive number -- who is right where the event's team and v7
disagree. Full match (no labels): events per alive minute, unattributed share, agreement with v7.

  python -I src/bas_eval.py [fold]
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import bench  # noqa: E402
import broadcast  # noqa: E402
import evaluate as ev  # noqa: E402
import model_v7  # noqa: E402

VIDEO = os.path.join(bench.ROOT, "cache", "video")
MAX_DIST = 3.0          # m: the nearest player must be this close to the ball to attribute the event


def nearest_team(blocks, t):
    """(team 0/1 or -1, distance m) of the player nearest to the ball at tactical frame t (ball within +-4 frames)."""
    for b in blocks:
        if b["a"] <= t <= b["b"]:
            L, W = b["pitch"]
            for dt in (0, -1, 1, -2, 2, -3, 3, -4, 4):
                i = t - b["a"] + dt
                if not 0 <= i < len(b["B"]) or not np.isfinite(b["B"][i, 0]):
                    continue
                d = np.hypot((b["P"][i, :, 0] - b["B"][i, 0]) * L, (b["P"][i, :, 1] - b["B"][i, 1]) * W)
                if not np.isfinite(d).any():
                    return -1, np.nan
                j = int(np.nanargmin(d))
                return (int(b["team"][j]) if d[j] <= MAX_DIST else -1), float(d[j])
    return -1, np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    z = np.load(os.path.join(VIDEO, "bas_raw_fold%d.npz" % fold))
    t2b = broadcast.tactical_to_broadcast()
    b2t = np.full(atd.N_FRAMES["broadcast"] + 1, -1, np.int64)
    ok = t2b >= 0
    b2t[t2b[ok]] = np.flatnonzero(ok)
    from make_submission import merged_features
    f = merged_features()
    v7 = model_v7.predict(f)
    gt = ev.public_gt()
    graded = atd.graded_mask(len(v7))
    blocks = bench.blocks("barath")
    alive_min = np.sum(graded & (v7 < 2)) / 1500.0
    win = np.flatnonzero(gt < 2)
    print("public window: %d alive GT frames (%.1f min) | v7 frame team acc there %.3f" % (
        len(win), len(win) / 1500.0, np.mean(v7[win] == gt[win])), flush=True)
    for name in ("pass", "drive"):
        fr, conf = z["%s_frames" % name], z["%s_conf" % name]
        for thr in (0.2, 0.35, 0.5):
            sel = conf >= thr
            tt = b2t[np.clip(fr[sel], 0, len(b2t) - 1)]
            tt = tt[(tt >= 0) & graded[np.maximum(tt, 0)]]
            teams = np.array([nearest_team(blocks, int(t))[0] for t in tt])
            att = teams >= 0
            live = v7[tt] < 2
            agree = np.mean(teams[att & live] == v7[tt][att & live]) if (att & live).any() else np.nan
            inw = att & (gt[tt] < 2)
            line = ("%-5s >= %.2f | full match %4d events (%.1f / alive min), attributed %.2f, agree with v7 %.3f" % (
                name.upper(), thr, len(tt), len(tt) / alive_min, att.mean() if len(att) else np.nan, agree))
            if inw.any():
                e_ok = teams[inw] == gt[tt][inw]
                v_ok = v7[tt][inw] == gt[tt][inw]
                dis = teams[inw] != v7[tt][inw]
                line += (" | public: %d events, event-team acc %.3f, v7 acc at them %.3f, disagreements %d "
                         "(event right %d, v7 right %d)" % (inw.sum(), e_ok.mean(), v_ok.mean(), dis.sum(),
                                                            int((e_ok & dis).sum()), int((v_ok & dis).sum())))
            print(line, flush=True)


if __name__ == "__main__":
    main()
