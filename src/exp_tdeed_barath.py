"""Round 22, step 1 analysis: the SoccerNet 2025 Team-BAS baseline (T-DEED with a team head, colab/tdeed_barath.py) on
the BAR-ATH broadcast, read against v14 without any label. AGGREGATE output only -- no frames, no per-event rows.

Events (cache/video/tdeed_barath/barath/broadcast/results_spotting.json) are moved to the tactical timeline with the
broadcast lag of src/broadcast.py and kept where the broadcast shows live play (b_live) in a graded half.
  1. "left" / "right" -> H / A per half: the devkit does not define the sides, so each half takes the mapping under
     which PASS / DRIVE events (confidence >= 0.5) agree with v14 inside v14's long possessions (>= 2 s from any
     change); the agreement of the chosen mapping is the team head's accuracy on BAR-ATH up to v14's own errors.
  2. Video-implied possession changes: consecutive PASS / DRIVE / TACKLE / BLOCK events of different teams (both
     >= the threshold, <= 6 s apart); how many fall near a v14 change (+-1 s), how many do not (possible missed short
     possessions), and their overlap with v14's verifier candidates (src/exp_video_feas.py).
  python -I src/exp_tdeed_barath.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import broadcast  # noqa: E402
import model_v14  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
EVENTS = os.path.join(bench.ROOT, "cache", "video", "tdeed_barath", "barath", "broadcast", "results_spotting.json")
POSS = ("PASS", "DRIVE", "HIGH PASS", "CROSS", "HEADER", "SHOT", "PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK")
FPS = 25


def changes(lab):
    """Frames where a live team-to-team change happens (H -> A or A -> H, ignoring dead ball in between)."""
    live = np.flatnonzero(lab < 2)
    flips = live[1:][lab[live[1:]] != lab[live[:-1]]]
    return flips


def main():
    from make_submission import merged_features
    import exp_video_feas as vf
    f = merged_features()
    base, cands = vf.candidates(f)
    final = model_v14.predict(f)
    n = len(final)
    graded = atd.graded_mask(n)
    half = atd.half_of_frame(n)
    live_b = np.asarray(f["b_live"], bool)
    t2b = broadcast.tactical_to_broadcast()
    b2t = np.full(atd.N_FRAMES["broadcast"] + 1, -1, np.int64)
    ok = t2b >= 0
    b2t[t2b[ok]] = np.flatnonzero(ok)
    with open(EVENTS, encoding="utf-8") as fh:
        raw = json.load(fh)["predictions"]
    ev = []
    for e in raw:
        b = int(round(e["position"] * FPS / 1000.0))
        t = b2t[min(b, len(b2t) - 1)]
        if t >= 0 and graded[t] and live_b[t]:
            ev.append((int(t), e["label"], e.get("team"), float(e["confidence"])))
    alive_min = np.sum(graded & (final < 2)) / (FPS * 60.0)
    print("events on live broadcast frames in graded halves: %d of %d (alive minutes %.1f)" % (len(ev), len(raw), alive_min))
    for thr in (0.2, 0.5):
        cnt = {}
        for t, lab, team, c in ev:
            if c >= thr:
                cnt[lab] = cnt.get(lab, 0) + 1
        print("  conf >= %.1f per alive min: %s" % (thr, ", ".join("%s %.2f" % (k, v / alive_min)
                                                                  for k, v in sorted(cnt.items()))))

    # 1. left / right -> H / A per half, from v14's long possessions
    flips = changes(final)
    dist = np.full(n, 10 ** 6)
    if len(flips):
        idx = np.searchsorted(flips, np.arange(n))
        lo = np.abs(np.arange(n) - flips[np.clip(idx - 1, 0, len(flips) - 1)])
        hi = np.abs(np.arange(n) - flips[np.clip(idx, 0, len(flips) - 1)])
        dist = np.minimum(lo, hi)
    clear = (final < 2) & (dist >= 2 * FPS)
    mapping = {}
    for h in (1, 2):
        sel = [(t, team) for t, lab, team, c in ev if half[t] == h and lab in ("PASS", "DRIVE") and c >= 0.5 and clear[t]]
        if not sel:
            print("half %d: no clear PASS / DRIVE events" % h)
            continue
        left_is_h = np.mean([(team == "left") == (final[t] == 0) for t, team in sel])
        mapping[h] = {"left": 0, "right": 1} if left_is_h >= 0.5 else {"left": 1, "right": 0}
        print("half %d: %d clear PASS / DRIVE events; agreement with v14 if left = H %.3f, if left = A %.3f -> %s" % (
            h, len(sel), left_is_h, 1 - left_is_h, "left = H" if left_is_h >= 0.5 else "left = A"))
    if len(mapping) < 2:
        return
    team_of = lambda t, side: mapping[half[t]].get(side, -1)  # noqa: E731
    for thr in (0.2, 0.5, 0.8):
        sel = [(t, team_of(t, team)) for t, lab, team, c in ev if lab in POSS and c >= thr and final[t] < 2]
        if sel:
            agree_all = np.mean([k == final[t] for t, k in sel])
            agree_clear = np.mean([k == final[t] for t, k in sel if clear[t]]) if any(clear[t] for t, k in sel) else np.nan
            near = [(t, k) for t, k in sel if not clear[t]]
            agree_near = np.mean([k == final[t] for t, k in near]) if near else np.nan
            print("team events conf >= %.1f: %d | agreement with v14 all %.3f, inside long possessions %.3f, "
                  "within 2 s of a v14 change %.3f (%d events)" % (thr, len(sel), agree_all, agree_clear, agree_near,
                                                                   len(near)))

    # 2. video-implied possession changes vs v14's changes and candidates
    for thr in (0.3, 0.5):
        seq = sorted((t, team_of(t, team)) for t, lab, team, c in ev if lab in POSS and c >= thr)
        vchg = [(a[0], b[0], b[1]) for a, b in zip(seq[:-1], seq[1:])
                if a[1] != b[1] and a[1] >= 0 and b[1] >= 0 and b[0] - a[0] <= 6 * FPS]
        if not vchg:
            continue
        mids = np.array([(a + b) // 2 for a, b, k in vchg])
        near_v14 = np.array([np.any(np.abs(flips - a) <= FPS) or np.any(np.abs(flips - b) <= FPS)
                             for a, b, k in vchg]) if len(flips) else np.zeros(len(vchg), bool)
        in_cand = np.array([any(s - 12 <= m <= e + 12 for s, e, typ, q in cands) for m in mids])
        print("video-implied changes (conf >= %.1f): %d (%.2f / alive min; v14 makes %.2f) | near a v14 change "
              "%.3f | not near one %d, of which inside a verifier candidate %d" % (
                  thr, len(vchg), len(vchg) / alive_min, len(flips) / alive_min, near_v14.mean(),
                  int((~near_v14).sum()), int((~near_v14 & in_cand).sum())))
    # candidates: does the video see the candidate team on the ball?
    for thr in (0.3, 0.5):
        rows = []
        for s, e, typ, q in cands:
            if not graded[min(s, n - 1)]:
                continue
            k = base[s:e].copy()
            team = int(np.bincount(k[k < 2], minlength=2).argmax()) if (k < 2).any() else -1
            cand_team = 1 - team if typ == "ins" else team
            evs = [team_of(t, side) for t, lab, side, c in ev if lab in POSS and c >= thr and s - 12 <= t <= e + 12]
            rows.append((typ, cand_team, evs))
        for typ in ("ins", "del"):
            r = [x for x in rows if x[0] == typ]
            seen = [x for x in r if x[2]]
            support = [np.mean([k == x[1] for k in x[2]]) for x in seen]
            print("  %s candidates (conf >= %.1f): %d, with a team event %d, mean share of events for the candidate "
                  "team %.3f" % (typ, thr, len(r), len(seen), np.mean(support) if support else np.nan))


if __name__ == "__main__":
    main()
