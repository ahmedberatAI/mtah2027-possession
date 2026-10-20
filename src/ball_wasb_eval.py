"""Zero-shot check of the pretrained WASB soccer detector on the tactical video. AGGREGATE output.

Samples W windows of K consecutive frames spread over both halves, runs WASB on triplets, and
compares the heatmap peak with the ATD ball projected to pixels through the homography (where the
ATD sees the ball). Also reports how often WASB fires confidently where the ATD has no ball.
"""
import os
import sys
import time

import av
import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import evaluate as ev  # noqa: E402
import wasb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def ball_pixels():
    d = atd.load("tactical")
    B = d["ball"][:, :2].astype(np.float64)
    tab = pd.read_csv(atd.paths()["homography"])
    H = tab[["h{}{}".format(r, c) for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)
    den = H[:, 2, 0] * B[:, 0] + H[:, 2, 1] * B[:, 1] + H[:, 2, 2]
    px = (H[:, 0, 0] * B[:, 0] + H[:, 0, 1] * B[:, 1] + H[:, 0, 2]) / den
    py = (H[:, 1, 0] * B[:, 0] + H[:, 1, 1] * B[:, 1] + H[:, 1, 2]) / den
    return px, py


def read_window(container, start, k):
    """Frames start-1 .. start+k (inclusive) as 512x288 RGB arrays, by seeking near start."""
    stream = container.streams.video[0]
    tb = stream.time_base
    container.seek(int(max(0, start - 60) / 25 / tb), stream=stream, backward=True, any_frame=False)
    out = {}
    for fr in container.decode(stream):
        idx = int(round(float(fr.pts * tb) * 25))
        if idx > start + k:
            break
        if idx >= start - 1:
            out[idx] = cv2.resize(fr.to_ndarray(format="rgb24"), (wasb.IN_W, wasb.IN_H),
                                  interpolation=cv2.INTER_AREA)
    return out


if __name__ == "__main__":
    W, K = int(sys.argv[1]) if len(sys.argv) > 1 else 20, 30
    px, py = ball_pixels()
    seen = np.isfinite(px)
    lab = ev.to_codes(ev.pm.read_label_csv(os.path.join(os.path.dirname(HERE), "submissions",
                                                        "baseline_v5.csv"))["label"].values)
    rng = np.random.default_rng(0)
    starts = []
    for _, a, b in atd.HALVES:
        starts += sorted(rng.integers(a + 100, b - 100, W // 2).tolist())
    model = wasb.load()
    container = av.open(atd.paths()["video"])
    rows = []
    t0 = time.time()
    for s in starts:
        frames = read_window(container, s, K)
        idx = [i for i in range(s, s + K) if all(j in frames for j in (i - 1, i, i + 1))]
        batch = np.stack([wasb.preprocess([frames[i - 1], frames[i], frames[i + 1]]) for i in idx])
        for i, hm in zip(idx, wasb.heatmaps(model, batch)):
            x, y, sc = wasb.peak(hm[1])
            rows.append((i, x, y, sc))
    print("decoded+inferred %d frames in %.0fs" % (len(rows), time.time() - t0))
    r = pd.DataFrame(rows, columns=["f", "x", "y", "s"])
    r["seen"] = seen[r.f]
    r["lab"] = lab[r.f]
    inside = r.seen & (px[r.f] > 0) & (px[r.f] < 1920) & (py[r.f] > 0) & (py[r.f] < 1080)
    r["err"] = np.where(inside, np.hypot(r.x - px[r.f], r.y - py[r.f]), np.nan)
    a = r[inside]
    print("frames with ATD ball in image: %d" % len(a))
    print("pixel error vs ATD: p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f" % tuple(np.percentile(a.err, [25, 50, 75, 90])))
    for thr in (0.1, 0.3, 0.5, 0.7):
        m = a.s >= thr
        print("score>=%.1f: fires on %.0f%% of ATD-ball frames; within 10/20/40 px: %.0f%% / %.0f%% / %.0f%%" % (
            thr, 100 * m.mean(), 100 * (a.err[m] < 10).mean(), 100 * (a.err[m] < 20).mean(), 100 * (a.err[m] < 40).mean()))
    for name, m in (("ATD ball missing, v5 alive", ~r.seen & (r.lab < 2)), ("ATD ball missing, v5 dead", ~r.seen & (r.lab == 2))):
        sub = r[m]
        if len(sub):
            print("%s: %d frames; score>=0.5 on %.0f%%; >=0.3 on %.0f%%" % (name, len(sub), 100 * (sub.s >= 0.5).mean(), 100 * (sub.s >= 0.3).mean()))
    print("score distribution where ATD sees ball: p10 %.2f p50 %.2f p90 %.2f" % tuple(np.percentile(a.s, [10, 50, 90])))
