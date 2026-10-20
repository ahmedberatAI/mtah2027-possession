"""WASB on native-resolution 512x288 crops that contain the ATD ball (random placement).
Separates 'model cannot see a 2-3 px ball after downscaling' from a pipeline bug. AGGREGATE output."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import av, numpy as np, pandas as pd
import atd, wasb
from ball_wasb_eval import ball_pixels
sys.stdout.reconfigure(encoding="utf-8")
px, py = ball_pixels()
rng = np.random.default_rng(1)
starts = []
for _, a, b in atd.HALVES:
    starts += sorted(rng.integers(a + 100, b - 100, 10).tolist())
model = wasb.load()
c = av.open(atd.paths()["video"]); st = c.streams.video[0]; tb = st.time_base
rows = []
for s in starts:
    c.seek(int(max(0, s - 60) / 25 / tb), stream=st, backward=True)
    fr = {}
    for f in c.decode(st):
        i = int(round(float(f.pts * tb) * 25))
        if i > s + 30: break
        if i >= s - 1: fr[i] = f.to_ndarray(format="rgb24")
    for i in range(s, s + 30):
        if not all(j in fr for j in (i - 1, i, i + 1)) or not np.isfinite(px[i]): continue
        if not (20 < px[i] < 1900 and 20 < py[i] < 1060): continue
        x0 = int(np.clip(px[i] - rng.uniform(60, 452), 0, 1920 - 512)); y0 = int(np.clip(py[i] - rng.uniform(40, 248), 0, 1080 - 288))
        crops = [fr[j][y0:y0 + 288, x0:x0 + 512] for j in (i - 1, i, i + 1)]
        hm = wasb.heatmaps(model, wasb.preprocess(crops)[None])[0, 1]
        k = int(np.argmax(hm)); yy, xx = divmod(k, 512)
        rows.append((np.hypot(x0 + xx - px[i], y0 + yy - py[i]), float(hm[yy, xx])))
r = np.array(rows)
print("native crops evaluated:", len(r))
print("pixel error: p25 %.1f p50 %.1f p75 %.1f p90 %.1f" % tuple(np.percentile(r[:, 0], [25, 50, 75, 90])))
for thr in (0.1, 0.3, 0.5, 0.7):
    m = r[:, 1] >= thr
    print("score>=%.1f: %3.0f%% of crops; of those within 5/10/20 px: %3.0f%% %3.0f%% %3.0f%%" % (thr, 100 * m.mean(), 100 * (r[m, 0] < 5).mean() if m.any() else 0, 100 * (r[m, 0] < 10).mean() if m.any() else 0, 100 * (r[m, 0] < 20).mean() if m.any() else 0))
print("overall within 10 px (any score): %.0f%%" % (100 * (r[:, 0] < 10).mean()))
