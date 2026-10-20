"""Systematic spatial/temporal offset between WASB peaks (native crops) and ATD ball pixels. AGGREGATE."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import av, numpy as np
import atd, wasb
from ball_wasb_eval import ball_pixels
sys.stdout.reconfigure(encoding="utf-8")
px, py = ball_pixels()
rng = np.random.default_rng(2)
starts = []
for _, a, b in atd.HALVES:
    starts += sorted(rng.integers(a + 100, b - 100, 10).tolist())
model = wasb.load()
c = av.open(atd.paths()["video"]); st = c.streams.video[0]; tb = st.time_base
print("first video pts (s):", float(st.start_time * tb) if st.start_time is not None else None)
D = {s: [] for s in range(-6, 7)}; vec = []
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
        x0 = int(np.clip(px[i] - 256, 0, 1920 - 512)); y0 = int(np.clip(py[i] - 144, 0, 1080 - 288))
        hm = wasb.heatmaps(model, wasb.preprocess([fr[j][y0:y0 + 288, x0:x0 + 512] for j in (i - 1, i, i + 1)])[None])[0, 1]
        k = int(np.argmax(hm)); yy, xx = divmod(k, 512); X, Y = x0 + xx + 0.5, y0 + yy + 0.5
        if hm[yy, xx] < 0.5: continue
        for sh in D:
            j = i + sh
            if np.isfinite(px[j]): D[sh].append(np.hypot(X - px[j], Y - py[j]))
        e = np.hypot(X - px[i], Y - py[i])
        if e < 40: vec.append((X - px[i], Y - py[i]))
for sh in D:
    a = np.array(D[sh]); print("ATD frame shift %+d: median err %.1f px, within 10 px %.0f%%" % (sh, np.median(a), 100 * np.mean(a < 10)))
v = np.array(vec); print("offset WASB - ATD (confident, <40 px): n %d  dx mean %.1f med %.1f | dy mean %.1f med %.1f" % (len(v), v[:, 0].mean(), np.median(v[:, 0]), v[:, 1].mean(), np.median(v[:, 1])))
