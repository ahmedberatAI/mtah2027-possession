# %% [markdown]
# # Ball detector for the tactical video (Colab GPU)
#
# Self-supervised: the provided ATD sees the ball on ~57% of frames. Projecting those positions
# through the per-frame homography gives pixel pseudo-labels; a small detector fine-tuned on them
# then looks for the ball on the ~43% of frames where the ATD has none.
#
# **Data policy:** run this only in your own Colab runtime on your own Drive copy of the bundle.
# Do not enable / use Gemini or any hosted-LLM feature on these files, do not share outputs
# publicly (frame-level derived data). The output CSV stays in your Drive and is consumed by
# `src/` locally.
#
# Expected Drive layout (adjust `ROOT`):
# `MyDrive/mtah2027/tactical/BAR-ATH_tactical.mp4`, `..._atd.txt`, `..._atd.xml`, `..._homography.csv`

# %%
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/mtah2027/tactical'
WORK = '/content/work'
import os
os.makedirs(WORK, exist_ok=True)
!nvidia-smi
!pip -q install ultralytics==8.3.40 av

# %%
import numpy as np, pandas as pd, av, cv2, random, shutil, glob
STEM = 'BAR-ATH_tactical'
!cp "{ROOT}/{STEM}.mp4" {WORK}/video.mp4

def load_atd_ball(path):
    out = np.full((199319, 2), np.nan, np.float32)
    with open(path) as fh:
        for line in fh:
            fid, _, ball = line.rstrip('\n').split(':')
            bx, by, _ = ball.split(',')
            if bx != 'NaN':
                out[int(fid) - 1] = (float(bx), float(by))
    return out

ball = load_atd_ball(f'{ROOT}/{STEM}_atd.txt')
H = pd.read_csv(f'{ROOT}/{STEM}_homography.csv')
Hm = H[[f'h{r}{c}' for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)

def project(Hf, x, y):
    d = Hf[2, 0] * x + Hf[2, 1] * y + Hf[2, 2]
    return (Hf[0, 0] * x + Hf[0, 1] * y + Hf[0, 2]) / d, (Hf[1, 0] * x + Hf[1, 1] * y + Hf[1, 2]) / d

seen = np.isfinite(ball[:, 0])
px = np.full(len(ball), np.nan); py = np.full(len(ball), np.nan)
for i in np.flatnonzero(seen):
    px[i], py[i] = project(Hm[i], ball[i, 0], ball[i, 1])
# ATD gives the ball's ground-contact point; the visible ball centre is ~6 px higher at 1080p
# (measured locally: median dy -5.4 px, mean -6.7 px vs the pretrained WASB peak)
py = py - 6.0
inside = seen & (px > 8) & (px < 1912) & (py > 8) & (py < 1072)
# drop kinematically implausible pseudo-labels (pixel jumps between consecutive frames)
jump = np.full(len(px), 0.0); jump[1:] = np.hypot(np.diff(px), np.diff(py))
good = inside & ~(jump > 60)
print('pseudo-labelled frames:', good.sum())

# %%
# Build a tiled YOLO dataset: 640x640 crops containing the pseudo-labelled ball (+ random negatives)
DS = f'{WORK}/ds'
for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
    os.makedirs(f'{DS}/{sub}', exist_ok=True)
rng = np.random.default_rng(0)
cand = np.flatnonzero(good)
# stride sampling across the whole match; second half frames > 150000 held out for validation
train_f = rng.choice(cand[cand < 150000], 12000, replace=False)
val_f = rng.choice(cand[cand >= 150000], 1500, replace=False)
want = {int(f): 'train' for f in train_f}; want.update({int(f): 'val' for f in val_f})
BOX = 14  # px, ball box size at 1080p in the tactical view
container = av.open(f'{WORK}/video.mp4')
for idx, frame in enumerate(container.decode(video=0)):
    if idx not in want:
        continue
    img = frame.to_ndarray(format='bgr24')
    split = want[idx]
    cx, cy = px[idx], py[idx]
    x0 = int(np.clip(cx - rng.integers(64, 576), 0, 1920 - 640)); y0 = int(np.clip(cy - rng.integers(64, 576), 0, 1080 - 640))
    crop = img[y0:y0 + 640, x0:x0 + 640]
    cv2.imwrite(f'{DS}/images/{split}/{idx}.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(f'{DS}/labels/{split}/{idx}.txt', 'w') as fh:
        fh.write('0 %.6f %.6f %.6f %.6f\n' % ((cx - x0) / 640, (cy - y0) / 640, BOX / 640, BOX / 640))
    if split == 'train' and rng.random() < 0.3:   # negative crop away from the ball
        nx, ny = int(rng.integers(0, 1280)), int(rng.integers(0, 440))
        if abs(nx + 320 - cx) > 400 or abs(ny + 320 - cy) > 400:
            cv2.imwrite(f'{DS}/images/train/{idx}_neg.jpg', img[ny:ny + 640, nx:nx + 640])
            open(f'{DS}/labels/train/{idx}_neg.txt', 'w').close()
container.close()
open(f'{DS}/data.yaml', 'w').write(f'path: {DS}\ntrain: images/train\nval: images/val\nnames: [ball]\n')

# %%
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
model.train(data=f'{DS}/data.yaml', imgsz=640, epochs=30, batch=64, workers=8, seed=0,
            deterministic=True, mosaic=0.5, degrees=0, fliplr=0.5, project=WORK, name='ball')

# %%
# Inference on every frame: 2x3 tiles of 640 covering 1920x1080 (overlapping), keep the best box.
best = YOLO(f'{WORK}/ball/weights/best.pt')
tiles = [(x, y) for y in (0, 440) for x in (0, 640, 1280)]
rows = []
container = av.open(f'{WORK}/video.mp4')
batch, meta = [], []
def flush():
    if not batch:
        return
    res = best.predict(batch, imgsz=640, conf=0.05, verbose=False, half=True)
    per = {}
    for (fi, x0, y0), r in zip(meta, res):
        for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if fi not in per or c > per[fi][2]:
                per[fi] = (b[0] + x0, b[1] + y0, float(c))
    for fi, v in per.items():
        rows.append((fi, *v))
    batch.clear(); meta.clear()
for idx, frame in enumerate(container.decode(video=0)):
    img = frame.to_ndarray(format='bgr24')
    for x0, y0 in tiles:
        batch.append(img[y0:y0 + 640, x0:x0 + 640]); meta.append((idx, x0, y0))
    if len(batch) >= 192:
        flush()
flush()
det = pd.DataFrame(rows, columns=['frame', 'px', 'py', 'conf']).sort_values('frame')
det.to_csv(f'{ROOT}/../ball_detections_yolo.csv', index=False)
print(len(det), 'frames with a detection')
