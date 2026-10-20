"""Self-supervised ball detector for the tactical video (local GPU). AGGREGATE output only.

  python src/ball_yolo.py dataset   # ATD ball -> pixel pseudo-labels, native-res 640 crops
  python src/ball_yolo.py train     # fine-tune YOLO11s (ultralytics, AGPL-3.0)
  python src/ball_yolo.py infer     # every graded frame, 3x2 overlapping 640 tiles -> top-k dets
  python src/ball_yolo.py eval      # recall / precision vs ATD on held-out frames, gap coverage

Pseudo-labels: ATD ball (normalised pitch) -> pixels through the per-frame homography, moved 6 px
up (ATD = ground-contact point; measured vs WASB peaks), dropped where the pixel track jumps.
Train on frames < VAL_START, validate on frames >= VAL_START (end of 2nd half).
Nothing here is meant to be looked at as images; ultralytics' own plots are disabled.
"""
import os
import sys
import time

import av
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import atd  # noqa: E402

WORK = os.path.join(ROOT, "cache", "ball_yolo")
DS = os.path.join(WORK, "ds")
VAL_START = 172000
N_TRAIN, N_VAL = 16000, 2000
BOX = 14
TILE = 640
TILES = [(x, y) for y in (0, 440) for x in (0, 640, 1280)]
sys.stdout.reconfigure(encoding="utf-8")


def ball_pixels():
    """ATD ball -> tactical-video pixels via the per-frame homography (NaN where unseen)."""
    import pandas as pd
    B = atd.load("tactical")["ball"][:, :2].astype(np.float64)
    tab = pd.read_csv(atd.paths()["homography"])
    H = tab[["h{}{}".format(r, c) for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)
    den = H[:, 2, 0] * B[:, 0] + H[:, 2, 1] * B[:, 1] + H[:, 2, 2]
    return ((H[:, 0, 0] * B[:, 0] + H[:, 0, 1] * B[:, 1] + H[:, 0, 2]) / den,
            (H[:, 1, 0] * B[:, 0] + H[:, 1, 1] * B[:, 1] + H[:, 1, 2]) / den)


def pseudo_labels():
    px, py = ball_pixels()
    py = py - 6.0
    seen = np.isfinite(px)
    inside = seen & (px > 8) & (px < 1912) & (py > 8) & (py < 1072)
    jump = np.zeros(len(px))
    jump[1:] = np.hypot(np.diff(px), np.diff(py))
    good = inside & ~(np.nan_to_num(jump, nan=0) > 60) & atd.graded_mask()
    return px, py, good


def iter_frames(frame_set=None, start=0):
    """Sequential decode; yields (index, rgb ndarray) for wanted indices (all if None)."""
    c = av.open(atd.paths()["video"])
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    tb = st.time_base
    if start > 0:
        c.seek(int(max(0, start - 50) / 25 / tb), stream=st, backward=True)
    for f in c.decode(st):
        i = int(round(float(f.pts * tb) * 25))
        if i < start or (frame_set is not None and i not in frame_set):
            continue
        yield i, f.to_ndarray(format="bgr24")
    c.close()


def dataset():
    px, py, good = pseudo_labels()
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(good)
    tr = rng.choice(cand[cand < VAL_START - 250], N_TRAIN, replace=False)
    va = rng.choice(cand[cand >= VAL_START], N_VAL, replace=False)
    want = {int(f): "train" for f in tr}
    want.update({int(f): "val" for f in va})
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(DS, sub), exist_ok=True)
    t = time.time()
    n = 0
    for i, img in iter_frames(set(want)):
        split = want[i]
        cx, cy = px[i], py[i]
        x0 = int(np.clip(cx - rng.uniform(48, TILE - 48), 0, 1920 - TILE))
        y0 = int(np.clip(cy - rng.uniform(48, TILE - 48), 0, 1080 - TILE))
        cv2.imwrite(os.path.join(DS, "images", split, "%d.jpg" % i), img[y0:y0 + TILE, x0:x0 + TILE],
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        with open(os.path.join(DS, "labels", split, "%d.txt" % i), "w") as fh:
            fh.write("0 %.6f %.6f %.6f %.6f\n" % ((cx - x0) / TILE, (cy - y0) / TILE, BOX / TILE, BOX / TILE))
        if split == "train" and rng.random() < 0.35:
            for _ in range(10):   # a negative tile far from the (visible) ball
                nx, ny = TILES[rng.integers(len(TILES))]
                if not (nx - 30 <= cx <= nx + TILE + 30 and ny - 30 <= cy <= ny + TILE + 30):
                    cv2.imwrite(os.path.join(DS, "images", "train", "%d_neg.jpg" % i),
                                img[ny:ny + TILE, nx:nx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 93])
                    open(os.path.join(DS, "labels", "train", "%d_neg.txt" % i), "w").close()
                    break
        n += 1
    with open(os.path.join(DS, "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n" % DS.replace("\\", "/"))
    print("dataset: %d labelled crops in %.0fs" % (n, time.time() - t))


def train(epochs=40):
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")
    model.train(data=os.path.join(DS, "data.yaml"), imgsz=TILE, epochs=epochs, batch=48, workers=6,
                seed=0, deterministic=True, mosaic=0.5, degrees=0, fliplr=0.5, scale=0.3,
                hsv_h=0.01, project=WORK, name="train", exist_ok=True, plots=False, verbose=False)


def infer(weights=None, conf=0.05, topk=3, batch_frames=32):
    from ultralytics import YOLO
    import torch
    weights = weights or os.path.join(WORK, "train", "weights", "best.pt")
    model = YOLO(weights)
    graded = atd.graded_mask()
    n = atd.N_FRAMES["tactical"]
    det = np.full((n, topk, 3), np.nan, np.float32)
    frames, idx = [], []
    t = time.time()

    def flush():
        tiles = [im[y:y + TILE, x:x + TILE] for im in frames for x, y in TILES]
        res = model.predict(tiles, imgsz=TILE, conf=conf, half=True, verbose=False, device=0)
        for k, fi in enumerate(idx):
            cands = []
            for j, (x0, y0) in enumerate(TILES):
                r = res[k * len(TILES) + j]
                if len(r.boxes):
                    for (bx, by, _, _), c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        cands.append((float(c), bx + x0, by + y0))
            cands.sort(reverse=True)
            kept = []
            for c, x, y in cands:     # merge duplicates from overlapping tiles
                if all(np.hypot(x - kx, y - ky) > 12 for _, kx, ky in kept):
                    kept.append((c, x, y))
                if len(kept) == topk:
                    break
            for r_, (c, x, y) in enumerate(kept):
                det[fi, r_] = (x, y, c)
        frames.clear()
        idx.clear()

    done = 0
    for i, img in iter_frames(start=atd.HALVES[0][1]):
        if i > atd.HALVES[1][2]:
            break
        if not graded[i]:
            continue
        frames.append(img)
        idx.append(i)
        if len(frames) == batch_frames:
            flush()
            done += batch_frames
            if done % 20000 < batch_frames:
                print("  %d frames, %.0f fps" % (done, done / (time.time() - t)), flush=True)
    if frames:
        flush()
    np.savez(os.path.join(WORK, "dets.npz"), det=det)
    print("inference done in %.0f min" % ((time.time() - t) / 60))


def evaluate():
    px, py, good = pseudo_labels()
    det = np.load(os.path.join(WORK, "dets.npz"))["det"]
    graded = atd.graded_mask()
    top = det[:, 0]
    has = np.isfinite(top[:, 0])
    err = np.hypot(top[:, 0] - px, top[:, 1] - py)
    print("== held-out frames (>= %d) with a pseudo-label" % VAL_START)
    for name, m0 in (("held-out", good & (np.arange(len(px)) >= VAL_START)), ("all graded", good)):
        for thr in (0.1, 0.25, 0.5):
            m = m0 & has & (top[:, 2] >= thr)
            print("%-10s conf>=%.2f: fires %.0f%% | within 8/15/30 px %.0f%% / %.0f%% / %.0f%% | median err %.1f" % (
                name, thr, 100 * m.sum() / m0.sum(), 100 * np.mean(err[m] < 8), 100 * np.mean(err[m] < 15),
                100 * np.mean(err[m] < 30), np.nanmedian(err[m])))
    atd_missing = graded & ~np.isfinite(px)
    for thr in (0.25, 0.5):
        m = atd_missing & has & (top[:, 2] >= thr)
        print("ATD-missing graded frames with a detection conf>=%.2f: %.1f%% (%d of %d)" % (
            thr, 100 * m.sum() / atd_missing.sum(), m.sum(), atd_missing.sum()))


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    cmd = sys.argv[1]
    {"dataset": dataset, "train": train, "infer": infer, "eval": evaluate}[cmd]()
