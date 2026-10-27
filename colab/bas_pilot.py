"""Pilot (round 7, research R-6; docs/RAPOR.md): the 2023 SoccerNet ball-action-spotting winner (R. Baikulov, MIT,
github.com/lRomul/ball-action-spotting; one fold model of its final experiment ball_finetune_long_004) on the BAR-ATH
broadcast video, read from the user's own Drive mounted on this VM. Runs on the Colab VM only (colab/run_bas.sh).

Frames: 1080p -> 1280x720 grayscale (the model's 720p SoccerNet input, padded to 1280x736 by its own processor);
the model reads 33-frame stacks at a 2-frame step, so feeding the even frames gives a PASS / DRIVE probability on
every even frame (12.5 fps, the model's native rate). Horizontal-flip TTA as the author's predictor.
Output: /content/work/bas_raw_fold<k>.npz (frame_indexes, raw float16 (n, 2)) and the author's post-processing
peaks; prints aggregates only, never frames.
"""
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

# path of the broadcast video on the mounted Drive (a shortcut to the shared match folder shows up under
# /content/drive/.shortcut-targets-by-id/<folder id>/broadcast/); set by the caller, not kept in the repository
SRC = os.environ.get("BAS_VIDEO", "/content/drive/MyDrive/broadcast/BAR-ATH_broadcast.mp4")
WORK = "/content/work"
BAS = "/content/bas"
EXP = "ball_finetune_long_004"
FOLD = int(os.environ.get("BAS_FOLD", "5"))
# the author's public weights (drive folder 1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz): fold -> (file, drive id); the score
# in the name is each fold's own held-out game, so the folds are not ranked by it
MODELS = {0: ("model-006-0.864002.pth", "1uJEYltwSjOJmvfDY5SzZh7RL5j3-kyM_"),
          1: ("model-006-0.862339.pth", "1u6TdSDayZzYatsd2LJWSq_YBet4h1HIT"),
          2: ("model-006-0.758246.pth", "1gjAQSHhLFzvitz61ABxnn-_zCW8xrnpx"),
          3: ("model-006-0.887217.pth", "1d7uybKF1_3bFPE5llvuExtWk0oV-0bZy"),
          4: ("model-006-0.869269.pth", "1ahhX3sEE6DcuNlIJKOZcFTu-9gi3VxZ1"),
          5: ("model-006-0.901643.pth", "1raYTlfD1vGf7OVstK1K1m01aazcC3jKC"),
          6: ("model-006-0.897602.pth", "1vywJ-1ZFXexD9ReN5SSltF2OLFHGV-og")}


def weights(fold):
    import gdown
    name, fid = MODELS[fold]
    path = "ball_action/experiments/%s/fold_%d/%s" % (EXP, fold, name)
    score = float(re.search(r"-(\d+(?:\.\d+)?)\.pth$", name).group(1))
    out = os.path.join(BAS, "data", path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.isfile(out):
        gdown.download(id=fid, output=out, quiet=True)
    print("model fold %d: %s (cv score %.4f, %.1f MB)" % (fold, os.path.basename(path), score,
                                                          os.path.getsize(out) / 1e6), flush=True)
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    video = os.path.join(WORK, "broadcast.mp4")
    if not os.path.isfile(video) or os.path.getsize(video) != os.path.getsize(SRC):
        t = time.time()
        shutil.copyfile(SRC, video + ".part")
        os.replace(video + ".part", video)
        print("copied %.1f GB from Drive in %.0f s" % (os.path.getsize(video) / 1e9, time.time() - t), flush=True)
    if not os.path.isdir(BAS):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/lRomul/ball-action-spotting", BAS],
                       check=True)
    model_path = weights(FOLD)
    os.chdir(BAS)
    sys.path.insert(0, BAS)
    import cv2
    import torch
    import src  # noqa: F401  (registers the author's argus model class)
    from src.predictors import MultiDimStackerPredictor
    from src.utils import post_processing

    predictor = MultiDimStackerPredictor(model_path, device="cuda:0", tta=True)
    cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    preds, bad = {}, 0
    t0 = time.time()
    predictor.reset_buffers()
    with torch.autocast("cuda", dtype=torch.float16):
        for i in range(n):
            if i % 2:
                cap.grab()
                continue
            ok, frame = cap.read()
            if not ok:
                bad += 1
                frame = np.zeros((720, 1280, 3), np.uint8)
            g = cv2.cvtColor(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            pred, pi = predictor.predict(torch.from_numpy(g), i)
            if pred is not None and pi >= 0:
                preds[pi] = pred.float().cpu().numpy()
            if i % 20000 == 0:
                print("  frame %d / %d, %.0f s" % (i, n, time.time() - t0), flush=True)
    idx = np.array(sorted(preds))
    raw = np.stack([preds[k] for k in idx]).astype(np.float16)
    out = os.path.join(WORK, "bas_raw_fold%d.npz" % FOLD)
    # the author's post-processing (Gaussian sigma 3 frames, peaks >= 0.2, >= 15 frames apart) on the 25 fps series
    full = np.arange(idx[0], idx[-1] + 1)
    peaks = {}
    for c, name in enumerate(("PASS", "DRIVE")):
        series = np.interp(full, idx, raw[:, c].astype(np.float32))
        fr, conf = post_processing(list(full), series, gauss_sigma=3.0, height=0.2, distance=15)
        peaks[name] = (np.array(fr), np.array(conf, np.float32))
    np.savez_compressed(out, frame_indexes=idx, raw=raw, pass_frames=peaks["PASS"][0], pass_conf=peaks["PASS"][1],
                        drive_frames=peaks["DRIVE"][0], drive_conf=peaks["DRIVE"][1])
    mins = n / 25 / 60
    print("predicted %d frames (%d unreadable) in %.0f s -> %s (%.2f MB)" % (len(idx), bad, time.time() - t0, out,
                                                                           os.path.getsize(out) / 1e6))
    for c, name in enumerate(("PASS", "DRIVE")):
        fr, conf = peaks[name]
        print("%-5s mean prob %.4f | peaks >= 0.2: %d (%.1f / video min), >= 0.5: %d" % (
            name, float(raw[:, c].astype(np.float32).mean()), len(fr), len(fr) / mins, int((conf >= 0.5).sum())))


if __name__ == "__main__":
    main()
