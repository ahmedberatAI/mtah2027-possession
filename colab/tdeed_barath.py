"""Round 22, step 1 (Colab VM only, colab/run_tdeed.sh): the SoccerNet 2025 Team Ball Action Spotting baseline (T-DEED
with a team head; SoccerNet/sn-teamspotting, GPL-3.0, the devkit's public checkpoint) on the BAR-ATH broadcast video.

Frames as the devkit extracts them (796x448, 25 fps, frame<i>.jpg), its test-time path (clips of 100 at stride 2, 3/4
overlap, soft NMS) and its event file (results_spotting.json: position in ms from the video start, label, team "left" /
"right", confidence). The dense per-frame class x team scores (stride 2) are kept as well (scores.npz) for a later
fusion. Output in /content/mtah/out/tdeed_barath/, copied to the user's Drive (MyDrive/mtah_out/); prints aggregates
only: event counts per class and team.
"""
import glob
import json
import os
import shutil
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

W = "/content/tdeed"
REPO = os.path.join(W, "sn-teamspotting")
FRAMES = os.path.join(W, "frames")
VIDEO = "barath/broadcast"
OUT = "/content/mtah/out/tdeed_barath"
DRIVE_OUT = "/content/drive/MyDrive/mtah_out/tdeed_barath"
sys.path.insert(0, REPO)
os.chdir(REPO)

from dataset.frame import ActionSpotVideoDataset  # noqa: E402
from model.model import TDEEDModel  # noqa: E402
from util.dataset import load_classes  # noqa: E402
from util.eval import INFERENCE_BATCH_SIZE, WINDOWS_SNB, process_frame_predictions, soft_non_maximum_supression  # noqa: E402
from util.io import load_json, store_json_snb  # noqa: E402


def model_args():
    """The baseline config as train_tdeed_bas.update_args reads it (no wandb, no stored training clips)."""
    config = load_json("config/SoccerNetBall/SoccerNetBall_baseline.json")
    args = types.SimpleNamespace(model="SoccerNetBall_baseline", acc_grad_iter=1, seed=1, **config)
    args.frame_dir, args.crop_dim, args.only_test = FRAMES, None, True
    return args


def load_model(args, classes):
    """TDEEDModel with the two prediction heads of joint training (Team-BAS classes, SoccerNet classes)."""
    joint = load_classes(os.path.join("data", args.joint_train["dataset"], "class.txt"), event_team=args.event_team)
    model = TDEEDModel(args=args)
    heads = [len(classes) // 2 + 1, len(joint) // 2 + 1]
    model._model.update_pred_head(heads)
    model._num_classes = int(np.sum(heads))
    ckpt = sorted(glob.glob(os.path.join(W, "ckpt", "**", "*.pt"), recursive=True))
    print("checkpoint files: %d; using %s (%.1f MB); heads %s" % (
        len(ckpt), os.path.basename(ckpt[0]), os.path.getsize(ckpt[0]) / 1e6, heads), flush=True)
    model.load(torch.load(ckpt[0], map_location="cpu"))
    return model


def main():
    t0 = time.time()
    args = model_args()
    classes = load_classes("data/soccernetball/class.txt", event_team=True)
    model = load_model(args, classes)
    n = len(glob.glob(os.path.join(FRAMES, VIDEO, "frame*.jpg")))
    split = os.path.join(W, "barath.json")
    with open(split, "w") as fh:
        json.dump([{"video": VIDEO, "num_frames": n}], fh)
    data = ActionSpotVideoDataset(classes, split, FRAMES, args.modality, args.clip_len,
                                  overlap_len=args.clip_len // 4 * 3, stride=2, dataset="soccernetball",
                                  event_team=True)
    pred_dict = {v: (np.zeros((length, len(classes) + 1), np.float32), np.zeros(length, np.int32))
                 for v, length, _ in data.videos}
    print("frames %d, clips %d, model ready in %.0f s" % (n, len(data), time.time() - t0), flush=True)
    done = 0
    for clip in DataLoader(data, num_workers=8, pin_memory=True, batch_size=INFERENCE_BATCH_SIZE):
        _, batch_scores = model.predict(clip["frame"])
        for i in range(clip["frame"].shape[0]):
            scores, support = pred_dict[clip["video"][i]]
            pred = batch_scores[i]
            start = clip["start"][i].item()
            if start < 0:
                pred, start = pred[-start:, :], 0
            end = min(start + pred.shape[0], scores.shape[0])
            pred = pred[:end - start, :]
            scores[start:end, :] += pred
            support[start:end] += (pred.sum(axis=1) != 0) * 1
        done += clip["frame"].shape[0]
        if done % 2000 < INFERENCE_BATCH_SIZE:
            print("  %d / %d clips, %.0f s" % (done, len(data), time.time() - t0), flush=True)
    events = soft_non_maximum_supression(process_frame_predictions(data, classes, pred_dict, threshold=0.01),
                                         window=WINDOWS_SNB[1], threshold=0.01)
    os.makedirs(OUT, exist_ok=True)
    store_json_snb(OUT, events, stride=data._stride)
    scores, support = pred_dict[VIDEO]
    np.savez_compressed(os.path.join(OUT, "scores.npz"), scores=scores.astype(np.float16), support=support,
                        classes=json.dumps(classes), stride=data._stride)
    ev = [e for g in events for e in g["events"]]
    print("events %d (all confidences >= 0.01), %.0f s" % (len(ev), time.time() - t0), flush=True)
    for thr in (0.2, 0.5):
        counts = {}
        for e in ev:
            if e["score"] >= thr:
                key = "%s/%s" % (e["label"], e.get("team", "?"))
                counts[key] = counts.get(key, 0) + 1
        print("  confidence >= %.1f: %s" % (thr, ", ".join("%s %d" % kv for kv in sorted(counts.items()))), flush=True)
    if os.path.isdir("/content/drive/MyDrive"):
        shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True)
        print("copied to the user's Drive: MyDrive/mtah_out/tdeed_barath", flush=True)


if __name__ == "__main__":
    main()
