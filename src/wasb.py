"""Pretrained WASB (HRNet, soccer) ball detector: model loading + inference on frame triplets.

Input: three consecutive RGB frames (t-1, t, t+1), each 1920x1080 resized to 512x288 and ImageNet-
normalised, stacked to 9 channels. Output: 3 heatmaps (one per input frame); we use the middle one.
Weights: external_data/wasb/wasb_soccer/wasb_soccer_best.pth.tar (MIT), loaded with weights_only.
"""
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "third_party"))
from wasb_hrnet import HRNet  # noqa: E402

WEIGHTS = os.path.join(ROOT, "external_data", "wasb", "wasb_soccer", "wasb_soccer_best.pth.tar")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IN_W, IN_H = 512, 288


class AttrDict(dict):
    def __getattr__(self, k):
        v = self[k]
        return AttrDict(v) if isinstance(v, dict) else v


def load(device="cpu"):
    cfg = yaml.safe_load(open(os.path.join(HERE, "third_party", "wasb_model.yaml")))
    model = HRNet(AttrDict(cfg))
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(sd)
    return model.eval().to(device)


def preprocess(frames_rgb):
    """frames_rgb: list of 3 HxWx3 uint8 arrays already resized to 512x288 -> (9, 288, 512)."""
    x = [((f.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for f in frames_rgb]
    return np.concatenate(x, 0)


@torch.no_grad()
def heatmaps(model, batch, device="cpu"):
    """batch: (B, 9, 288, 512) float32 -> (B, 3, 288, 512) sigmoid heatmaps."""
    y = model(torch.from_numpy(batch).to(device))[0]
    return torch.sigmoid(y).cpu().numpy()


def peak(hm):
    """(288, 512) heatmap -> (x_px, y_px in 1920x1080, score)."""
    i = int(np.argmax(hm))
    yy, xx = divmod(i, hm.shape[1])
    return (xx + 0.5) * 1920 / IN_W, (yy + 0.5) * 1080 / IN_H, float(hm[yy, xx])
