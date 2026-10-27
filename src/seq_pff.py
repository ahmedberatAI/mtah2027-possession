"""Balanced-source temporal team models; no labels enter the network inputs.

Both models see the v7 feature set for one team, plus missingness indicators.
The two team orders are averaged antisymmetrically in logit space at inference.
25 Hz source features are sampled at 12.5 Hz; output logits return to 25 Hz.
"""
import numpy as np
import torch
from torch import nn

STEP = 2
CORE = 256
HALO = 128
WINDOW = CORE + 2 * HALO
CLIP = 6.0


def views(f):
    return np.stack([np.hstack([f['XF'][::STEP], f[k][::STEP]])
                     for k in ('XH', 'XA')], axis=1).astype(np.float32)


class Scaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)

    @classmethod
    def fit(cls, source_samples):
        # Equal source weight, independent of source duration; both team views
        # are included in each source's samples.
        moments = []
        for a in source_samples:
            a = np.asarray(a, np.float64).copy()
            a[~np.isfinite(a)] = np.nan
            count = np.isfinite(a).sum(0)
            mean = np.nansum(a, 0) / np.maximum(count, 1)
            second = np.nansum(a * a, 0) / np.maximum(count, 1)
            moments.append((mean, second, count > 0))
        count = sum(m[2] for m in moments)
        mean = sum(m[0] * m[2] for m in moments) / np.maximum(count, 1)
        second = sum(m[1] * m[2] for m in moments) / np.maximum(count, 1)
        scale = np.sqrt(np.maximum(0, second - mean * mean))
        scale = np.where(scale > 1e-5, scale, 1.)
        return cls(mean, scale)

    def transform(self, x):
        missing = ~np.isfinite(x)
        z = np.clip(np.nan_to_num((x-self.mean)/self.scale, nan=0.,
                                 posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)
        return np.concatenate([z, missing.astype(np.float32),
                               np.ones((*x.shape[:-1], 1), np.float32)], axis=-1)


def window(raw, start, side, scaler):
    """start is the central output start; never read beyond a half boundary."""
    left, right = start-HALO, start+CORE+HALO
    a, b = max(0, left), min(len(raw), right)
    out = np.zeros((WINDOW, 2*raw.shape[-1]+1), np.float32)
    if b > a:
        out[a-left:b-left] = scaler.transform(np.asarray(raw[a:b, side], np.float32))
    return out


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, drop):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = self.drop(torch.nn.functional.gelu(self.norm1(y)))
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(torch.nn.functional.gelu(self.norm2(y)))


class Net(nn.Module):
    def __init__(self, d_in, kind, width, drop):
        super().__init__()
        self.kind = kind
        self.input = nn.Sequential(nn.Linear(d_in, width), nn.LayerNorm(width), nn.GELU())
        if kind == 'tcn':
            # 2 * sum(dilations) = 126 steps of context on either side.
            self.temporal = nn.Sequential(*[TemporalBlock(width, d, drop)
                                             for d in (1, 2, 4, 8, 16, 32)])
            outwidth = width
        elif kind == 'bigru':
            self.temporal = nn.GRU(width, width, 2, batch_first=True,
                                   bidirectional=True, dropout=drop)
            outwidth = 2*width
        else:
            raise ValueError(kind)
        self.output = nn.Sequential(nn.Dropout(drop), nn.Linear(outwidth, 1))

    def forward(self, x):
        z = self.temporal(self.input(x))
        if self.kind == 'bigru':
            z = z[0]
        return self.output(z).squeeze(-1)


def balanced_loss(logits, target, valid, source):
    """Per-window live BCE, then exact 50/50 source means."""
    losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    per = (losses*valid).sum(1) / valid.sum(1).clamp(min=1)
    return .5*(per[source == 0].mean()+per[source == 1].mean())


@torch.no_grad()
def predict_raw(net, scaler, raw, full_length, batch=32):
    net.eval()
    device = next(net.parameters()).device
    starts = list(range(0, len(raw), CORE))
    logits = np.empty(len(raw), np.float32)
    for offset in range(0, len(starts), max(1, batch//2)):
        ss = starts[offset:offset+max(1, batch//2)]
        xx = np.stack([window(raw, s, side, scaler) for s in ss for side in (0, 1)])
        yy = net(torch.from_numpy(xx).to(device))[:, HALO:HALO+CORE].cpu().numpy()
        for i, s in enumerate(ss):
            n = min(CORE, len(raw)-s)
            logits[s:s+n] = .5*(yy[2*i, :n]-yy[2*i+1, :n])
    full = np.interp(np.arange(full_length), np.arange(len(raw))*STEP, logits)
    return (1/(1+np.exp(-np.clip(full, -20, 20)))).astype(np.float32)
