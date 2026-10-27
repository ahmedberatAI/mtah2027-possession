"""Ball frames that BAR-ATH's tactical ATD filled in by prediction (idea 8, docs/RAPOR.md).

Over-smooth runs of the tactical ball -- |2nd difference| < 2 mm for >= 8 frames while it moves (> 1 m/s) -- are
3.1% of the seen graded frames; there the tactical and broadcast balls disagree by 1.66 m (median; 3.15 m for runs
of >= 25 frames) against 0.75 m elsewhere, so they are extrapolations rather than detections. clean_blocks() replaces
them with the aligned broadcast ball where that is valid and live, else marks them missing.
"""
import numpy as np

import atd

EPS_FLAT = 0.002
MIN_RUN = 8


def fabricated(bxy):
    """Boolean mask of over-smooth (predicted) frames of a ball track in metres."""
    n = len(bxy)
    seen = np.isfinite(bxy[:, 0])
    a2 = np.full(n, np.nan)
    a2[1:-1] = np.linalg.norm(bxy[2:] - 2 * bxy[1:-1] + bxy[:-2], axis=1)
    v = np.full(n, np.nan)
    v[1:-1] = np.linalg.norm(bxy[2:] - bxy[:-2], axis=1) * 12.5
    flat = (np.nan_to_num(a2, nan=1.0) < EPS_FLAT) & seen
    e = np.diff(np.concatenate(([0], flat.astype(np.int8), [0])))
    flag = np.zeros(n, bool)
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if t - s >= MIN_RUN and np.nanmedian(v[s:t]) > 1.0:
            flag[s:t] = True
    return flag


def clean_blocks(blocks, f):
    """Copies of bench's BAR-ATH blocks with fabricated tactical-ball frames replaced (broadcast) or dropped."""
    flag = fabricated(atd.to_metres(atd.load("tactical")["ball"][:, :2]))
    bro_ok = f["b_seen"] & f["b_live"]
    out = []
    for b in blocks:
        a, e = b["a"], b["b"]
        B = f["bxy"][a:e + 1].copy()
        fl = flag[a:e + 1]
        B[fl] = np.where(bro_ok[a:e + 1, None][fl], f["b_ball"][a:e + 1][fl], np.nan)
        bb = dict(b)
        bb["B"] = (B / np.array(atd.PITCH, np.float32)).astype(np.float32)
        out.append(bb)
    return out
