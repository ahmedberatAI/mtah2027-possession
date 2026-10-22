"""Attention-to-ball cues for the dead-ball model (idea D1, docs/FIKIRLER.md), as bench feature plugins.

While play is alive, moving players run towards / with the ball; at a stoppage they turn away and walk
back into position (open DFL, clean data: dead-ball AUC speed 0.929 -> speed + attention 0.944, four
windows + LightGBM 0.974; src/exp_attention_blend.py). The tracked ball is often missing around
stoppages, so the reference point is the ball interpolated between its last and next sightings (the
last sighting alone past the end of the half).

  bench.run("base+feat_dead:attention", "native", var="v3")
"""
import numpy as np

from mlfeat import velocity, win_mean

WINS = ((-25, 25), (-75, 0), (0, 75), (-150, 150))


def _ball_ref(bxy):
    """Ball position with gaps filled by linear interpolation in time (edges: nearest sighting)."""
    out = bxy.copy()
    seen = np.isfinite(bxy[:, 0])
    if seen.sum() < 2:
        return out
    t = np.arange(len(bxy))
    for k in (0, 1):
        out[:, k] = np.interp(t, t[seen], bxy[seen, k])
    return out


def attention(P, B, team, gk, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    pxy = P[:, :, :2].astype(np.float32) * np.array([L, W], np.float32)
    bxy = _ball_ref(B[:, :2].astype(np.float32) * np.array([L, W], np.float32))
    v = velocity(pxy)
    sp = np.linalg.norm(v, axis=2)
    sp[sp > 12] = np.nan
    to = bxy[:, None, :] - pxy
    dist = np.linalg.norm(to, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v * to, axis=2) / np.maximum(sp * dist, 1e-6)
        mov = sp > 1.0
        align = np.nanmean(np.where(mov, cos, np.nan), 1)
        toward = np.nanmean(np.where(np.isfinite(sp), mov & (cos > 0.7), np.nan), 1)
        # players near the ball (< 15 m) only: the ones that react first
        near = dist < 15
        align_near = np.nanmean(np.where(mov & near, cos, np.nan), 1)
    F = {}
    for name, x in (("att_align", align), ("att_toward", toward), ("att_align_near", align_near)):
        for a, b in WINS:
            F["%s_%d_%d" % (name, a, b)] = win_mean(x, a, b)
    F["att_align_fwd_minus_bwd"] = F["att_align_0_75"] - F["att_align_-75_0"]
    return F, {}, {}
