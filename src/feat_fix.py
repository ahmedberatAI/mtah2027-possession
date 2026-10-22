"""Bench feature plugins that REPLACE base features found to be domain-shifted by bugs (src/exp_advval.py).

own_half: mlfeat's own_half_frac_c12 compares NaN positions with "< 0.5", which is False, so absent slots
count as "not in own half". BAR-ATH has 46 slots (23 per team) against ~30 in DFL, so the share was diluted
differently in the two domains (univariate AUC DFL vs BAR-ATH 0.017). Here only present players count.
"""
import numpy as np

from mlfeat import win_mean


def own_half(P, B, team, gk, pitch):
    L = float(pitch[0])
    x = P[:, :, 0].astype(np.float32) * L
    present = np.isfinite(x)
    th, ta = team == 0, team == 1
    with np.errstate(invalid="ignore"):
        gkx_h = np.nanmean(np.where((th & gk)[None, :], x, np.nan)) if (th & gk).any() else np.nan
        cx_h = np.nanmean(np.where(th[None, :], x, np.nan), 1)
        cx_a = np.nanmean(np.where(ta[None, :], x, np.nan), 1)
    home_left = (gkx_h < L / 2) if np.isfinite(gkx_h) else (np.nanmean(cx_h) < np.nanmean(cx_a))
    sgn_h = 1.0 if home_left else -1.0
    own = np.where(th[None, :], (x - L / 2) * sgn_h < 0.5, (x - L / 2) * -sgn_h < 0.5)
    val = np.where(present & (th | ta)[None, :], own, np.nan)
    with np.errstate(invalid="ignore"):
        frac = np.nanmean(val, 1)
    return {"own_half_frac_c12": win_mean(frac, -12, 12)}, {}, {}
