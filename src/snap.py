"""Boundary snapping after decoding: move each predicted boundary to the nearest physical event
within a window, when the ball is observed there.

  alive -> D : first frame the ball is beyond a line (crossing)
  D -> alive : first frame of the restart kick (ball speed jumps above v_kick after being slow)
  H <-> A    : first frame the new team controls the ball (nearest within r_ctrl, no opponent closer)

Works on one block (a half). Inputs are per-frame arrays for that block.
"""
import numpy as np


def segments(lab):
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return b


def snap(lab, ball_out, bspeed, seen, dh, da, win=50, v_kick=4.0, r_ctrl=1.5, excl=0.5,
         do_dead=True, do_restart=True, do_team=True):
    lab = lab.copy()
    n = len(lab)
    out = np.nan_to_num(ball_out, nan=-1) > 0
    bs = np.nan_to_num(bspeed, nan=0)
    ctrl_h = seen & (dh <= r_ctrl) & ~(da < dh + excl)
    ctrl_a = seen & (da <= r_ctrl) & ~(dh < da + excl)
    for t in segments(lab):
        prev, new = lab[t - 1], lab[t]
        lo, hi = max(1, t - win), min(n - 1, t + win)
        cand = None
        if do_dead and prev < 2 and new == 2:
            idx = np.arange(lo, hi)
            c = idx[out[idx] & ~out[idx - 1]]
            if len(c):
                cand = c[np.argmin(np.abs(c - t))]
        elif do_restart and prev == 2 and new < 2:
            idx = np.arange(lo, hi)
            c = idx[(bs[idx] > v_kick) & (bs[idx - 1] <= v_kick) & seen[idx]]
            if len(c):
                cand = c[np.argmin(np.abs(c - t))]
        elif do_team and prev < 2 and new < 2:
            m = ctrl_h if new == 0 else ctrl_a
            idx = np.arange(lo, hi)
            c = idx[m[idx] & ~m[idx - 1]]
            if len(c):
                cand = c[np.argmin(np.abs(c - t))]
        if cand is None or cand == t:
            continue
        # move the boundary without crossing a neighbouring boundary
        if cand < t:
            seg_start = t - 1
            while seg_start > 0 and lab[seg_start - 1] == prev:
                seg_start -= 1
            if cand > seg_start + 1:
                lab[cand:t] = new
        else:
            seg_end = t
            while seg_end < n - 1 and lab[seg_end + 1] == new:
                seg_end += 1
            if cand < seg_end:
                lab[t:cand] = prev
    return lab
