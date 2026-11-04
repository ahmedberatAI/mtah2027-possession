"""Dead-ball boundary timing (round 29, docs/RAPOR.md; evaluated by src/exp_dtime.py), the step v16 adds to v15.

A label sequence keeps its dead stretches; each stop (alive -> dead) and restart (dead -> alive) moves onto the nearest
boundary of the same kind in a reference decode -- the learned dead model's own -- within WIN frames and never across
another boundary; the moved stretch takes the label it is pulled into.
"""
import numpy as np

import decode2
import exp_v7

WIN = 50
STOP, RESTART = 1, -1
EPS = 1e-4


def dbounds(lab):
    """(frame, STOP / RESTART) of every alive <-> dead boundary."""
    d = (np.asarray(lab) == 2).astype(np.int8)
    f = np.flatnonzero(d[1:] != d[:-1]) + 1
    return f, np.where(d[f] == 1, STOP, RESTART)


def retime(lab, ref, kinds=(STOP, RESTART), win=WIN):
    """lab's stops / restarts of the given kinds moved onto ref's nearest boundary of the same kind within win frames,
    never across another boundary of lab."""
    lab = np.asarray(lab)
    out = lab.copy()
    allb = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    rf, rk = dbounds(ref)
    for f, k in zip(*dbounds(lab)):
        if k not in kinds:
            continue
        c = rf[(rk == k) & (np.abs(rf - f) <= win)]
        if not len(c):
            continue
        g = int(c[np.argmin(np.abs(c - f))])
        i = int(np.searchsorted(allb, f))
        lo = allb[i - 1] if i > 0 else 0
        hi = allb[i + 1] if i + 1 < len(allb) else len(out)
        if lo < g < hi and g != f:
            if g < f:
                out[g:f] = lab[f]
            else:
                out[f:g] = lab[f - 1]
    return out


def alive_logit(p_dead):
    p = np.clip(p_dead, EPS, 1 - EPS)
    return np.log((1 - p) / p)


def learned_decode(p_dead, ph, pen_team=12.0, pen_dead=30.0, min_poss=8, min_dead=50):
    """The learned dead model's own decode: its alive logit with the team probability ph and BAR-ATH's decoder
    settings (the scores of src/exp_scdead.py decode_scores)."""
    a = alive_logit(p_dead)
    q = np.clip(ph, EPS, 1 - EPS)
    sp = decode2.duration_spec(pen_team, pen_dead, min_poss, min_dead, exp_v7.ramp(175, 48.0))
    return sp.decode(np.stack([a / 2 + np.log(q), a / 2 + np.log(1 - q), -a / 2], 1)).astype(np.int8)
