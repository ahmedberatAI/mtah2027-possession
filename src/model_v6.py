"""v6 = v5 + referee-whistle evidence from the tactical video's audio (src/whistle_ref.py).

Whistles = the `wh_topn` strongest local maxima (+-1 s suppression, graded frames) of the
referee-whistle matched filter (template from the two kick-off whistles; see whistle_ref.py and
docs/RAPOR.md Ek D). A whistle only acts where it can say "play stopped earlier than v5
thinks": v5 (the same scores without whistles) has the ball alive at the whistle and its own
alive -> D switch follows within `wh_lead` frames. Such a whistle gives a bonus `wh_bonus` to
ENTERING the dead-ball state at frames w + wh_e0 .. w + wh_e1 (+-0.32 s, the detector's timing
precision), so the decoder moves that D start back to the whistle when the alive evidence in
between is weak.

It is a switch bonus, not D evidence, and it is restricted to those whistles on purpose: with a
bonus on every whistle the decoder delays D starts to the whistle and splits stoppages around
mid-stoppage whistles (restart signals, cards) to collect it again.

The decoder is decode.viterbi_min_dur with a per-frame entry bonus added to the switch reward.
"""
import numpy as np
from numba import njit

import atd
import model_v5

DEFAULT = dict(model_v5.DEFAULT)
DEFAULT.update(wh_spec="ref:s", wh_topn=100, wh_bonus=30.0, wh_e0=-8, wh_e1=8, wh_lead=75, wh_only_late=1)
_peaks = {}


def whistle_peaks(spec, topn):
    if spec not in _peaks:
        import whistle_eval
        s = whistle_eval.get_score(spec)
        _peaks[spec] = whistle_eval.peaks(s, atd.graded_mask(len(s)))
    return _peaks[spec][:topn]


def enter_bonus(n, p, base=None):
    """Entry bonus for D around the selected whistles. base = v5 labels (no whistle term)."""
    bonus = np.zeros((n, 3))
    if p["wh_bonus"] <= 0 or p["wh_topn"] <= 0:
        return bonus
    pk = whistle_peaks(p["wh_spec"], int(p["wh_topn"]))
    if p["wh_only_late"] and base is not None:
        alive = (base == 0) | (base == 1)
        starts = np.flatnonzero((base[1:] == 2) & alive[:-1]) + 1
        nxt = starts[np.clip(np.searchsorted(starts, pk), 0, len(starts) - 1)]
        pk = pk[alive[pk] & (nxt > pk) & (nxt - pk <= p["wh_lead"])]
    for w in pk:
        lo, hi = max(0, w + int(p["wh_e0"])), min(n, w + int(p["wh_e1"]) + 1)
        bonus[lo:hi, 2] = p["wh_bonus"]
    return bonus


@njit(cache=True)
def _viterbi_enter(score, penalty, min_dur, enter):
    """decode._viterbi_expanded with `enter[t, k]` added whenever state k is entered at frame t."""
    n, K = score.shape
    offs = np.zeros(K + 1, np.int64)
    for k in range(K):
        offs[k + 1] = offs[k] + min_dur[k]
    S = offs[K]
    state_of = np.zeros(S, np.int64)
    pos_of = np.zeros(S, np.int64)
    for k in range(K):
        for j in range(min_dur[k]):
            state_of[offs[k] + j] = k
            pos_of[offs[k] + j] = j
    NEG = -1e18
    delta = np.full(S, NEG)
    back = np.zeros((n, S), np.int32)
    for k in range(K):
        delta[offs[k]] = score[0, k]
        delta[offs[k + 1] - 1] = max(delta[offs[k + 1] - 1], score[0, k])
    new = np.empty(S)
    for t in range(1, n):
        for s in range(S):
            k = state_of[s]
            j = pos_of[s]
            if j == 0:
                best = NEG
                arg = -1
                for k2 in range(K):
                    if k2 == k:
                        continue
                    s2 = offs[k2 + 1] - 1
                    v = delta[s2] - penalty[k2, k] + enter[t, k]
                    if v > best:
                        best = v
                        arg = s2
                if min_dur[k] == 1:
                    v = delta[s]
                    if v > best:
                        best = v
                        arg = s
                new[s] = best + score[t, k]
                back[t, s] = arg
            elif j < min_dur[k] - 1:
                new[s] = delta[s - 1] + score[t, k]
                back[t, s] = s - 1
            else:
                a = delta[s - 1]
                b = delta[s]
                if b >= a:
                    new[s] = b + score[t, k]
                    back[t, s] = s
                else:
                    new[s] = a + score[t, k]
                    back[t, s] = s - 1
        for s in range(S):
            delta[s] = new[s]
    best = NEG
    arg = 0
    for s in range(S):
        if delta[s] > best:
            best = delta[s]
            arg = s
    path = np.empty(n, np.int64)
    s = arg
    for t in range(n - 1, -1, -1):
        path[t] = state_of[s]
        s = back[t, s]
    return path


def decode_halves_enter(S, pen, min_dur, enter):
    out = np.full(len(S), -1, np.int64)
    for _, a, b in atd.HALVES:
        out[a:b + 1] = _viterbi_enter(np.ascontiguousarray(S[a:b + 1], dtype=np.float64),
                                      np.ascontiguousarray(pen, dtype=np.float64),
                                      np.maximum(1, np.asarray(min_dur, dtype=np.int64)),
                                      np.ascontiguousarray(enter[a:b + 1], dtype=np.float64))
    return out


def predict(f, params=None):
    import train_dfl
    train_dfl.ensure_barath_probs()     # make_submission.py only does this for model_v5
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = model_v5.scores(f, p)
    pen = np.array([[0, p["pen_team"], p["pen_dead"]], [p["pen_team"], 0, p["pen_dead"]],
                    [p["pen_dead"], p["pen_dead"], 0]], dtype=np.float64)
    min_dur = np.array([p["min_poss"], p["min_poss"], p["min_dead"]])
    base = None
    if p["wh_only_late"]:
        base = decode_halves_enter(S, pen, min_dur, np.zeros((f["n"], 3)))     # = model_v5.predict
    path = decode_halves_enter(S, pen, min_dur, enter_bonus(f["n"], p, base))
    labels = np.full(f["n"], 3, np.int8)
    labels[path >= 0] = path[path >= 0]
    return labels
