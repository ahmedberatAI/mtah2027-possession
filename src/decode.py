"""Segment-friendly decoding of per-frame state scores.

viterbi_min_dur maximises  sum_t score[t, s_t]  -  sum_{switches} penalty[s_{t-1}, s_t]
subject to every segment of state k lasting at least min_dur[k] frames. Minimum durations are
implemented by expanding state k into a chain of min_dur[k] sub-states (an explicit-duration
HMM with a geometric tail), so the cost is O(n * sum(min_dur)).

Scores are rewards (think log-likelihoods); penalties are positive costs.
"""
import numpy as np
from numba import njit

H, A, D = 0, 1, 2


@njit(cache=True)
def _viterbi_expanded(score, penalty, min_dur):
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
    # a segment may start at t=0 in any state, at its first sub-state; allow the first segment
    # to be shorter than min_dur by also allowing entry at the last sub-state.
    for k in range(K):
        delta[offs[k]] = score[0, k]
        delta[offs[k + 1] - 1] = max(delta[offs[k + 1] - 1], score[0, k])
    new = np.empty(S)
    for t in range(1, n):
        # best "segment finished" value per state: last sub-state of k
        for s in range(S):
            k = state_of[s]
            j = pos_of[s]
            if j == 0:
                # entering k from the last sub-state of some other state
                best = NEG
                arg = -1
                for k2 in range(K):
                    if k2 == k:
                        continue
                    s2 = offs[k2 + 1] - 1
                    v = delta[s2] - penalty[k2, k]
                    if v > best:
                        best = v
                        arg = s2
                if min_dur[k] == 1:
                    # staying in a 1-long chain
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
                # last sub-state: come from previous sub-state or stay
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
    # end: allow ending anywhere (the last segment may be cut by the end of the half)
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


def viterbi_min_dur(score, penalty, min_dur=None):
    score = np.ascontiguousarray(score, dtype=np.float64)
    K = score.shape[1]
    penalty = np.ascontiguousarray(penalty, dtype=np.float64)
    if min_dur is None:
        min_dur = np.ones(K, np.int64)
    min_dur = np.maximum(1, np.asarray(min_dur, dtype=np.int64))
    return _viterbi_expanded(score, penalty, min_dur)


def decode_halves(score, penalty, min_dur, halves):
    """Run the decoder independently inside each (start, stop_inclusive) block."""
    out = np.full(len(score), -1, np.int64)
    for a, b in halves:
        out[a:b + 1] = viterbi_min_dur(score[a:b + 1], penalty, min_dur)
    return out
