"""The metric's Bayes decision on the stacker's joint posterior (round 88, docs/RAPOR.md).

v21 decodes the mix of the stacker's per-frame marginals with v20's smoothed blend by Viterbi with hand-set penalties.
But the stacker is a path model trained on the likelihood of whole paths, and its samples are calibrated: on the open
sets they have about as many possessions per half, and of the same lengths, as the truth (DFL 121 against 115, Metrica
119 / 116, SkillCorner 116 / 116). The ranking metric scores possessions as objects (spellF1, IoU 0.3-0.7), so the
labelling here is chosen for the score it is expected to get under those samples:

    samples     m paths from the path model's exact posterior (forward filtering, backward sampling), per half, at 5 Hz;
                with several stackers, an equal share from each
    candidates  every segment some sample used, and the segments of a reference path (always allowed, so a labelling
                always exists)
    value       a possession candidate is worth its chance of matching a same-team segment of a sample, averaged over
                the IoU thresholds, minus lam (about F*/2: adding a possession pays when it matches more often than
                that); a dead-ball candidate is worth 0 and is allowed only where most samples, or the reference, are dead
    labelling   the chain of candidates with the largest total value (dynamic programming over segment ends)

No labels are read anywhere. Random draws use numba's generator with fixed seeds, so the output is deterministic.
"""
import numpy as np
from numba import njit

STEP = 5
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])


@njit(cache=True, nogil=True)
def _lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


@njit(cache=True, nogil=True)
def ffbs(e, A, m, seed):
    """m label paths (H 0, A 1, D 2) from the exact posterior of a path model with K latent states per team then dead
    (e: (T, 2K + 1) emissions, A: (2K + 1, 2K + 1) moves). Returns segments (start, end, label) and per-sample offsets."""
    np.random.seed(seed)
    T, S = e.shape
    fw = np.zeros((T, S))
    fw[0] = e[0]
    for t in range(1, T):
        for s in range(S):
            fw[t, s] = _lse(fw[t - 1] + A[:, s]) + e[t, s]
    seg = np.zeros((6000 * m, 3), np.int32)
    off = np.zeros(m + 1, np.int64)
    w = 0
    K = (S - 1) // 2
    G = np.zeros(S, np.int64)
    G[K:2 * K] = 1
    G[2 * K] = 2
    for k in range(m):
        path = np.zeros(T, np.int8)
        x = fw[T - 1]
        x = np.exp(x - x.max())
        s_ = min(int(np.searchsorted(np.cumsum(x), np.random.random() * x.sum())), S - 1)
        path[T - 1] = G[s_]
        for t in range(T - 2, -1, -1):
            x = fw[t] + A[:, s_]
            x = np.exp(x - x.max())
            s_ = min(int(np.searchsorted(np.cumsum(x), np.random.random() * x.sum())), S - 1)
            path[t] = G[s_]
        a = 0
        for t in range(1, T + 1):
            if t == T or path[t] != path[a]:
                seg[w, 0], seg[w, 1], seg[w, 2] = a, t, path[a]
                w += 1
                a = t
        off[k + 1] = w
    return seg[:w].copy(), off


@njit(cache=True, nogil=True)
def match_counts(x, y, k, seg, off, ious):
    """In how many samples a segment of label k matches (x, y) at each IoU threshold."""
    m = off.shape[0] - 1
    res = np.zeros(ious.shape[0], np.int32)
    for s in range(m):
        lo, hi = off[s], off[s + 1]
        a, b = lo, hi
        while a < b:
            mid = (a + b) // 2
            if seg[mid, 1] <= x:
                a = mid + 1
            else:
                b = mid
        best = 0.0
        i = a
        while i < hi and seg[i, 0] < y:
            if seg[i, 2] == k:
                inter = min(y, seg[i, 1]) - max(x, seg[i, 0])
                union = max(y, seg[i, 1]) - min(x, seg[i, 0])
                v = inter / union
                if v > best:
                    best = v
            i += 1
        for j in range(ious.shape[0]):
            if best >= ious[j]:
                res[j] += 1
    return res


@njit(cache=True, nogil=True)
def rewards(cand, seg, off, ious, lam, cum, dshare=0.5):
    """Per candidate (x, y, k, in_reference): possession: mean chance of matching - lam; dead: 0 where at least dshare
    of the samples are dead or the reference is, else not allowed (-inf)."""
    m = off.shape[0] - 1
    out = np.full(cand.shape[0], -np.inf)
    for i in range(cand.shape[0]):
        x, y, k = cand[i, 0], cand[i, 1], cand[i, 2]
        share = (cum[y, k] - cum[x, k]) / (y - x)
        if k == 2:
            if share >= dshare or cand[i, 3] == 1:
                out[i] = 0.0
            continue
        c_ = match_counts(x, y, k, seg, off, ious)
        out[i] = np.mean(c_ / m) - lam
    return out


@njit(cache=True, nogil=True)
def chain(n, cand, r):
    """The full labelling of [0, n) with the largest total value; candidates sorted by their end. Consecutive segments
    always change label, so a possession is never split by the decision itself."""
    best = np.full((n + 1, 3), -np.inf)
    back = np.full((n + 1, 3), -1, np.int64)
    prev = np.full((n + 1, 3), -1, np.int8)
    for i in range(cand.shape[0]):
        if r[i] == -np.inf:
            continue
        x, y, k = cand[i, 0], cand[i, 1], cand[i, 2]
        if x == 0:
            v, pk = r[i], -1
        else:
            v, pk = -np.inf, -1
            for q in range(3):
                if q != k and best[x, q] + r[i] > v:
                    v, pk = best[x, q] + r[i], q
        if v > best[y, k]:
            best[y, k] = v
            back[y, k] = i
            prev[y, k] = pk
    lab = np.zeros(n, np.int8)
    k = int(np.argmax(best[n]))
    if best[n, k] == -np.inf:
        return lab, False
    t = n
    while t > 0:
        i = back[t, k]
        lab[cand[i, 0]:cand[i, 1]] = k
        pk = prev[t, k]
        t = cand[i, 0]
        k = pk
    return lab, True


def reference_segments(path, n):
    """The reference labelling (25 fps) on the 5 Hz grid as segments."""
    mp = np.asarray(path)[STEP // 2::STEP][:n]
    st = np.r_[0, np.flatnonzero(mp[1:] != mp[:-1]) + 1]
    en = np.r_[st[1:], n]
    return np.stack([st, en, mp[st]], 1).astype(np.int32)


def solve(n, seg, off, ref_seg, lam, ious=IOUS, dshare=0.5):
    both = np.vstack([seg, ref_seg])
    key = both[:, 2].astype(np.int64) + 3 * (both[:, 1].astype(np.int64) + (n + 1) * both[:, 0].astype(np.int64))
    uniq, idx = np.unique(key, return_index=True)
    cand = np.zeros((len(uniq), 4), np.int64)
    cand[:, :3] = both[idx]
    rk = ref_seg[:, 2].astype(np.int64) + 3 * (ref_seg[:, 1].astype(np.int64) + (n + 1) * ref_seg[:, 0].astype(np.int64))
    cand[np.isin(uniq, rk), 3] = 1
    cand = cand[np.argsort(cand[:, 1], kind="stable")]
    Pk = np.zeros((n, 3))
    for a0, b0, k in seg:
        Pk[a0:b0, k] += 1
    Pk /= off.shape[0] - 1
    cum = np.zeros((n + 1, 3))
    cum[1:] = np.cumsum(Pk, axis=0)
    return chain(n, cand, rewards(cand, seg, off, ious, lam, cum, dshare))


def keep_reference_dead(labels, reference):
    """Option (round 88): frames the decision calls dead but the reference calls alive take the label of the decision's
    nearest possession, so that the dead/alive call stays the reference's where the decision would only extend a dead
    stretch. The decision's possessions are kept."""
    out = np.asarray(labels, np.int8).copy()
    ref = np.asarray(reference)[:len(out)]
    m = (out == 2) & (ref < 2)
    if not m.any():
        return out
    n = len(out)
    idx = np.arange(n)
    alive = out < 2
    last = np.maximum.accumulate(np.where(alive, idx, -1))
    nxt = np.minimum.accumulate(np.where(alive, idx, n)[::-1])[::-1]
    for i in np.flatnonzero(m):
        a, b = last[i], nxt[i]
        da = i - a if a >= 0 else n
        db = b - i if b < n else n
        if min(da, db) == n:
            continue
        out[i] = out[a] if da <= db else out[b]
    return out


def decide(members, reference, lam, m, seed, dshare=0.5):
    """members: [(emissions (T, S), moves (S, S))] of the stackers on one half; reference: a labelling of the half at
    25 fps (H 0, A 1, D 2), whose segments stay available. Returns the decided labelling at 25 fps."""
    reference = np.asarray(reference, np.int8)
    n = min(min(len(e) for e, _ in members), len(reference) // STEP)
    ref_seg = reference_segments(reference, n)
    segs, offs, base = [], [0], 0
    per = max(1, m // len(members))
    for j, (e, A) in enumerate(members):
        seg, off = ffbs(np.asarray(e[:n], np.float64), np.asarray(A, np.float64), per, seed + j)
        segs.append(seg)
        offs += list(off[1:] + base)
        base += len(seg)
    seg = np.vstack(segs)
    off = np.array(offs, np.int64)
    lab, ok = solve(n, seg, off, ref_seg, lam, dshare=dshare)
    if not ok:
        return reference.copy()
    out = np.repeat(lab, STEP)
    return np.r_[out, np.full(len(reference) - len(out), out[-1], np.int8)].astype(np.int8)
