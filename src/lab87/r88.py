"""Round 88: the metric's Bayes decision on the path model's own joint posterior. Round 73 sampled from a pseudo-posterior built from v21's marginals and penalties with a temperature (any T loses). The stacker is trained on the whole path's likelihood, so its forward-filtering backward-sampling gives real samples of spells: whether a short spell exists at all, and its extent, jointly. Candidates: every segment a sample used, and v21's path, always allowed; each spell is worth its mean chance of matching over the IoU thresholds minus lambda. Optionally the emissions take v20's smoothed blend and the alive evidence, as v21's mix does. Open data only, aggregate output.

    python -I r88.py grid SEED "dfl,metrica,sc,st"
"""
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402
from numba import njit  # noqa: E402

STEP = 5
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
M = int(os.environ.get("R88_M", "200"))
OUTD = os.path.join(lab.REPO, "cache", "lab87", "foldnets")
GROUP = np.array([0, 0, 1, 1, 2])


@njit(cache=True, nogil=True)
def _lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


@njit(cache=True, nogil=True)
def ffbs(e, A, m, seed):
    """m label paths (H 0, A 1, D 2) from the exact posterior of the 5-state path model; (segments, offsets)."""
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
    K = (S - 1) // 2                # K latent states per team, then dead; today's path model has K = 2
    G = np.zeros(S, np.int64)
    G[K:2 * K] = 1
    G[2 * K] = 2
    p = np.zeros(S)
    for k in range(m):
        path = np.zeros(T, np.int8)
        x = fw[T - 1]
        x = np.exp(x - x.max())
        s_ = np.searchsorted(np.cumsum(x), np.random.random() * x.sum())
        s_ = min(int(s_), S - 1)
        path[T - 1] = G[s_]
        for t in range(T - 2, -1, -1):
            x = fw[t] + A[:, s_]
            x = np.exp(x - x.max())
            s_ = np.searchsorted(np.cumsum(x), np.random.random() * x.sum())
            s_ = min(int(s_), S - 1)
            path[t] = G[s_]
        a = 0
        for t in range(1, T + 1):
            if t == T or path[t] != path[a]:
                seg[w, 0], seg[w, 1], seg[w, 2] = a, t, path[a]
                w += 1
                a = t
        off[k + 1] = w
    return seg[:w], off


def emit(K, s, tag="Hy"):
    with open(os.path.join(OUTD, "emit_%s_%d_%s.pkl" % (tag, s, K)), "rb") as fh:
        return pickle.load(fh)


def pool5(x, n):
    return np.asarray(x[:n * STEP], np.float64).reshape(n, STEP).mean(1)


def half_arms(e, A, a21, q21, s50, ev, beta_gamma, lams, seed, tau=1.0):
    import exp_mbr as mb
    # v21's path is the fallback, always allowed
    P = lab.get  # noqa: F841
    sp = lab._spec(6.0, 24.0)
    q_ = np.clip(q21, 1e-4, 1 - 1e-4)
    S = np.stack([a21 / 2 + np.log(q_), a21 / 2 + np.log(1 - q_), -a21 / 2], 1)
    map_path = sp.decode(S).astype(np.int8)
    n = min(len(e), len(map_path) // STEP)
    mp = map_path[STEP // 2::STEP][:n]
    st = np.r_[0, np.flatnonzero(mp[1:] != mp[:-1]) + 1]
    en = np.r_[st[1:], n]
    mp_seg = np.stack([st, en, mp[st]], 1).astype(np.int32)
    # as in v21's emission: S = [a/2 + log q, a/2 + log(1-q), -a/2]; per 5 Hz step, the mean over its frames
    p5 = np.clip(pool5(s50, n), 1e-4, 1 - 1e-4)
    A5 = pool5(np.clip(ev, -20, 20), n)
    res = {}
    for i, (beta, gamma) in enumerate(beta_gamma):
        ee = np.asarray(e[:n], np.float64).copy()
        # the blend moves H against A only: normalised so that the sum over the two teams, the alive side, does not move
        L5 = beta * np.log(p5 / (1 - p5)) / 2
        c5 = np.logaddexp(L5, -L5) - np.log(2.0)
        ee[:, 0] += L5 - c5
        ee[:, 1] += L5 - c5
        ee[:, 2] += -L5 - c5
        ee[:, 3] += -L5 - c5
        ee[:, :4] += gamma * A5[:, None] / 2
        ee[:, 4] -= gamma * A5 / 2
        seg, off = ffbs(ee / tau, np.asarray(A, np.float64) / tau, M, seed + i)
        Pk = np.zeros((n, 3))
        for a0, b0, k in seg:
            Pk[a0:b0, k] += 1
        Pk /= M
        for lam in lams:
            lab_, ok = mb.solve(n, seg, off, Pk, lam, IOUS, mp_seg)
            out = np.repeat(lab_, STEP)
            n_f = len(map_path)
            out = np.r_[out, np.full(n_f - len(out), out[-1] if len(out) else 2, np.int8)] if ok else map_path
            res[(beta, gamma, lam)] = out.astype(np.int8)
    return map_path, res


def main():
    lab.setup()
    t0 = time.time()
    s = int(sys.argv[2])
    sets = sys.argv[3].split(",")
    bgs = [tuple(float(z) for z in x.split(":")) for x in os.environ.get("R88_BG", "0:0 0.5:0 1:0 0.5:0.25 1:0.25").split()]
    lams = tuple(float(x) for x in os.environ.get("R88_L", "0.3 0.4 0.5").split())
    tau = float(os.environ.get("R88_TAU", "1"))
    primary_rows = {}
    for K in sets:
        P = lab.get(K)
        hq, al = lab.stacker("Hy", s, K)
        A, Q = lab.mix(K, s, hq, al)
        E = emit(K, s, os.environ.get("R88_TAG", "Hy"))
        jobs = []
        for i in range(len(E)):
            n = min(len(A[i]), len(P["ev"][i]))
            jobs.append((E[i][0], E[i][1], A[i][:n], Q[i][:n], P["s50"][s][i][:n], P["ev"][i][:n], bgs, lams, 880000 + 100 * s + i, tau))
        with ThreadPoolExecutor(int(os.environ.get("R88_THREADS", "8"))) as pool:
            done = list(pool.map(lambda j: half_arms(*j), jobs))
        base = lab.score(K, [d[0] for d in done])
        primary_rows[K] = base
        print("%s v21 path: %s" % (K, " | ".join("%s %.4f" % (c, np.mean(list(v.values()))) for c, v in base.items())), flush=True)
        for bg in bgs:
            for lam in lams:
                r = lab.score(K, [d[1][(bg[0], bg[1], lam)] for d in done])
                ns = np.mean([np.sum((np.diff(d[1][(bg[0], bg[1], lam)]) != 0) & (d[1][(bg[0], bg[1], lam)][1:] < 2)) for d in done])
                nb = np.mean([np.sum((np.diff(d[0]) != 0) & (d[0][1:] < 2)) for d in done])
                print("  %s beta %.2f gamma %.2f lambda %.2f: %s; spells per half %.1f (v21 %.1f)" % (K, bg[0], bg[1], lam, " | ".join("%s %+.4f" % (c, np.mean(list(r[c].values())) - np.mean(list(base[c].values()))) for c in r), ns, nb), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
