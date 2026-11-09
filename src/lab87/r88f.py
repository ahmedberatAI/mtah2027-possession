"""Round 88: now that the path model's joint posterior decides, its quality has a proper scoring rule of its own: the held-out path likelihood. Per stacker tag and seed, mean negative log-likelihood per step of the true path (free state summed out, steps with a mixed label left free), on each open set under the primary reading. Lower is better. Open data only, aggregate output.

    python -I r88f.py "Hy:0 Hy:1 Hy:2 Hyw:0"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402
from numba import njit  # noqa: E402
from r88 import STEP, emit  # noqa: E402


@njit(cache=True, nogil=True)
def logz(e, A, mask):
    """log-sum over paths; mask (T, 5) adds -inf-like to states that disagree with the label; returns log Z."""
    T, S = e.shape
    al = e[0] + mask[0]
    new = np.zeros(S)
    for t in range(1, T):
        for s in range(S):
            x = al + A[:, s]
            m = x.max()
            new[s] = m + np.log(np.exp(x - m).sum()) + e[t, s] + mask[t, s]
        al[:] = new
    m = al.max()
    return m + np.log(np.exp(al - m).sum())


def step_labels(y, n):
    y = np.asarray(y)[:n * STEP].reshape(-1, STEP)
    out = np.full(len(y), -1)
    for k in (0, 1, 2):
        out[(y == k).sum(1) >= 3] = k
    return out


def nll(E, tr):
    """Sum of (log Z - log Z(label)) and steps."""
    tot, steps = 0.0, 0
    for (e, A), t in zip(E, tr):
        n = min(len(e), len(t) // STEP)
        e_ = np.asarray(e[:n], np.float64)
        A_ = np.asarray(A, np.float64)
        y = step_labels(t, n)
        S = e_.shape[1]
        K = (S - 1) // 2
        mask = np.full((n, S), -1e4)
        mask[y == 0, 0:K] = 0.0
        mask[y == 1, K:2 * K] = 0.0
        mask[y == 2, 2 * K] = 0.0
        mask[y < 0] = 0.0
        z = logz(e_, A_, np.zeros((n, S)))
        zl = logz(e_, A_, mask)
        tot += z - zl
        steps += n
    return tot / steps


def main():
    lab.setup()
    primary = {"dfl": "evctrl", "metrica": "ctrl", "sc": "evctrl", "st": "evctrl"}
    for item in sys.argv[1].split():
        tag, s = item.split(":")
        vals = []
        for K in lab.SETS:
            vals.append("%s %.4f" % (K, nll(emit(K, int(s), tag), lab.get(K)["truth"][primary[K]])))
        print("%s seed %s: %s" % (tag, s, " | ".join(vals)), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
