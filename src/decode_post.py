"""Posterior version of decode2's explicit-duration decoder (round 18).

decode2.Spec's macro-states, sub-state chains and costs are read as a Gibbs distribution over labellings,
P(path) ~ exp((sum of emission log-scores - sum of costs) / tau): the Viterbi path is its mode. Forward-backward in
log space gives per-frame posteriors over the output labels (H, A, D), and forward-filtering backward-sampling draws
whole labellings from it (seeded, reproducible). Same state graph as decode2._run: a chain sub-state j advances to
j + 1 (cost adv), the last one may stay (adv), a segment may end from j >= min_exit (exit_cost) into the first
sub-state of another macro-state (pen). Time-varying channels are not supported (no decoder in use has them).
"""
import numpy as np
from numba import njit

NEG = -1e300


@njit(cache=True, nogil=True)
def _lse(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + np.log1p(np.exp(b - a))


@njit(cache=True, nogil=True)
def _forward(emis, emis_idx, offs, L, min_exit, exit_cost, adv, pen, init, tau, alpha):
    n = emis.shape[0]
    M = L.shape[0]
    S = offs[M]
    for s in range(S):
        alpha[0, s] = (emis[0, emis_idx[s]] - init[s]) / tau if init[s] < 1e290 else NEG
    ex = np.empty(M)
    for t in range(1, n):
        for m in range(M):
            acc = NEG
            for j in range(min_exit[m], L[m]):
                s = offs[m] + j
                acc = _lse(acc, alpha[t - 1, s] - exit_cost[s] / tau)
            ex[m] = acc
        for m in range(M):
            s0 = offs[m]
            acc = NEG
            for m2 in range(M):
                if m2 != m and pen[m2, m] < 1e290:
                    acc = _lse(acc, ex[m2] - pen[m2, m] / tau)
            if L[m] == 1:
                acc = _lse(acc, alpha[t - 1, s0] - adv[s0] / tau)
            alpha[t, s0] = acc + emis[t, emis_idx[s0]] / tau
            for j in range(1, L[m]):
                s = s0 + j
                acc = alpha[t - 1, s - 1] - adv[s - 1] / tau
                if j == L[m] - 1:
                    acc = _lse(acc, alpha[t - 1, s] - adv[s] / tau)
                alpha[t, s] = acc + emis[t, emis_idx[s]] / tau


@njit(cache=True, nogil=True)
def _backward_marginals(emis, emis_idx, out, offs, L, min_exit, exit_cost, adv, pen, tau, alpha, n_out, post):
    """post[t, k] = P(label_t = k); beta is kept for one frame only."""
    n = emis.shape[0]
    M = L.shape[0]
    S = offs[M]
    beta = np.zeros(S)
    nb = np.empty(S)
    logz = NEG
    for s in range(S):
        logz = _lse(logz, alpha[n - 1, s])
    E = np.empty(M)
    for t in range(n - 1, -1, -1):
        tot = np.full(n_out, NEG)
        for s in range(S):
            tot[out[s]] = _lse(tot[out[s]], alpha[t, s] + beta[s] - logz)
        for k in range(n_out):
            post[t, k] = np.exp(tot[k])
        if t == 0:
            break
        for m in range(M):
            s0 = offs[m]
            E[m] = emis[t, emis_idx[s0]] / tau + beta[s0]
        for m in range(M):
            s0 = offs[m]
            ent = NEG
            for m2 in range(M):
                if m2 != m and pen[m, m2] < 1e290:
                    ent = _lse(ent, E[m2] - pen[m, m2] / tau)
            for j in range(L[m]):
                s = s0 + j
                acc = NEG
                if j < L[m] - 1:
                    acc = _lse(acc, -adv[s] / tau + emis[t, emis_idx[s + 1]] / tau + beta[s + 1])
                else:
                    acc = _lse(acc, -adv[s] / tau + emis[t, emis_idx[s]] / tau + beta[s])
                if j >= min_exit[m]:
                    acc = _lse(acc, -exit_cost[s] / tau + ent)
                nb[s] = acc
        for s in range(S):
            beta[s] = nb[s]
    return logz


@njit(cache=True, nogil=True)
def _sample(offs, L, min_exit, exit_cost, adv, pen, macro, tau, alpha, k, seed, paths):
    """k labellings (as sub-state paths) by forward-filtering backward-sampling."""
    np.random.seed(seed)
    n = alpha.shape[0]
    M = L.shape[0]
    S = offs[M]
    w = np.empty(S)
    for r in range(k):
        mx = NEG
        for s in range(S):
            mx = max(mx, alpha[n - 1, s])
        tot = 0.0
        for s in range(S):
            w[s] = np.exp(alpha[n - 1, s] - mx)
            tot += w[s]
        u = np.random.random() * tot
        s_next = S - 1
        for s in range(S):
            u -= w[s]
            if u <= 0:
                s_next = s
                break
        paths[r, n - 1] = s_next
        for t in range(n - 2, -1, -1):
            m = macro[s_next]
            j = s_next - offs[m]
            if j >= 1:
                a = alpha[t, s_next - 1] - adv[s_next - 1] / tau
                if j == L[m] - 1:
                    b = alpha[t, s_next] - adv[s_next] / tau
                    u = np.random.random()
                    pa = 1.0 / (1.0 + np.exp(b - a)) if a >= b else np.exp(a - b) / (1.0 + np.exp(a - b))
                    s_next = s_next - 1 if u < pa else s_next
                else:
                    s_next = s_next - 1
            else:
                cnt = 0
                mx = NEG
                for m2 in range(M):
                    if m2 == m or pen[m2, m] > 1e290:
                        continue
                    for j2 in range(min_exit[m2], L[m2]):
                        s = offs[m2] + j2
                        v = alpha[t, s] - (exit_cost[s] + pen[m2, m]) / tau
                        w[cnt] = v
                        cnt += 1
                        mx = max(mx, v)
                stay = NEG
                if L[m] == 1:
                    stay = alpha[t, s_next] - adv[s_next] / tau
                    mx = max(mx, stay)
                tot = 0.0
                for i in range(cnt):
                    w[i] = np.exp(w[i] - mx)
                    tot += w[i]
                ws = np.exp(stay - mx) if stay > NEG else 0.0
                u = np.random.random() * (tot + ws)
                if u >= tot:
                    paths[r, t] = s_next
                    continue
                chosen = -1
                i = 0
                for m2 in range(M):
                    if m2 == m or pen[m2, m] > 1e290:
                        continue
                    for j2 in range(min_exit[m2], L[m2]):
                        u -= w[i]
                        i += 1
                        if u <= 0 and chosen < 0:
                            chosen = offs[m2] + j2
                s_next = chosen if chosen >= 0 else offs[M - 1]
            paths[r, t] = s_next


def _forward_table(spec, emis, tau):
    emis = np.ascontiguousarray(emis, np.float64)
    alpha = np.empty((len(emis), int(spec.offs[-1])))
    _forward(emis, spec.emis_idx, spec.offs, spec.L, spec.min_exit, spec.exit_cost, spec.adv, spec.pen, spec.init,
             float(tau), alpha)
    return emis, alpha


def marginals(spec, emis, tau=1.0):
    """(n, 3) posterior probabilities of the output labels (0 H, 1 A, 2 D) and log Z."""
    emis, alpha = _forward_table(spec, emis, tau)
    n_out = int(spec.out.max()) + 1
    post = np.empty((len(emis), n_out))
    logz = _backward_marginals(emis, spec.emis_idx, spec.out.astype(np.int64), spec.offs, spec.L, spec.min_exit,
                               spec.exit_cost, spec.adv, spec.pen, float(tau), alpha, n_out, post)
    return post, logz


def sample(spec, emis, k, tau=1.0, seed=0):
    """(k, n) labellings drawn from the posterior (output labels, int8)."""
    emis, alpha = _forward_table(spec, emis, tau)
    paths = np.empty((k, len(emis)), np.int64)
    _sample(spec.offs, spec.L, spec.min_exit, spec.exit_cost, spec.adv, spec.pen, spec.macro.astype(np.int64),
            float(tau), alpha, int(k), int(seed), paths)
    return spec.out[paths]
