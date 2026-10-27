"""Generalised segment decoder for the decoder ideas in docs/RAPOR.md (B1 restart-aware, B2 duration-aware,
B4 typed penalties, B5 context-dependent switch costs).

Macro-states m = 0..M-1. Each has an emission column (0 H, 1 A, 2 D), an output label and a chain of L[m]
sub-states: sub-state j -> j + 1 costs adv[m][j]; the last sub-state may stay (cost adv[m][L - 1]); the
segment may end from any sub-state j >= min_exit[m] at an extra exit_cost[m][j]. Entering macro m from m2
costs pen[m2, m] (inf = forbidden) plus, when ch[m2, m] >= 0, chan[t, ch[m2, m]] at the first frame t of
the new segment (time-varying costs). With L = min_dur, min_exit = L - 1, zero adv / exit costs and the
three macro-states H, A, D this is exactly decode.viterbi_min_dur.

Scores are rewards (log-probabilities), costs are positive.
"""
import numpy as np
from numba import njit

INF = 1e300


@njit(cache=True, nogil=True)
def _run(emis, emis_idx, offs, L, min_exit, exit_cost, adv, pen, ch, chan, init, back):
    n = emis.shape[0]
    M = L.shape[0]
    S = offs[M]
    NEG = -1e300
    delta = np.full(S, NEG)
    new = np.full(S, NEG)
    for s in range(S):
        if init[s] < 1e290:
            delta[s] = emis[0, emis_idx[s]] - init[s]
    ex = np.empty(M)
    exarg = np.zeros(M, np.int64)
    for t in range(1, n):
        for m in range(M):
            best = NEG
            arg = offs[m] + L[m] - 1
            for j in range(min_exit[m], L[m]):
                s = offs[m] + j
                v = delta[s] - exit_cost[s]
                if v > best:
                    best = v
                    arg = s
            ex[m] = best
            exarg[m] = arg
        for m in range(M):
            s0 = offs[m]
            best = NEG
            arg = s0
            for m2 in range(M):
                if m2 == m:
                    continue
                c = pen[m2, m]
                if c > 1e290:
                    continue
                k = ch[m2, m]
                if k >= 0:
                    c += chan[t, k]
                v = ex[m2] - c
                if v > best:
                    best = v
                    arg = exarg[m2]
            if L[m] == 1:
                v = delta[s0] - adv[s0]
                if v > best:
                    best = v
                    arg = s0
            new[s0] = best + emis[t, emis_idx[s0]]
            back[t, s0] = arg
            for j in range(1, L[m]):
                s = s0 + j
                a_ = delta[s - 1] - adv[s - 1]
                if j == L[m] - 1:
                    b_ = delta[s] - adv[s]
                    if b_ >= a_:
                        new[s] = b_ + emis[t, emis_idx[s]]
                        back[t, s] = s
                        continue
                new[s] = a_ + emis[t, emis_idx[s]]
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
        path[t] = s
        s = back[t, s]
    return path


class Spec:
    """macros: list of dict(emis, out, L, min_exit=L-1, exit_cost=None, adv=None); pen (M, M) costs (np.inf =
    forbidden, diagonal ignored); ch (M, M) channel indices into chan (-1 = none)."""

    def __init__(self, macros, pen, ch=None):
        M = len(macros)
        self.M = M
        self.L = np.array([int(m["L"]) for m in macros], np.int64)
        self.offs = np.concatenate(([0], np.cumsum(self.L))).astype(np.int64)
        S = int(self.offs[-1])
        self.emis_idx = np.zeros(S, np.int64)
        self.out = np.zeros(S, np.int8)
        self.macro = np.zeros(S, np.int64)
        self.min_exit = np.array([int(m.get("min_exit", m["L"] - 1)) for m in macros], np.int64)
        self.exit_cost = np.zeros(S)
        self.adv = np.zeros(S)
        self.init = np.full(S, INF)
        for i, m in enumerate(macros):
            a, b = self.offs[i], self.offs[i + 1]
            self.emis_idx[a:b] = m["emis"]
            self.out[a:b] = m["out"]
            self.macro[a:b] = i
            if m.get("exit_cost") is not None:
                self.exit_cost[a:b] = m["exit_cost"]
            if m.get("adv") is not None:
                self.adv[a:b] = m["adv"]
            # like decode.py: a half may start in the first or (a cut segment) the last sub-state
            self.init[a] = 0.0
            self.init[b - 1] = 0.0
        pen = np.array(pen, np.float64)
        self.pen = np.where(np.isfinite(pen), pen, INF)
        self.ch = np.full((M, M), -1, np.int64) if ch is None else np.asarray(ch, np.int64)

    def decode(self, emis, chan=None, return_macro=False):
        emis = np.ascontiguousarray(emis, np.float64)
        n = len(emis)
        if chan is None:
            chan = np.zeros((n, 1))
        chan = np.ascontiguousarray(chan, np.float64)
        S = int(self.offs[-1])
        back = np.empty((n, S), np.int16 if S < 32000 else np.int32)
        path = _run(emis, self.emis_idx, self.offs, self.L, self.min_exit, self.exit_cost, self.adv, self.pen,
                    self.ch, chan, self.init, back)
        if return_macro:
            return self.out[path], self.macro[path]
        return self.out[path]


def std_spec(pen_team, pen_dead, min_poss, min_dead, pen_in=None, pen_out=None, ch_team=-1):
    """H, A, D with min durations; B4: separate costs for entering / leaving D; B5: channel on H <-> A."""
    pi = pen_dead if pen_in is None else pen_in
    po = pen_dead if pen_out is None else pen_out
    macros = [dict(emis=0, out=0, L=min_poss), dict(emis=1, out=1, L=min_poss), dict(emis=2, out=2, L=min_dead)]
    pen = [[0, pen_team, pi], [pen_team, 0, pi], [po, po, 0]]
    ch = [[-1, ch_team, -1], [ch_team, -1, -1], [-1, -1, -1]]
    return Spec(macros, pen, ch)


def restart_spec(pen_team, pen_in, pen_same, pen_switch, min_poss, min_dead, ch_restart=True, ch_team=-1):
    """B1: D split by the team that had the ball before the stoppage (macros H, A, D|H, D|A; both D output D).
    Channels (when ch_restart): 0 = restart by the same team, 1 = restart by the other team; ch_team on H <-> A."""
    inf = np.inf
    macros = [dict(emis=0, out=0, L=min_poss), dict(emis=1, out=1, L=min_poss),
              dict(emis=2, out=2, L=min_dead), dict(emis=2, out=2, L=min_dead)]
    pen = [[0, pen_team, pen_in, inf],
           [pen_team, 0, inf, pen_in],
           [pen_same, pen_switch, 0, inf],
           [pen_switch, pen_same, inf, 0]]
    s, w = (0, 1) if ch_restart else (-1, -1)
    ch = [[-1, ch_team, -1, -1], [ch_team, -1, -1, -1], [s, w, -1, -1], [w, s, -1, -1]]
    return Spec(macros, pen, ch)


def duration_spec(pen_team, pen_dead, min_poss, min_dead, poss_exit, dead_exit=None, poss_adv=None,
                  dead_adv=None, ch_team=-1):
    """B2: explicit-duration chains. poss_exit[j] = extra cost of ending a possession spell after j + 1 frames
    (len = chain length, exits allowed from min_poss - 1 on); optional per-frame stay costs."""
    Lp = len(poss_exit)
    macros = [dict(emis=0, out=0, L=Lp, min_exit=min_poss - 1, exit_cost=poss_exit, adv=poss_adv),
              dict(emis=1, out=1, L=Lp, min_exit=min_poss - 1, exit_cost=poss_exit, adv=poss_adv)]
    if dead_exit is None:
        macros.append(dict(emis=2, out=2, L=min_dead))
    else:
        macros.append(dict(emis=2, out=2, L=len(dead_exit), min_exit=min_dead - 1, exit_cost=dead_exit,
                           adv=dead_adv))
    pen = [[0, pen_team, pen_dead], [pen_team, 0, pen_dead], [pen_dead, pen_dead, 0]]
    ch = [[-1, ch_team, -1], [ch_team, -1, -1], [-1, -1, -1]]
    return Spec(macros, pen, ch)
