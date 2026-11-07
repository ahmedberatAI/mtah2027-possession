"""Round 73 (docs/RAPOR.md): choose the spells that maximise the expected score, not the most likely path. v21's Viterbi path answers "which labelling is most likely"; the metric asks "how many possessions match". With F1 near 0.8, a spell is worth predicting when its chance of matching a true spell passes about 0.4, and how much it is worth depends on its neighbours (a missed spell also merges the two around it). A single penalty cannot express that. The evidence (the mix's q and alive evidence, calibrated, round 70) with the decoder's costs at a temperature T defines a distribution over whole segmentations: a semi-Markov model at 5 Hz, sampled exactly. Every spell some sample used gets its chance of matching at the five IoU thresholds from the samples; dynamic programming picks the full labelling with the largest sum of (chance - lambda). D only where the samples say dead: nothing writes D into live play to split a spell. No training. Aggregate output only.

  python -I src/exp_mbr.py grid SEED_INDEX "SETS"      T and lambda against v21's path on the open sets
"""
import os
import pickle
import sys
import warnings

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP = 5
DMAX = 900               # steps (180 s); a longer segment is cut in a sample, which is rare
DMIN_P, DMIN_D = 2, 10   # the decoder's minimum durations, 8 and 50 frames, at this step
PEN_T, PEN_D, C_R, R0 = 6.0, 30.0, 24.0, 175.0
M = int(os.environ.get("MBR_SAMPLES", "200"))
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
TS = tuple(float(x) for x in os.environ.get("MBR_T", "10 15 25 40").split())
LS = tuple(float(x) for x in os.environ.get("MBR_L", "0.4").split())


@njit(cache=True, nogil=True)
def _lae(a, b):
    if a > b:
        return a + np.log1p(np.exp(b - a)) if a - b < 40.0 else a
    return b + np.log1p(np.exp(a - b)) if b - a < 40.0 else b


@njit(cache=True, nogil=True)
def _exit(k, d, c, r0):
    if k == 2:
        return 0.0
    x = 1.0 - d * 5.0 / r0
    return c * x if x > 0 else 0.0


@njit(cache=True, nogil=True)
def forward(C, tm, pt, pd, c, r0, dmax, dp, dd):
    """C (n + 1, 3) cumulative emission per step. al[t, k]: log-sum of labellings of [0, t) whose last segment is k; pre[t, k]: ready to start k at t."""
    n = C.shape[0] - 1
    al = np.full((n + 1, 3), -np.inf)
    pre = np.full((n + 1, 3), -np.inf)
    pre[0, :] = 0.0
    for t in range(1, n + 1):
        for k in range(3):
            dmin = dp if k < 2 else dd
            acc = -np.inf
            for d in range(dmin, min(dmax, t) + 1):
                a = t - d
                if pre[a, k] == -np.inf:
                    continue
                v = pre[a, k] + (C[t, k] - C[a, k] - _exit(k, d, c, r0)) / tm
                acc = _lae(acc, v)
            al[t, k] = acc
        for k in range(3):
            v = -np.inf
            for j in range(3):
                if j == k or al[t, j] == -np.inf:
                    continue
                v = _lae(v, al[t, j] - (pt if (j < 2 and k < 2) else pd) / tm)
            pre[t, k] = v
    return al, pre


@njit(cache=True, nogil=True)
def sample(C, al, pre, tm, pt, pd, c, r0, dmax, dp, dd, m, seed):
    """m labellings from the exact posterior: (segments (start, end, k), offsets)."""
    np.random.seed(seed)
    n = C.shape[0] - 1
    cap = 8000 * m
    seg = np.zeros((cap, 3), np.int32)
    off = np.zeros(m + 1, np.int64)
    w = 0
    for s in range(m):
        t = n
        best, k = -np.inf, 0
        for j in range(3):
            v = al[n, j] - np.log(-np.log(np.random.random() + 1e-300))
            if v > best:
                best, k = v, j
        start = w
        while t > 0 and w < cap:
            dmin = dp if k < 2 else dd
            best = -np.inf
            pick = -1
            for d in range(dmin, min(dmax, t) + 1):
                a = t - d
                if pre[a, k] == -np.inf:
                    continue
                v = pre[a, k] + (C[t, k] - C[a, k] - _exit(k, d, c, r0)) / tm - np.log(-np.log(np.random.random() + 1e-300))
                if v > best:
                    best, pick = v, d
            if pick < 0:          # a dead end of the minimum durations at the very start of a half
                pick = t
            a = t - pick
            seg[w, 0], seg[w, 1], seg[w, 2] = a, t, k
            w += 1
            if a == 0:
                break
            best = -np.inf
            nk = (k + 1) % 3
            for j in range(3):
                if j == k or al[a, j] == -np.inf:
                    continue
                v = al[a, j] - (pt if (j < 2 and k < 2) else pd) / tm - np.log(-np.log(np.random.random() + 1e-300))
                if v > best:
                    best, nk = v, j
            k = nk
            t = a
        # this sample's segments were written backwards
        seg[start:w] = seg[start:w][::-1].copy()
        off[s + 1] = w
    return seg[:w], off


@njit(cache=True, nogil=True)
def match_counts(x, y, k, seg, off, ious):
    """In how many samples a segment of team k matches (x, y) at each threshold."""
    m = off.shape[0] - 1
    res = np.zeros(ious.shape[0], np.int32)
    for s in range(m):
        lo, hi = off[s], off[s + 1]
        # first segment ending after x
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
def rewards(cand, seg, off, ious, lam, cum):
    """Per candidate segment (x, y, k): spells: mean chance of matching - lam; dead: 0, only where the samples say dead; -inf = not allowed."""
    m = off.shape[0] - 1
    out = np.full(cand.shape[0], -np.inf)
    for i in range(cand.shape[0]):
        x, y, k = cand[i, 0], cand[i, 1], cand[i, 2]
        share = (cum[y, k] - cum[x, k]) / (y - x)
        if k == 2:
            if share >= 0.5 or cand[i, 3] == 1:
                out[i] = 0.0
            continue
        c_ = match_counts(x, y, k, seg, off, ious)
        out[i] = np.mean(c_ / m) - lam
    return out


@njit(cache=True, nogil=True)
def chain(n, cand, r):
    """The full labelling of [0, n) from the candidates with the largest sum of rewards; candidates sorted by end."""
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


def solve(n, seg, off, Pk, lam, ious, mp_seg):
    """Candidates: every segment some sample used, and the MAP path's (always allowed, so a labelling exists)."""
    both = np.vstack([seg, mp_seg])
    key = both[:, 2].astype(np.int64) + 3 * (both[:, 1].astype(np.int64) + (n + 1) * both[:, 0].astype(np.int64))
    uniq, idx = np.unique(key, return_index=True)
    cand = np.zeros((len(uniq), 4), np.int64)
    cand[:, :3] = both[idx]
    mk = mp_seg[:, 2].astype(np.int64) + 3 * (mp_seg[:, 1].astype(np.int64) + (n + 1) * mp_seg[:, 0].astype(np.int64))
    cand[np.isin(uniq, mk), 3] = 1
    cand = cand[np.argsort(cand[:, 1], kind="stable")]
    cum = np.zeros((n + 1, 3))
    cum[1:] = np.cumsum(Pk, axis=0)
    return chain(n, cand, rewards(cand, seg, off, ious, lam, cum))


def half_arms(a, q, ts, ls, seed):
    """{(T, lambda): labels at 25 fps}, the MAP path's boundaries among the candidates."""
    import decode2
    import exp_v7
    n_f = min(len(a), len(q))
    q_ = np.clip(np.asarray(q, np.float64)[:n_f], 1e-4, 1 - 1e-4)
    a_ = np.asarray(a, np.float64)[:n_f]
    S = np.stack([a_ / 2 + np.log(q_), a_ / 2 + np.log(1 - q_), -a_ / 2], 1)
    map_path = decode2.duration_spec(PEN_T, PEN_D, 8, 50, exp_v7.ramp(175, C_R)).decode(S).astype(np.int8)
    n = n_f // STEP
    C = np.zeros((n + 1, 3))
    C[1:] = np.cumsum(S[:n * STEP].reshape(n, STEP, 3).sum(1), axis=0)
    mp = map_path[STEP // 2::STEP][:n]
    st = np.r_[0, np.flatnonzero(mp[1:] != mp[:-1]) + 1]
    en = np.r_[st[1:], n]
    mp_seg = np.stack([st, en, mp[st]], 1).astype(np.int32)
    res = {}
    for t_i, tm in enumerate(ts):
        al, pre = forward(C, tm, PEN_T, PEN_D, C_R, R0, DMAX, DMIN_P, DMIN_D)
        seg, off = sample(C, al, pre, tm, PEN_T, PEN_D, C_R, R0, DMAX, DMIN_P, DMIN_D, M, seed + t_i)
        Pk = np.zeros((n, 3))
        for a0, b0, k in seg:
            Pk[a0:b0, k] += 1
        Pk /= M
        for lam in ls:
            lab, ok = solve(n, seg, off, Pk, lam, IOUS, mp_seg)
            out = np.repeat(lab, STEP)
            out = np.r_[out, np.full(n_f - len(out), out[-1] if len(out) else 2, np.int8)] if ok else map_path
            res[(tm, lam)] = out.astype(np.int8)
    return map_path, res


def load_set(k, s_i):
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_stack as xs
    from exp_stack import OUT, bench, ec, ee, eg2, exp_smooth, ss, torch
    sys.argv = _argv
    import exp_deadmix as dm
    if k == "st":
        import exp_st
        fsrc, src, convs = "st@v6", exp_st.SRC, tuple(exp_st.CONVS)
        d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s_i))
        s50 = d["s50"]
        ck = torch.load(os.path.join(OUT, "sall_Hy_%d.pt" % s_i), map_location="cpu", weights_only=False)
        device = ec.pick_device()
        net = xs.make_net("Hy", ck["d_in"], ck.get("e", 0)).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        zs = []
        for r, (q_, a_) in zip(d["rows"], xs.bf_mean("st@v3", (0, 1, 2))):
            z = ec.pool(xs._bf_cols(q_, a_).astype(np.float64)).astype(np.float32)
            n_ = min(len(r[1]), len(z))
            zs.append(xs.score_half(net, np.hstack([r[1][:n_], z[:n_]]), r[3], device, xs.step_of("Hy"), "Hy"))
        hq, al = [z[0] for z in zs], [z[1] for z in zs]
    else:
        fsrc, src, convs = ee.V6[k][0], ee.V3[k][1], tuple(dm.CONVS[ee.V3[k][1]])
        v, g, o = eg2.parts(src, s_i, True)
        s50 = exp_smooth.smooth([0.25 * np.asarray(x, np.float64) + 0.25 * np.asarray(y, np.float64) + 0.5 * ss.pk(np.asarray(z, np.float32), 0) for x, y, z in zip(v, g, o)], 25)
        h = bench._load(os.path.join(OUT, "sheld_Hy_%d_%s.pkl" % (s_i, k)))
        hq, al = h["q"], h["alive"]
    ev = bench._load(os.path.join(OUT, "evidence_%s.pkl" % ("st" if k == "st" else k)))["ev"]
    q = xs._mix(hq, s50, 0.5)
    a = [0.5 * np.asarray(e)[:len(x)] + 0.5 * np.asarray(x) for e, x in zip(ev, al)]
    return bench, fsrc, src, convs, a, q


def grid(s_i, sets):
    from concurrent.futures import ThreadPoolExecutor
    _argv = list(sys.argv)
    sys.argv = [sys.argv[0], "", "none"]
    import exp_blend_weight as bw
    from exp_stack import er
    sys.argv = _argv
    s_i = int(s_i)
    names = ["path"] + ["T%g l%g" % (t, l) for t in TS for l in LS]
    cells = {}
    half_arms(np.zeros(2000), np.full(2000, 0.5), (25.0,), (0.4,), 0)      # compile before the threads
    for k in sets:
        bench, fsrc, src, convs, a, q = load_set(k, s_i)
        blocks = bench.blocks(fsrc)
        labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}

        def one(i):
            return half_arms(a[i], q[i], TS, LS, 20261130 + 1000 * s_i + i)
        with ThreadPoolExecutor(int(os.environ.get("MBR_WORKERS", "8"))) as pool:
            done = list(pool.map(one, range(len(a))))
        arms = {"path": [d[0] for d in done]}
        for t in TS:
            for l in LS:
                arms["T%g l%g" % (t, l)] = [d[1][(t, l)] for d in done]
        for name, preds in arms.items():
            preds = [np.asarray(p)[:len(labs[convs[0]][i])] for i, p in enumerate(preds)]
            lab0 = [np.asarray(labs[convs[0]][i])[:len(preds[i])] for i in range(len(preds))]
            for c in convs:
                key = (src, c, "v6" if k == "st" else "edges D", name)
                cells.setdefault(key, []).append(er.by_match(blocks, [np.asarray(labs[c][i])[:len(preds[i])] for i in range(len(preds))], preds))
            spells = np.mean([np.sum(np.diff(np.r_[2, p]) != 0) for p in preds])
            print("  %-8s %-12s segments per half %.1f" % (k, name, spells), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    with open(os.path.join(bench.ROOT, "tools", "runs", "r73_cells_%d_%s.pkl" % (s_i, "".join(s[0] for s in sets))), "wb") as fh:
        pickle.dump(cells, fh)
    mean, _ = er.summarize(cells, names, ref="path")
    if {"dfl", "metrica", "sc"} <= set(sets):
        for cand in names[1:]:
            bw.decide(mean, "path", cand)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "grid":
        grid(sys.argv[2], sys.argv[3].split() if len(sys.argv) > 3 else ("dfl", "metrica", "sc"))
    else:
        raise SystemExit(__doc__)
