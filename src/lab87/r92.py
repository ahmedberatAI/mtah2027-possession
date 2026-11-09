"""Round 92: where does v22 still lose? The error budget of round 70 on v22's own output (mixture of three seeds, M 3000, lambda .5), each oracle fixing one kind of error: the true dead/alive call, painting missed possessions by length and place, dropping made-up ones, the timing of team changes, the true team on live frames. Same functions as round 70 (scratchpad lit60/budget21.py). Open data only, aggregate output."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402
from r88 import emit  # noqa: E402

FPS = 25


def poss(lab_):
    import metrica_sample as ms
    sp = ms.spells(lab_)
    out = []
    for i, (a, b, k) in enumerate(sp):
        pdead = i == 0 or sp[i - 1][1] != a
        ndead = i + 1 == len(sp) or sp[i + 1][0] != b
        cat = ("between stops" if ndead else "after restart") if pdead else ("before stop" if ndead else "squeezed")
        out.append((a, b, k, cat))
    return out


def iou(a, b, x, y):
    inter = min(b, y) - max(a, x)
    return inter / (max(b, y) - min(a, x)) if inter > 0 else 0.0


def best(a, b, k, others):
    return max([iou(a, b, x, y) for x, y, kk, _ in others if kk == k and y > a and x < b] + [0.0])


def paint(pred, tp, cats, lo=0, hi=4):
    out = np.asarray(pred).copy()
    pp = poss(out)
    for a, b, k, cat in tp:
        if cat in cats and lo <= (b - a) / FPS < hi and best(a, b, k, pp) < 0.5:
            out[a:b] = k
    return out


def drop_false(pred, tp, lo, hi):
    out = np.asarray(pred).copy()
    for a, b, k, _ in poss(out):
        if not lo <= (b - a) / FPS < hi or best(a, b, k, tp) >= 0.3:
            continue
        prev = out[a - 1] if a > 0 else 3
        nxt = out[b] if b < len(out) else 3
        if (prev == 1 - k and nxt in (1 - k, 2)) or (nxt == 1 - k and prev == 2):
            out[a:b] = 1 - k
    return out


def flip_timing(pred, g, win=50):
    p = np.asarray(pred).copy()
    gg = np.asarray(g)
    tc = np.flatnonzero((gg[1:] != gg[:-1]) & (gg[1:] < 2) & (gg[:-1] < 2)) + 1
    tk = gg[tc]
    allb = np.flatnonzero(p[1:] != p[:-1]) + 1
    out = p.copy()
    for j, f in enumerate(allb):
        if p[f] > 1 or p[f - 1] > 1:
            continue
        cand = tc[(tk == p[f]) & (np.abs(tc - f) <= win)]
        if not len(cand):
            continue
        t = int(cand[np.argmin(np.abs(cand - f))])
        lo = allb[j - 1] if j > 0 else 0
        hi = allb[j + 1] if j + 1 < len(allb) else len(p)
        if lo < t < hi and t != f:
            if t < f:
                out[t:f] = p[f]
            else:
                out[f:t] = p[f - 1]
    return out


def true_dead(pred, g):
    """The truth's dead/alive call; frames that become alive take the nearest possession of the prediction."""
    import stack_decode as sd
    out = np.asarray(pred).copy()
    out[g == 2] = 2
    fill = (g < 2) & (out == 2)
    if fill.any():
        tmp = out.copy()
        tmp[fill] = 2
        out2 = sd.keep_reference_dead(tmp, np.where(fill, 0, 2))
        out[fill] = out2[fill]
        out[(g < 2) & (out == 2)] = g[(g < 2) & (out == 2)]
    return out


def main():
    lab.setup()
    import stack_decode as sd
    t0 = time.time()
    primary = {"dfl": "evctrl", "metrica": "ctrl", "sc": "evctrl", "st": "evctrl"}
    for K in lab.SETS:
        P = lab.get(K)
        E = {s: emit(K, s) for s in (0, 1, 2)}
        mixes = {s: lab.mix(K, s, *lab.stacker("Hy", s, K)) for s in (0, 1, 2)}
        bagA, bagQ = lab.mean_paths([mixes[s] for s in (0, 1, 2)])
        ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
        with ThreadPoolExecutor(8) as pool:
            dec = list(pool.map(lambda i: sd.decide([E[s][i] for s in (0, 1, 2)], ref[i], 0.5, 3000, 890000 + i), range(len(ref))))
        tr = P["truth"][primary[K]]
        ns = [min(len(a), len(b)) for a, b in zip(tr, dec)]
        tr = [a[:n] for a, n in zip(tr, ns)]
        dec = [np.asarray(b)[:n] for b, n in zip(dec, ns)]
        import exp_refine as er

        def sc(pr):
            return float(np.mean(list(er.by_match(P["groups"], tr, pr).values())))
        base = sc(dec)
        tp = [poss(g) for g in tr]
        arms = {
            "true dead/alive": [true_dead(p, g) for p, g in zip(dec, tr)],
            "paint missed < 4 s": [paint(p, t, ("squeezed", "after restart", "before stop", "between stops"), 0, 4) for p, t in zip(dec, tp)],
            "  of which squeezed": [paint(p, t, ("squeezed",), 0, 4) for p, t in zip(dec, tp)],
            "paint missed 4-8 s": [paint(p, t, ("squeezed", "after restart", "before stop", "between stops"), 4, 8) for p, t in zip(dec, tp)],
            "paint missed 8+ s": [paint(p, t, ("squeezed", "after restart", "before stop", "between stops"), 8, 1e9) for p, t in zip(dec, tp)],
            "drop made-up 0-4 s": [drop_false(p, t, 0, 4) for p, t in zip(dec, tp)],
            "drop made-up 4-8 s": [drop_false(p, t, 4, 8) for p, t in zip(dec, tp)],
            "timing of team changes": [flip_timing(p, g) for p, g in zip(dec, tr)],
            "true team on live frames": [np.where((g < 2) & (p < 2), g, p) for p, g in zip(dec, tr)],
        }
        lines = ["%s %s: v22 %.4f" % (K, primary[K], base)]
        for name, pr in arms.items():
            lines.append("   %-26s %+.4f" % (name, sc(pr) - base))
        print("\n".join(lines), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
