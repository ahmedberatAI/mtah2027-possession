"""Round 93 (diagnosis, open data only, aggregate output): where v22's remaining loss sits in the game's structure.
Round 92 split the loss by length; this splits it by the possession's place around stoppages: after a restart,
before a stop, between two stops, squeezed between two possessions. For each place: paint the missed true
possessions (< 4 s and >= 4 s), and drop the made-up ones; and how often v22 gives the wrong team at a restart
(first second of a true possession that starts after a dead stretch) or at a stop (last second before one).
v22's decisions are cached (cache/lab87/dec_v22_<set>.pkl) for later rounds."""
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402
from r88 import emit  # noqa: E402
from r92 import best, poss  # noqa: E402

FPS = 25
CATS = ("after restart", "before stop", "between stops", "squeezed")
PRIMARY = {"dfl": "evctrl", "metrica": "ctrl", "sc": "evctrl", "st": "evctrl"}


def v22_decisions(K):
    path = os.path.join(lab.PACKS, "dec_v22_%s.pkl" % K)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import stack_decode as sd
    E = {s: emit(K, s) for s in (0, 1, 2)}
    mixes = {s: lab.mix(K, s, *lab.stacker("Hy", s, K)) for s in (0, 1, 2)}
    bagA, bagQ = lab.mean_paths([mixes[s] for s in (0, 1, 2)])
    ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
    with ThreadPoolExecutor(8) as pool:
        dec = list(pool.map(lambda i: sd.decide([E[s][i] for s in (0, 1, 2)], ref[i], 0.5, 3000, 890000 + i), range(len(ref))))
    out = dict(dec=[np.asarray(d, np.int8) for d in dec], ref=[np.asarray(r, np.int8) for r in ref])
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def paint_cat(pred, tp, cat, short):
    out = np.asarray(pred).copy()
    pp = poss(out)
    for a, b, k, c in tp:
        if c != cat or ((b - a) / FPS < 4) != short:
            continue
        if best(a, b, k, pp) < 0.5:
            out[a:b] = k
    return out


def drop_cat(pred, tp, cat):
    """Made-up predicted possessions (no same-team true possession at IoU >= 0.3) of one place: their frames go to the
    neighbouring possession's team, or to dead if both neighbours are dead."""
    out = np.asarray(pred).copy()
    for a, b, k, c in poss(out):
        if c != cat or best(a, b, k, tp) >= 0.3:
            continue
        prev = out[a - 1] if a > 0 else 2
        nxt = out[b] if b < len(out) else 2
        if prev < 2:
            out[a:b] = prev
        elif nxt < 2:
            out[a:b] = nxt
        else:
            out[a:b] = 2
    return out


def edge_team(pred, tp, where):
    """Restart (first second of true possessions after a dead stretch) or stop (last second before one): share of
    such possessions whose predicted live frames there are mostly the other team; the paint oracle for those only."""
    p = np.asarray(pred)
    out = p.copy()
    n_all = n_wrong = n_dead = 0
    for a, b, k, c in tp:
        if where == "restart" and c not in ("after restart", "between stops"):
            continue
        if where == "stop" and c not in ("before stop", "between stops"):
            continue
        w = slice(a, min(b, a + FPS)) if where == "restart" else slice(max(a, b - FPS), b)
        seg = p[w]
        live = seg[seg < 2]
        n_all += 1
        if len(live) < 0.5 * len(seg):
            n_dead += 1
            continue
        if np.mean(live == k) < 0.5:
            n_wrong += 1
            out[a:b] = k
    return out, n_all, n_wrong, n_dead


def main():
    lab.setup()
    import exp_refine as er
    t0 = time.time()
    for K in lab.SETS:
        P = lab.get(K)
        dec = v22_decisions(K)["dec"]
        tr = P["truth"][PRIMARY[K]]
        ns = [min(len(a), len(b)) for a, b in zip(tr, dec)]
        tr = [a[:n] for a, n in zip(tr, ns)]
        dec = [np.asarray(b)[:n] for b, n in zip(dec, ns)]

        def sc(pr):
            return float(np.mean(list(er.by_match(P["groups"], tr, pr).values())))
        base = sc(dec)
        tp = [poss(g) for g in tr]
        pp = [poss(d) for d in dec]
        lines = ["%s %s: v22 %.4f, halves %d" % (K, PRIMARY[K], base, len(dec))]
        # counts per half: true possessions by place and length, how many v22 finds (IoU >= 0.5), made-up ones
        for cat in CATS:
            for short in (True, False):
                t_sel = [(a, b, k) for t in tp for a, b, k, c in t if c == cat and ((b - a) / FPS < 4) == short]
                found = sum(1 for t, p_ in zip(tp, pp) for a, b, k, c in t if c == cat and ((b - a) / FPS < 4) == short and best(a, b, k, p_) >= 0.5)
                made = sum(1 for t, p_ in zip(tp, pp) for a, b, k, c in p_ if c == cat and ((b - a) / FPS < 4) == short and best(a, b, k, t) < 0.3)
                gain = sc([paint_cat(d, t, cat, short) for d, t in zip(dec, tp)]) - base
                lines.append("   %-14s %s: true %5.1f / half, found %.2f, made-up %4.1f / half, paint missed %+.4f" % (cat, "< 4 s" if short else ">= 4 s", len(t_sel) / len(dec), found / max(1, len(t_sel)), made / len(dec), gain))
            lines.append("   %-14s drop made-up (all lengths) %+.4f" % (cat, sc([drop_cat(d, t, cat) for d, t in zip(dec, tp)]) - base))
        for where in ("restart", "stop"):
            res = [edge_team(d, t, where) for d, t in zip(dec, tp)]
            n_all = sum(r[1] for r in res)
            n_wrong = sum(r[2] for r in res)
            n_dead = sum(r[3] for r in res)
            lines.append("   wrong team at the %-7s: %d of %d (%.3f), predicted dead there %d (%.3f); paint those %+.4f" % (where, n_wrong, n_all, n_wrong / max(1, n_all), n_dead, n_dead / max(1, n_all), sc([r[0] for r in res]) - base))
        print("\n".join(lines), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
