"""Round 94 (pre-registered in docs/RAPOR.md): the stacker's capacity with three seeds under the decision.
W3 = the sample mixture of Hyw's three seeds (width 128), B6 = Hy x 3 + Hyw x 3; M 3000, lambda 0.5, v22's reference.
Against v22 (cached decisions of round 93, same seeds) with the rule's bootstrap; provider held out with two seeds.
Open data only, aggregate output.

    python -I r94.py open
    python -I r94.py lopo
"""
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402
from r88 import emit  # noqa: E402
from r93 import v22_decisions  # noqa: E402

OUTD = os.path.join(lab.REPO, "cache", "lab87", "foldnets")
CRF = os.path.join(lab.REPO, "cache", "bench", "crf")


def open_sets():
    import stack_decode as sd
    t0 = time.time()
    res = {}
    for K in lab.SETS:
        Ey = {s: emit(K, s) for s in (0, 1, 2)}
        Ew = {s: emit(K, s, "Hyw") for s in (0, 1, 2)}
        v = v22_decisions(K)
        ref, v22 = v["ref"], v["dec"]
        mixes = {s: lab.mix(K, s, *lab.stacker("Hy", s, K)) for s in (0, 1, 2)}
        singles = {s: [np.asarray(x) for x in lab.decode(*mixes[s])] for s in (0, 1, 2)}
        with ThreadPoolExecutor(8) as pool:
            w3 = list(pool.map(lambda i: sd.decide([Ew[s][i] for s in (0, 1, 2)], ref[i], 0.5, 3000, 890000 + i), range(len(ref))))
            b6 = list(pool.map(lambda i: sd.decide([Ey[s][i] for s in (0, 1, 2)] + [Ew[s][i] for s in (0, 1, 2)], ref[i], 0.5, 3000, 890000 + i), range(len(ref))))
        res[K] = {"v21": [lab.score(K, singles[s]) for s in (0, 1, 2)], "v22": lab.score(K, v22), "W3": lab.score(K, w3), "B6": lab.score(K, b6)}
        print("%s done %.0f s" % (K, time.time() - t0), flush=True)
    base = {(K, c): float(np.mean([np.mean(list(r[c].values())) for r in res[K]["v21"]])) for K in lab.SETS for c in res[K]["v21"][0]}
    v22m = {(K, c): float(np.mean(list(res[K]["v22"][c].values()))) for K in lab.SETS for c in res[K]["v22"]}
    print("v22 levels: %s" % " | ".join("%s %s %.4f" % (k[0], k[1], v22m[k]) for k in lab.CTX if k in v22m), flush=True)
    for name in ("v22", "W3", "B6"):
        vals = {(K, c): float(np.mean(list(res[K][name][c].values()))) for K in lab.SETS for c in res[K][name]}
        print("%-4s against v21: %s" % (name, " | ".join("%s %s %+.4f" % (k[0], k[1], vals[k] - base[k]) for k in lab.CTX if k in vals)), flush=True)
        if name != "v22":
            print("%-4s against v22: %s" % (name, " | ".join("%s %s %+.4f" % (k[0], k[1], vals[k] - v22m[k]) for k in lab.CTX if k in vals)), flush=True)
    lab.xs()
    import exp_blend_weight as bw
    src = {"dfl": "dfl@v3", "metrica": "metrica@v3", "sc": "sc@v3"}
    for name in ("W3", "B6"):
        table = {}
        for K in ("dfl", "metrica", "sc"):
            for c in res[K]["v22"]:
                table[(src[K], c, "edges D", "v22")] = dict(res[K]["v22"][c])
                table[(src[K], c, "edges D", name)] = dict(res[K][name][c])
        print("-- %s against v22" % name, flush=True)
        bw.decide(table, "v22", name)
    with open(os.path.join(lab.PACKS, "r94_open.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    print("done %.0f s" % (time.time() - t0), flush=True)


def lopo():
    """Provider held out: the two-seed mixtures of Hy (v22's form) and Hyw (W3's form), same reference (v21's recipe on Hy's two hold-out seeds)."""
    import stack_decode as sd
    t0 = time.time()
    seeds = (0, 1)
    cells = {}
    for K in ("dfl", "metrica", "sc"):
        P = lab.get(K)
        with open(os.path.join(CRF, "evidence_lopo_%s.pkl" % K), "rb") as fh:
            ev = pickle.load(fh)["ev"]
        mixes, Ey, Ew = {}, {}, {}
        for s in seeds:
            with open(os.path.join(CRF, "slopo_Hy_%d_%s.pkl" % (s, K)), "rb") as fh:
                h = pickle.load(fh)
            with open(os.path.join(OUTD, "emitlopo_Hy_%d_%s.pkl" % (s, K)), "rb") as fh:
                Ey[s] = pickle.load(fh)
            with open(os.path.join(OUTD, "emitlopo_Hyw_%d_%s.pkl" % (s, K)), "rb") as fh:
                Ew[s] = pickle.load(fh)
            A, Q = [], []
            for e, b, q, x in zip(ev, P["s50"][s], h["q"], h["alive"]):
                n = min(len(e), len(b), len(q), len(x))
                A.append(0.5 * np.asarray(e[:n], np.float64) + 0.5 * np.asarray(x[:n], np.float64))
                Q.append(lab.sg(0.5 * lab.lg(q[:n]) + 0.5 * lab.lg(b[:n])))
            mixes[s] = (A, Q)
        bagA, bagQ = lab.mean_paths([mixes[s] for s in seeds])
        ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
        with ThreadPoolExecutor(8) as pool:
            dy = list(pool.map(lambda i: sd.decide([Ey[s][i] for s in seeds], ref[i], 0.5, 2000, 890000 + i), range(len(ref))))
            dw = list(pool.map(lambda i: sd.decide([Ew[s][i] for s in seeds], ref[i], 0.5, 2000, 890000 + i), range(len(ref))))
        for name, d in (("Hy", dy), ("Hyw", dw)):
            for c, v in lab.score(K, d).items():
                cells[(K, c, name)] = float(np.mean(list(v.values())))
        print("%s done %.0f s" % (K, time.time() - t0), flush=True)
    keys = [k for k in lab.CTX if k[0] != "st"]
    print("provider held out, Hy mixture (v22's form): %s" % " | ".join("%s %s %.4f" % (k[0], k[1], cells[(k[0], k[1], "Hy")]) for k in keys), flush=True)
    print("W3's form against it: %s" % " | ".join("%s %s %+.4f" % (k[0], k[1], cells[(k[0], k[1], "Hyw")] - cells[(k[0], k[1], "Hy")]) for k in keys), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    lab.setup()
    {"open": open_sets, "lopo": lopo}[sys.argv[1]]()
