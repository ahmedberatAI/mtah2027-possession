"""Round 94, information after W3 failed its provider hold-out: B6's form under the hold-out (the four-member mixture of
Hy and Hyw hold-out stackers, seeds 0 and 1, M 2000) against v22's form (Hy, seeds 0 and 1), same reference.
Open data only, aggregate output."""
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from lab import np  # noqa: E402

OUTD = os.path.join(lab.REPO, "cache", "lab87", "foldnets")
CRF = os.path.join(lab.REPO, "cache", "bench", "crf")


def main():
    lab.setup()
    import stack_decode as sd
    t0 = time.time()
    seeds = (0, 1)
    cells = {}
    for K in ("dfl", "metrica", "sc"):
        P = lab.get(K)
        with open(os.path.join(CRF, "evidence_lopo_%s.pkl" % K), "rb") as fh:
            ev = pickle.load(fh)["ev"]
        mixes, E = {}, {}
        for s in seeds:
            with open(os.path.join(CRF, "slopo_Hy_%d_%s.pkl" % (s, K)), "rb") as fh:
                h = pickle.load(fh)
            for t in ("Hy", "Hyw"):
                with open(os.path.join(OUTD, "emitlopo_%s_%d_%s.pkl" % (t, s, K)), "rb") as fh:
                    E[(t, s)] = pickle.load(fh)
            A, Q = [], []
            for e, b, q, x in zip(ev, P["s50"][s], h["q"], h["alive"]):
                n = min(len(e), len(b), len(q), len(x))
                A.append(0.5 * np.asarray(e[:n], np.float64) + 0.5 * np.asarray(x[:n], np.float64))
                Q.append(lab.sg(0.5 * lab.lg(q[:n]) + 0.5 * lab.lg(b[:n])))
            mixes[s] = (A, Q)
        bagA, bagQ = lab.mean_paths([mixes[s] for s in seeds])
        ref = [np.asarray(x) for x in lab.decode(bagA, bagQ)]
        with ThreadPoolExecutor(8) as pool:
            dy = list(pool.map(lambda i: sd.decide([E[("Hy", s)][i] for s in seeds], ref[i], 0.5, 2000, 890000 + i), range(len(ref))))
            db = list(pool.map(lambda i: sd.decide([E[("Hy", s)][i] for s in seeds] + [E[("Hyw", s)][i] for s in seeds], ref[i], 0.5, 2000, 890000 + i), range(len(ref))))
        for name, d in (("Hy", dy), ("B6", db)):
            for c, v in lab.score(K, d).items():
                cells[(K, c, name)] = float(np.mean(list(v.values())))
        print("%s done %.0f s" % (K, time.time() - t0), flush=True)
    keys = [k for k in lab.CTX if k[0] != "st"]
    print("B6's form against v22's, provider held out: %s" % " | ".join("%s %s %+.4f" % (k[0], k[1], cells[(k[0], k[1], "B6")] - cells[(k[0], k[1], "Hy")]) for k in keys), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
