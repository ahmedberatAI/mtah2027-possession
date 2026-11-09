"""Label-free health check of v22 on the target, per half: the decision's own expected spellF1 under the samples, possessions
in the decision and in the samples (all, < 4 s), dead share; the same numbers for the open sets' halves for comparison.
No label of the match is read; the public window is not treated apart; aggregate output only."""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import numpy as np  # noqa: E402


def stats(members, ref, lab_, seed, m=3000):
    """Expected F1 of the labelling lab_ (25 fps) under the members' samples (5 Hz), and counts."""
    import stack_decode as sd
    n = min(min(len(e) for e, _ in members), len(ref) // sd.STEP)
    segs, offs, base = [], [0], 0
    per = max(1, m // len(members))
    for j, (e, A) in enumerate(members):
        seg, off = sd.ffbs(np.asarray(e[:n], np.float64), np.asarray(A, np.float64), per, seed + j)
        segs.append(seg)
        offs += list(off[1:] + base)
        base += len(seg)
    seg = np.vstack(segs)
    off = np.array(offs, np.int64)
    M = off.shape[0] - 1
    x = np.asarray(lab_[:n * sd.STEP])[sd.STEP // 2::sd.STEP][:n]
    st = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    en = np.r_[st[1:], n]
    tp = np.zeros(len(sd.IOUS))
    P = short = 0
    for a, b in zip(st, en):
        k = int(x[a])
        if k > 1:
            continue
        P += 1
        short += (b - a) < 20
        tp += sd.match_counts(int(a), int(b), k, seg, off, sd.IOUS) / M
    live = seg[seg[:, 2] < 2]
    T = len(live) / M
    Ts = np.sum((live[:, 1] - live[:, 0]) < 20) / M
    f1 = float(np.mean(2 * tp / (P + T)))
    return f1, P, short, T, Ts, float(np.mean(x == 2))


def target():
    import atd
    import model_v20
    import model_v21
    import model_v22 as m22
    import model_v7
    import stack_decode as sd
    import stack_model as smod
    from make_submission import merged_features
    f = merged_features()
    p = dict(m22.DEFAULT)
    a_all = model_v20.alive_evidence(f)
    mem = m22.ensure_members(f)
    sp = model_v7.spec(p)
    rows = []
    for h, a, e in atd.HALVES:
        ms = mem[a]
        n = min(min(len(x[0]), len(x[1]), len(x[2])) for x in ms)
        n = min(n, e + 1 - a)
        lq = np.mean([m22.W_Q * smod.logit(x[1][:n]) + (1 - m22.W_Q) * smod.logit(x[0][:n]) for x in ms], axis=0)
        q = np.clip(1 / (1 + np.exp(-lq)), 1e-4, 1 - 1e-4)
        al = (1 - m22.W_A) * a_all[a:a + n] + m22.W_A * np.mean([x[2][:n] for x in ms], axis=0)
        S = np.stack([al / 2 + np.log(q), al / 2 + np.log(1 - q), -al / 2], 1)
        if a == atd.HALVES[1][1]:
            S[:atd.KICKOFF_2ND - a, 2] += 1e6
        ref = sp.decode(S).astype(np.int8)
        members = [(x[3], x[4]) for x in ms]
        out = sd.decide(members, ref, p["lam"], int(p["samples"]), int(p["seed"]) + 1000 * h)
        # an independent draw of samples scores the decision and the reference (v21's recipe on the seed mean)
        rows.append(("half %d decision" % h,) + stats(members, ref, out, 777000 + h))
        rows.append(("half %d v21 recipe" % h,) + stats(members, ref, ref, 777000 + h))
    return rows


def open_sets():
    import lab
    from r88 import emit
    from r93 import v22_decisions
    lab.setup()
    rows = []
    for K in lab.SETS:
        E = {s: emit(K, s) for s in (0, 1, 2)}
        v = v22_decisions(K)
        res = [stats([E[s][i] for s in (0, 1, 2)], v["ref"][i], v["dec"][i], 777000 + i) for i in range(len(v["dec"]))]
        R = np.array([r for r in res])
        rows.append((K + " (mean of %d halves; p10, p90 of expected F1)" % len(R),) + tuple(R.mean(0)) + (float(np.percentile(R[:, 0], 10)), float(np.percentile(R[:, 0], 90))))
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    which = sys.argv[1]
    for r in (target() if which == "target" else open_sets()):
        extra = " | expected F1 p10 %.3f p90 %.3f" % (r[7], r[8]) if len(r) > 7 else ""
        print("%-58s expected spellF1 %.4f | possessions %d (< 4 s %d) | samples: %.1f (< 4 s %.1f) | dead share %.3f%s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], extra), flush=True)
