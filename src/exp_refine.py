"""Round 15, capability 1: re-localise decoded possession changes (existence and localisation decoupled).

Round-15 diagnostics (src/exp_short_timing.py): short spells that v10 finds are systematically too wide -- after a
restart they end 16-31 frames late (median), in play they start 10-36 frames early and end 9-26 late -- while > 4 s
spells are on time; 18-42 % of short spells by kind are found but below IoU 0.5. Cause: the +-25 logit smoothing spreads
a short signal, and B2's exit cost falls with the spell's length, which rewards stretching a short spell. Here the
shipped decoder still decides which spells exist; every in-play change it outputs (team to team, no D on either side)
is then moved, inside a +-W window that keeps both neighbours >= the decoder's minimum length, to the frame that
maximises the two-segment log-likelihood under a less-smoothed team probability q (no duration term):
    s* = argmax_s  sum_{u<s} log q_X(u) + sum_{u>=s} log q_Y(u)      (cumulative-sum change point)
Boundaries next to D are kept. Variants: q = raw / +-6 / +-12 v10 probability; scope = every change / changes next to a
spell < 8 s. W = 50 frames. Judged as the project rule says: three v10 seeds, ten external contexts (DFL-trk, DFL-evctrl,
Metrica-ctrl, SC phase, SC pp; learned and true D), per-match scores (Metrica's noise draws averaged within a game
first), paired match bootstrap (10,000), worst-case regret; < 0.005 keeps v10. AGGREGATE output only.

  python -I src/exp_refine.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_short as es  # noqa: E402
import exp_smooth  # noqa: E402

W = 50
SHORT_SCOPE = 200
CLIP = 0.02


def refine(pred, q, w=W, min_len=5, scope="all"):
    """Move each team-to-team change of `pred` to the cumulative-sum optimum of log q_X - log q_Y within +-w."""
    out = np.asarray(pred).copy()
    if len(out) < 2:
        return out
    p = np.clip(np.asarray(q, np.float64), CLIP, 1 - CLIP)
    gain_h = np.log(p) - np.log(1 - p)                 # log q_H - log q_A
    cut = np.flatnonzero(np.diff(out)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(out)]])
    labs = out[starts]
    for i in range(len(starts) - 1):
        x, y = labs[i], labs[i + 1]
        if x > 1 or y > 1 or x == y:
            continue
        if scope == "short" and min(ends[i] - starts[i], ends[i + 1] - starts[i + 1]) >= SHORT_SCOPE:
            continue
        t = starts[i + 1]
        lo = max(starts[i] + min_len, t - w)
        hi = min(ends[i + 1] - min_len, t + w)
        if hi <= lo:
            continue
        d = gain_h[lo:hi] if x == 0 else -gain_h[lo:hi]
        s = lo + int(np.argmax(np.concatenate([[0.0], np.cumsum(d)])))
        if s > t:
            out[t:s] = x
        elif s < t:
            out[s:t] = y
        ends[i] = starts[i + 1] = s
    return out


def by_match(blocks, labels, preds):
    groups = bench.spell_by_group(blocks, labels, preds)
    per = {}
    for (match, _), sc in groups.items():
        per.setdefault(str(match), []).append(float(sc[0]))
    return {m: float(np.mean(v)) for m, v in per.items()}


def boot(delta, n=10000, seed=20261029):
    rng = np.random.default_rng(seed)
    s = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(1)
    return np.quantile(s, [0.025, 0.975])


def evaluate(variants, seeds=es.SEEDS, label=""):
    """variants: name -> fn(pred, raw, smoothed, src, dec, block_index) -> new pred. Returns per-context summaries."""
    cells = {}
    for src, convs, dec in es.CONTEXTS:
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        deads = [("learned D", learned, convs)] + [
            ("true D", [np.where(g == 2, 1 - es.EPS, es.EPS) for g in labs[c]], (c,)) for c in convs]
        for seed in seeds:
            raw = es.v10_raw(seed, src)
            smo = exp_smooth.smooth(raw, 25)
            for dname, pds, cs in deads:
                base = es.decode(dec, 48.0, pds, smo)
                outs = {"v10": base}
                for name, fn in variants.items():
                    outs[name] = [fn(p, raw[i], smo[i], src, dec, i) for i, p in enumerate(base)]
                for c in cs:
                    for name, preds in outs.items():
                        cells.setdefault((src, c, dname, name), []).append(by_match(blocks, labs[c], preds))
        print("  %s%s done" % (label, src), flush=True)
        bench._blocks.clear()
        bench._labels.clear()
    return summarize(cells, ["v10"] + list(variants))


def summarize(cells, names, ref="v10"):
    """cells: (src, conv, dead, name) -> list of per-seed {match: spellF1}; prints deltas vs `ref` and regrets."""
    ctxs = sorted({k[:3] for k in cells}, key=lambda k: (k[0], k[1], k[2]))
    mean = {}
    for ctx in ctxs:
        for name in names:
            runs = cells[ctx + (name,)]
            mean[ctx + (name,)] = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
    print("\n%-26s" % "context" + "".join("%30s" % n for n in names))
    regret = {n: 0.0 for n in names}
    for ctx in ctxs:
        base = mean[ctx + (ref,)]
        ms = sorted(base)
        vals = {n: float(np.mean([mean[ctx + (n,)][m] for m in ms])) for n in names}
        best = max(vals.values())
        line = "%-26s" % ("%s %s %s" % (ctx[0].split("@")[0], ctx[1], ctx[2]))
        for n in names:
            regret[n] = max(regret[n], best - vals[n])
            if n == ref:
                line += "%30s" % ("%.4f" % vals[n])
            else:
                dl = np.array([mean[ctx + (n,)][m] - base[m] for m in ms])
                lo, hi = boot(dl)
                line += "%30s" % ("%+.4f [%+.3f,%+.3f] %d/%d" % (dl.mean(), lo, hi, int((dl > 1e-12).sum()), len(dl)))
        print(line)
    print("worst-case regret: " + " | ".join("%s %.4f" % (n, regret[n]) for n in names), flush=True)
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    def make(kernel, scope):
        def fn(pred, raw, smo, src, dec, i):
            q = raw if kernel == 0 else exp_smooth.smooth([raw], kernel)[0]
            return refine(pred, q, min_len=int(dec[1]), scope=scope)
        return fn
    variants = {"%s/%s" % ("raw" if k == 0 else "+-%d" % k, sc): make(k, sc)
                for k in (0, 6, 12) for sc in ("all", "short")}
    evaluate(variants)


if __name__ == "__main__":
    main()
