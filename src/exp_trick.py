"""Round 35 (docs/RAPOR.md): same-day screen for short-possession tricks. A trick adds features to the verifier's candidate
rows of v15 (exp_kpu C1abETmS); the leave-one-match-out verifier is rerun with and without them on one v10 seed and the
difference is reported: insert AUC per source (the rows' own y, and the strict label IoU >= 0.5 with a true spell of the
team) and spell F1 per context with v15's thresholds. Screen: insert AUC +0.01 on DFL and Metrica, or spell F1 >= +0.003 on
both. AGGREGATE output only.

  python -I src/exp_trick.py screen NAME[,NAME...] [SEED]
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import bench  # noqa: E402
import exp_kpu  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_verify as ev  # noqa: E402
import metrica_sample as ms  # noqa: E402
import short_specialist as ss  # noqa: E402

BASE = "C1abETmS"
SCORE = {"dfl@v3": ("evctrl", "trk"), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp")}
TRICKS = {}


def trick(fn):
    """Register fn(ctx, rows) -> feature names; it sets those keys on the rows of one half (both dead-ball channels)."""
    TRICKS[fn.__name__] = fn
    return fn


class Ctx:
    """One half of a source: the block and, computed on demand, the frame cues and every member's raw evidence."""
    _members = {}

    def __init__(self, src, i, block):
        self.src, self.i, self.b = src, i, block
        self._cues = None

    @property
    def cues(self):
        if self._cues is None:
            self._cues = ea.frame_cues(self.b)
        return self._cues

    @classmethod
    def members(cls, src):
        """{"v10": [3 seeds], "gru": [3 members], "c1": [3 seeds]}: per half P(home) of every member, held out as in v15."""
        if src not in cls._members:
            gru = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
            outs = exp_kpu.outputs("C1", src)
            cls._members = {src: dict(v10=[es.v10_raw(k, src) for k in es.SEEDS], gru=[gru[k] for k in es.SEEDS],
                                      c1=[[ss.pk(np.asarray(o, np.float32), 0) for o in outs[ss.SEEDS[k]]] for k in es.SEEDS])}
        return cls._members[src]


def _toward(p, k):
    """Logit of P(home) seen from team k."""
    lg = ea.logit(p)
    return lg if k == 0 else -lg


@trick
def spread(ctx, rows):
    """T4 seed: how much the members of each model family agree on team k inside the candidate: mean and spread of their
    mean margin, and the weakest member."""
    mem = Ctx.members(ctx.src)
    names = []
    for fam, runs in mem.items():
        names += ["tr_%s_std" % fam, "tr_%s_min" % fam]
        lg = [ea.logit(np.asarray(run[ctx.i], np.float64)) for run in runs]
        for r in rows:
            sign = 1.0 if r["k"] == 0 else -1.0
            vals = [sign * float(np.mean(x[r["s"]:r["t"]])) for x in lg]
            r["tr_%s_std" % fam], r["tr_%s_min" % fam] = float(np.std(vals)), float(np.min(vals))
    return names


def _strict(lab, r):
    sp = [(a, b) for a, b, k in ms.spells(lab) if k == r["k"] and b > r["s"] and a < r["t"]]
    return any((min(b, r["t"]) - max(a, r["s"])) / (max(b, r["t"]) - min(a, r["s"])) >= 0.5 for a, b in sp)


def _f1(blocks, labs, base, rows, p):
    by = {}
    for n, r in enumerate(rows):
        by.setdefault(r["i"], []).append(n)
    preds = [ev.apply(base[i], [rows[n] for n in by.get(i, [])], p[by.get(i, [])], True, True, 0.65, 0.35)
             for i in range(len(blocks))]
    return float(np.mean(list(er.by_match(blocks, labs, preds).values())))


def screen(names, seed=0, vseed=0):
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    exp_kpu.use(BASE)
    data = {(src, d): ev._rows4(src, seed, d) for src in exp_kpu.SRCS for d in ev.DEADS}
    keys = exp_kpu.feats(BASE)
    new = []
    for src in exp_kpu.SRCS:
        blocks = bench.blocks(src)
        for i, b in enumerate(blocks):
            rows = [r for d in ev.DEADS for r in data[(src, d)]["rows"] if r["i"] == i]
            ctx = Ctx(src, i, b)
            for name in names:
                got = TRICKS[name](ctx, rows)
                new += [k for k in got if k not in new]
        bench._blocks.clear()
        print("  features computed: %s" % src, flush=True)
    rows = [r for d in ev.DEADS for src in exp_kpu.SRCS for r in data[(src, d)]["rows"]]
    for r in rows:
        for k in new:
            r.setdefault(k, np.nan)
    p0 = ev.predict_lomo(rows, keys, params)
    p1 = ev.predict_lomo(rows, keys + new, params)
    n0 = 0
    print("\n%s, %d new features, v10 seed %d, verifier seed %d" % ("+".join(names), len(new), seed, vseed))
    for d in ev.DEADS:
        for src in exp_kpu.SRCS:
            part = data[(src, d)]["rows"]
            idx = np.arange(n0, n0 + len(part))
            n0 += len(part)
            blocks = bench.blocks(src)
            convs = SCORE[src]
            labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
            y = np.array([r["y"] for r in part])
            ins = np.array([r["typ"] == "ins" for r in part])
            q0, q1 = p0[idx], p1[idx]
            strict = np.array([_strict(labs[convs[0]][r["i"]], r) if r["typ"] == "ins" else False for r in part])
            txt = "%-10s %-9s ins AUC %.3f -> %.3f, strict %.3f -> %.3f" % (
                src.split("@")[0], d, ea.auc(q0[ins & (y == 1)], q0[ins & (y == 0)]), ea.auc(q1[ins & (y == 1)], q1[ins & (y == 0)]),
                ea.auc(q0[ins & strict], q0[ins & ~strict]), ea.auc(q1[ins & strict], q1[ins & ~strict]))
            for c in convs:
                a, b = _f1(blocks, labs[c], data[(src, d)]["base"], part, q0), _f1(blocks, labs[c], data[(src, d)]["base"], part, q1)
                txt += " | %s %.4f %+.4f" % (c, a, b - a)
            print(txt, flush=True)
            bench._blocks.clear()
            bench._labels.clear()
    exp_kpu.use("v14")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if len(sys.argv) < 3 or sys.argv[1] != "screen":
        raise SystemExit(__doc__)
    screen(sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 0)


if __name__ == "__main__":
    main()
