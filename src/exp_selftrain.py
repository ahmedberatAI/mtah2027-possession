"""Round 9: self-training on the test match (transductive domain adaptation, no labels). v10 labels the held-out match;
its confident frames -- decoded H / A (v7 decoder, learned dead-ball probabilities) AND smoothed team probability
beyond 0.5 +- (CONF - 0.5) for that team -- become pseudo-labels; v10 is retrained with them added (weight ALPHA = one
match) and predicts the same match again. On BAR-ATH this would mean pseudo-labelling BAR-ATH itself with v10.
One target match at a time, as on BAR-ATH: DFL by match, Metrica by game (its 3 seeds), SkillCorner by match (the
first-stage SkillCorner probabilities come from the all-data model). Paired spell F1 against v10 and v7, learned and
true D. AGGREGATE output only.

  python -I src/exp_selftrain.py           (env ST_ALPHA, default 1.0; ST_CONF, default 0.9)
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_a2  # noqa: E402
import exp_combo  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

OUT = os.path.join(bench.CACHE, "selftrain")
EPS = 1e-4
W_M = 2.0
ALPHA = float(os.environ.get("ST_ALPHA", "1.0"))
CONF = float(os.environ.get("ST_CONF", "0.9"))
DEC = {"dfl@v3": (32.0, 5, 25), "metrica@v3": (30.0, 8, 50), "sc@v3": (30.0, 8, 50)}


def pseudo(src, raw, pdl):
    """Pseudo-label arrays (H / A where confident, X elsewhere) for every block of src."""
    phs = exp_smooth.smooth(raw, 25)
    dec = e2.decode_all(DEC[src], pdl, phs)
    out, kept = [], 0
    for y, p in zip(dec, phs):
        lab = np.full(len(y), bench.X, np.int8)
        lab[(y == bench.H) & (p > CONF)] = bench.H
        lab[(y == bench.A) & (p < 1 - CONF)] = bench.A
        kept += int((lab < 2).sum())
        out.append(lab)
    return out, kept / sum(len(y) for y in dec)


def probs():
    path = os.path.join(OUT, "probs_a%g_c%g.pkl" % (ALPHA, CONF))
    if os.path.isfile(path):
        return bench._load(path)
    raw = bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree_wm2.pkl"))
    feats = {src: exp_combo.aug_feats(src) for src in DEC}
    ld, lm = bench.labels("dfl", "native"), bench.labels("metrica@v3", "ctrl")
    blocks = {src: bench.blocks(src) for src in DEC}
    ps = {}
    for src in DEC:
        ps[src], share = pseudo(src, raw[src], exp_oracle.probs(src)[1])
        print("  %s: pseudo-labelled share of frames %.3f" % (src, share), flush=True)
    fd, fm = feats["dfl@v3"], feats["metrica@v3"]
    d_all, m_all = exp_a2.rows(fd, ld, 1.0), exp_a2.rows(fm, lm, W_M / 3.0)
    res = {src: [None] * len(blocks[src]) for src in DEC}
    for src in DEC:
        bl = blocks[src]
        for grp in sorted({b["group"] for b in bl}):
            te = [i for i, b in enumerate(bl) if b["group"] == grp]
            n_copies = len(te) / 2.0                      # Metrica: 3 seeds x 2 halves -> weight split over the copies
            parts = [exp_a2.rows([feats[src][i] for i in te], [ps[src][i] for i in te], ALPHA / n_copies)]
            if src == "dfl@v3":
                tr = [i for i, b in enumerate(bl) if b["group"] != grp]
                parts += [exp_a2.rows([fd[i] for i in tr], [ld[i] for i in tr], 1.0), m_all]
            elif src == "metrica@v3":
                tr = [i for i, b in enumerate(bl) if b["group"] != grp]
                parts += [d_all, exp_a2.rows([fm[i] for i in tr], [lm[i] for i in tr], W_M / 3.0)]
            else:
                parts += [d_all, m_all]
            mdl = exp_a2.fit(parts)
            for i in te:
                res[src][i] = bench.predict_team(mdl, feats[src][i])
        print("  %s done" % src, flush=True)
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    print("self-training alpha %g conf %g" % (ALPHA, CONF), flush=True)
    new = {k: exp_smooth.smooth(v, 25) for k, v in probs().items()}
    v10 = bench._load(os.path.join(bench.CACHE, "combo", "probs_v10_w25.pkl"))
    for src, conv, dec in exp_a2.SETS:
        phl, pdl = exp_oracle.probs(src)
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            r7 = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            r10 = e2.score(src, conv, e2.decode_all(dec, pds, v10[src]))
            v = e2.score(src, conv, e2.decode_all(dec, pds, new[src]))
            cells.append("%s v10 %.4f -> %.4f (vs v10 %+.4f, %d/%d; vs v7 %+.4f)" % (
                dname, r10.mean(), v.mean(), v.mean() - r10.mean(), int((v > r10).sum()), len(v), v.mean() - r7.mean()))
        print("%-10s %-6s acc %.4f (v10 %.4f) | %s" % (src, conv, e2.team_acc(new[src], labs),
                                                      e2.team_acc(v10[src], labs), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
