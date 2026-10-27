"""Round 9 (research a20): ball-free short-horizon team interaction features (src/bfree.py) appended to the v7
features, team model retrained as v7 (DFL native labels, sim V3; DFL leave-one-match-out, all-DFL model elsewhere),
v7 decoder; paired against v7 on DFL-trk, Metrica-ctrl and SkillCorner (phase / player-possession labels; learned and
true D) plus frame team accuracy. AGGREGATE output only.

  python -I src/exp_bfree.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import bfree  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402

OUT = os.path.join(bench.CACHE, "bfree")
TAG = "bfree_" + exp_oracle.TAG
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("sc@v3", "pp", (30.0, 8, 50)))
EPS = 1e-4


def _job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    return (i,) + bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])


def extra(src):
    path = os.path.join(OUT, "feat_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_job, [(src, i) for i in range(len(bench.blocks(src)))]), key=lambda r: r[0])
    out = [(r[1], r[2]) for r in res]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(out, path)
    return out


def probs(src):
    path = os.path.join(OUT, "ph_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    mpath = os.path.join(OUT, TAG + ".txt")
    bl = bench.blocks(src)
    ex = extra(src)

    def aug(i):
        f = bench._load(e2._feat_path(bl[i]))
        return dict(XF=f["XF"], XH=np.hstack([f["XH"], ex[i][0]]), XA=np.hstack([f["XA"], ex[i][1]]))

    if src == "dfl@v3":
        fd = [aug(i) for i in range(len(bl))]
        labs = bench.labels("dfl", "native")
        out = [None] * len(bl)
        for m in sorted({b["group"] for b in bl}):
            tr = [i for i, b in enumerate(bl) if b["group"] != m]
            mdl = bench.fit_team([fd[i] for i in tr], [labs[i] for i in tr], {"num_threads": bench.THREADS})
            for i, b in enumerate(bl):
                if b["group"] == m:
                    out[i] = bench.predict_team(mdl, fd[i])
            print("  LOMO %s done" % m, flush=True)
        bench.fit_team(fd, labs, {"num_threads": bench.THREADS}).save_model(mpath)
    else:
        mdl = lgb.Booster(model_file=mpath)
        out = [bench.predict_team(mdl, aug(i)) for i in range(len(bl))]
    bench._dump(out, path)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    probs("dfl@v3")
    cache = {}
    for src, conv, dec in SETS:
        if src not in cache:
            phl, pdl = exp_oracle.probs(src)
            cache[src] = (phl, pdl, probs(src))
        phl, pdl, phn = cache[src]
        labs = bench.labels(src, conv)
        cells = []
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            v = e2.score(src, conv, e2.decode_all(dec, pds, phn))
            cells.append("%s v7 %.4f -> %.4f (%+.4f, better in %d/%d)" % (dname, ref.mean(), v.mean(),
                                                                         v.mean() - ref.mean(), int((v > ref).sum()),
                                                                         len(v)))
        print("%-10s %-6s frame team acc v7 %.4f -> ball-free %.4f | %s" % (
            src, conv, e2.team_acc(phl, labs), e2.team_acc(phn, labs), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
