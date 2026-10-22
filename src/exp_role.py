"""Round 7c (user idea): role awareness without shirt numbers (src/roles.py).

1. Validation of the inferred roles against the metadata roles on the degraded bench blocks (sim V3): DFL
   PlayingPosition and SkillCorner player_role per player-half (majority of the inferred group over the frames he is
   seen), 5 groups (GK / CB / FB / MID / FWD) and 3 outfield lines (DEF / MID / FWD), mean depth rank per true group.
2. Role features of the players nearest to the ball appended to the v7 features, team model retrained as v7 (DFL
   native labels, sim V3; DFL leave-one-match-out, all-DFL model elsewhere), v7 decoder; paired against v7 on DFL-trk,
   Metrica-ctrl and SkillCorner (phase / player-possession labels; learned and true D) plus frame team accuracy.
AGGREGATE output only.

  python -I src/exp_role.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import roles  # noqa: E402

OUT = os.path.join(bench.CACHE, "role")
TAG = "role_" + exp_oracle.TAG
SETS = (("dfl@v3", "trk", (32.0, 5, 25)), ("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)),
        ("sc@v3", "pp", (30.0, 8, 50)))
EPS = 1e-4


def _job(job):
    src, i = job
    warnings.filterwarnings("ignore")
    b = bench.blocks(src)[i]
    R = roles.infer(b["P"], b["team"], b["gk"], b["pitch"], b["B"])
    fh, fa = roles.role_feats(b["P"], b["B"], b["team"], b["gk"], b["pitch"], R)
    return i, fh, fa, R["group"]


def role_blocks(src):
    """[(home view, away view, inferred group per frame and slot)] per block, cached."""
    path = os.path.join(OUT, "feat_%s.pkl" % src.replace("@", "-"))
    if os.path.isfile(path):
        return bench._load(path)
    from multiprocessing import Pool
    with Pool(bench.PROCS) as pool:
        res = sorted(pool.map(_job, [(src, i) for i in range(len(bench.blocks(src)))]), key=lambda r: r[0])
    out = [(r[1], r[2], r[3]) for r in res]
    bench._dump(out, path)
    return out


def validate(src, rb):
    """Inferred vs metadata groups per player-half; aggregate lines."""
    bl = bench.blocks(src)
    true_g, inf_g = [], []
    for b, (_, _, grp) in zip(bl, rb):
        if src.startswith("dfl"):
            meta = roles.dfl_slot_roles(b["group"])[int(b["key"][-1])]
        else:
            meta = roles.sc_slot_roles(b["group"])
        for j, g in enumerate(meta):
            if g is None or j >= grp.shape[1]:
                continue
            gj = grp[:, j]
            gj = gj[gj >= 0]
            if len(gj) < 3000:
                continue
            true_g.append(roles.GROUPS.index(g))
            inf_g.append(np.bincount(gj, minlength=len(roles.GROUPS)).argmax())
    t, p = np.array(true_g), np.array(inf_g)
    line = lambda g: np.where(g == 0, 0, np.where(g <= 2, 1, np.where(g == 3, 2, 3)))   # GK / DEF / MID / FWD
    conf = np.zeros((5, 5), int)
    np.add.at(conf, (t, p), 1)
    print("%-10s roles: %d player-halves | 5-group accuracy %.3f | line (GK/DEF/MID/FWD) accuracy %.3f" % (
        src, len(t), np.mean(t == p), np.mean(line(t) == line(p))), flush=True)
    print("           true -> inferred counts (rows %s): %s" % ("/".join(roles.GROUPS),
                                                                 " | ".join(" ".join("%3d" % v for v in r) for r in conf)),
          flush=True)


def feat_probs(src, rb):
    path = os.path.join(OUT, "ph_%s_%s.pkl" % (TAG, src.replace("@", "-")))
    if os.path.isfile(path):
        return bench._load(path)
    mpath = os.path.join(OUT, TAG + ".txt")
    bl = bench.blocks(src)

    def aug(i):
        f = bench._load(e2._feat_path(bl[i]))
        return dict(XF=f["XF"], XH=np.hstack([f["XH"], rb[i][0]]), XA=np.hstack([f["XA"], rb[i][1]]))

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
    rbs = {src: role_blocks(src) for src in ("dfl@v3", "metrica@v3", "sc@v3")}
    for src in ("dfl@v3", "sc@v3"):
        validate(src, rbs[src])
    feat_probs("dfl@v3", rbs["dfl@v3"])
    cache = {}
    for src, conv, dec in SETS:
        if src not in cache:
            phl, pdl = exp_oracle.probs(src)
            cache[src] = (phl, pdl, feat_probs(src, rbs[src]))
        phl, pdl, phr = cache[src]
        labs = bench.labels(src, conv)
        variants = [("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])]
        cells = []
        for dname, pds in variants:
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            v = e2.score(src, conv, e2.decode_all(dec, pds, phr))
            cells.append("%s v7 %.4f -> %.4f (%+.4f, better in %d/%d)" % (dname, ref.mean(), v.mean(), v.mean() - ref.mean(),
                                                                         int((v > ref).sum()), len(v)))
        print("%-10s %-6s frame team acc v7 %.4f -> roles %.4f | %s" % (src, conv, e2.team_acc(phl, labs),
                                                                       e2.team_acc(phr, labs), " | ".join(cells)),
              flush=True)


if __name__ == "__main__":
    main()
