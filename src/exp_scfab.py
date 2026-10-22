"""External check of idea 8 on SkillCorner (docs/FIKIRLER.md): SkillCorner marks its own extrapolated ball positions
(ball_data.is_detected = False), the analogue of the ATD frames BAR-ATH fills in by prediction.
  1. detector check: src/exp_fabball.fabricated() on the clean SkillCorner ball vs those flags
  2. effect: v7's team model on SkillCorner with extrapolated ball frames dropped (by SkillCorner's flag, and by the
     detector) vs as is; true D and learned D, v7 decoder. AGGREGATE output only.

  python -I src/exp_scfab.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb  # noqa: E402

import bench  # noqa: E402
import decode2  # noqa: E402
import exp_fabball  # noqa: E402
import exp_v7  # noqa: E402
import model_v7  # noqa: E402
import skillcorner as scr  # noqa: E402
import train_dfl as td  # noqa: E402

SRC = "sc@v3"


def _feat_job(job):
    b, version = job
    warnings.filterwarnings("ignore")
    return exp_fabball.block_features(b, version)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    V = model_v7.VERSION
    bl = bench.blocks(SRC)
    flags, det_flags = [], []
    tp = fp = fn = 0
    for m in scr.matches():
        d = scr.parse(m)
        L, W = (float(x) for x in d["pitch"])
        for h in (1, 2):
            ball = d["h%d_ball" % h][:, :2] * np.array([L, W])
            present = np.isfinite(ball[:, 0])
            extrap = present & ~d["h%d_bdet" % h]
            det = exp_fabball.fabricated(ball)
            tp += int((det & extrap).sum())
            fp += int((det & ~extrap & present).sum())
            fn += int((~det & extrap).sum())
            flags.append(extrap)
            det_flags.append(det)
    print("detector on the clean SkillCorner ball vs its own extrapolation flags: flagged %d frames, precision %.3f, "
          "recall %.3f (extrapolated share of present ball frames %.3f)" % (
              tp + fp, tp / max(1, tp + fp), tp / max(1, tp + fn),
              sum(int(f.sum()) for f in flags) / sum(int(np.isfinite(b["B"][:, 0]).sum()) for b in bl)), flush=True)
    mdl = lgb.Booster(model_file=os.path.join(bench.CACHE, "models", "%s_native_v3.txt" % bench._safe(V)))
    gts = bench.labels(SRC, "phase")
    learned = bench.dead_ext("base", "v3", SRC)
    true_d = [np.where(g == 2, 1 - 1e-4, 1e-4) for g in gts]
    sp = decode2.duration_spec(12.0, 30.0, 8, 50, exp_v7.ramp(175, 48.0))
    from multiprocessing import Pool
    res = {}
    for name, fl in (("as is", None), ("SkillCorner-extrapolated dropped", flags), ("detector-flagged dropped", det_flags)):
        if fl is None:
            feats = bench.features(V, SRC)
        else:
            jobs = []
            for b, f_ in zip(bl, fl):
                bb = dict(b)
                B = b["B"].copy()
                B[f_[:len(B)]] = np.nan
                bb["B"] = B
                jobs.append((bb, V))
            with Pool(bench.PROCS) as pool:
                feats = pool.map(_feat_job, jobs)
        phl = [bench.predict_team(mdl, f) for f in feats]
        acc = np.mean([np.mean((ph[g < 2] > 0.5) == (g[g < 2] == 0)) for ph, g in zip(phl, gts)])
        for dname, pds in (("learned D", learned), ("true D", true_d)):
            r = bench.spell_by_group(bl, gts, [sp.decode(td.emissions(a, p, 1.0, 1.0)).astype(np.int8)
                                               for a, p in zip(pds, phl)])
            res[(name, dname)] = np.array([r[k][0] for k in sorted(r)])
        base = (res.get(("as is", "learned D")), res.get(("as is", "true D")))
        cells = []
        for dname, ref in zip(("learned D", "true D"), base):
            v = res[(name, dname)]
            cells.append("%s %.4f (%+.4f, %d/20)" % (dname, v.mean(), v.mean() - ref.mean(), int((v > ref).sum())))
        print("%-33s team acc %.4f | %s" % (name, acc, " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
