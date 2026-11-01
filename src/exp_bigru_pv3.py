"""Round 21 (local GPU): the shared BiGRU recipe with PFF degraded like the rest of the data it meets.

The shared BiGRU learns DFL through sim.V3 (BAR-ATH-like ATD: smoothed position noise, fragmented player tracks, ball
gaps calibrated on BAR-ATH aggregates) but PFF through its raw broadcast coordinates ("no extra synthetic degradation",
reports/sequence_pff/design.json), so half of every minibatch differs from what the model meets on DFL@v3, Metrica@v3,
SkillCorner@v3 and BAR-ATH. Here every PFF half is degraded with sim.V3 too (seed 30000 + 10 x match index + period;
V3's ball model tops PFF's own missing ball up to its targets), the v7 features are recomputed and the recipe is
retrained unchanged: seven held-out DFL folds + one all-data model per seed (src/exp_bigru_m.py's plain mode on this
index). PFF labels as in the shared recipe (controlled, live only where the native labels agree). AGGREGATE output only.

  python -I src/exp_bigru_pv3.py prepare [WORKERS]   degraded PFF features, views and the sequence index (CPU)
  python -I src/exp_bigru_pv3.py train SEED          seven held-out DFL folds + the all-data model (GPU)
  python -I src/exp_bigru_pv3.py collect             held-out DFL probabilities, Metrica / SkillCorner members
"""
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import bench  # noqa: E402
import exp_bigru_m as bm  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "cache" / "bigru_pv3"
FEAT = OUT / "features"


def _one(job):
    mi, match = job
    import gradient
    import model_v7
    import seq_pff as seq
    import sim
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t = time.time()
    rows = []
    with np.load(gradient.CACHE / gradient.VERSION / (match + ".npz")) as d:
        pitch = tuple(float(v) for v in d["pitch"])
        for per in d["periods"]:
            tag = "h%d" % per
            lab = d[tag + "_labels"].copy()
            if not ((lab == 0) | (lab == 1)).any():
                continue
            key = "pffv3_%s_%s" % (match, tag)
            xp, yp = FEAT / (key + ".npy"), FEAT / (key + "_labels.npy")
            if not xp.exists():
                Pn, Bn = sim.degrade(d[tag + "_players"], d[tag + "_ball"], lab, pitch, seed=30000 + 10 * mi + int(per),
                                     params=sim.V3)
                block = dict(P=Pn[:, :, :2].astype(np.float32), B=Bn[:, :2].astype(np.float32), team=d["team"],
                             gk=d["gk"], pitch=pitch)
                fp = FEAT / (key + ".pkl")
                bench._block_feats((model_v7.VERSION, block, str(fp)))
                np.save(xp, seq.views(bench._load(str(fp))))
                fp.unlink()
            common = (d[tag + "_native"] < 2) & (lab < 2)
            lab[(lab < 2) & ~common] = 3
            np.save(yp, lab.astype(np.int8))
            rows.append(dict(key=key, source="pff", match=match, x=str(xp), y=str(yp), full_length=len(lab),
                             split="train"))
    print("degraded %s: %d halves, %.0f s" % (match, len(rows), time.time() - t), flush=True)
    return rows


def prepare(workers):
    from concurrent.futures import ProcessPoolExecutor
    import exp_seq_pff as neural
    import gradient
    FEAT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in neural.read(neural.OUT / "index.json") if r["source"] == "dfl"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_one, list(enumerate(gradient.matches()))):
            rows += part
    (OUT / "index.json").write_text(json.dumps(rows), encoding="utf-8")
    print("index: %d rows (%d degraded PFF halves)" % (len(rows), sum(r["source"] == "pff" for r in rows)), flush=True)


def members(src):
    return bench._load(str(OUT / "members_p.pkl"))[src]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stage = sys.argv[1]
    bm.OUT = OUT                                  # exp_bigru_m's plain mode, reading and writing this folder
    if stage == "prepare":
        prepare(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif stage == "train":
        bm.train(int(sys.argv[2]), None, False)
    elif stage == "collect":
        bm.collect(None, False)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
