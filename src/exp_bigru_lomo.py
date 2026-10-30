"""Round 17: leave-one-DFL-match-out training of the shared BiGRU recipe (GPU, Colab), so that the round-13 blend and
the round-16b blend + verifier can be judged on DFL. Open data only (DFL, PFF); no BAR-ATH data. AGGREGATE output only.

The recipe is the one frozen by the teammate's runs (reports/seed_balance/design.json, reports/sequence_pff/
selection.json): BiGRU width 64, dropout .15, lr .001, 11 epochs x 192 steps x 16 windows, exact 50/50 DFL / PFF
minibatches, v7 features with sim.V3 on DFL and raw PFF coordinates, training-only scaler drawn with seed 20261024,
seeds 20261025 / 26 / 27. For every DFL match m: train on the other six DFL matches (native labels) and all 64 PFF
matches (controlled labels), then predict m's two halves. Output: cache/bigru_lomo/dfl_lomo_probs.pkl
{seed: {"<match>/<half>": P(home) per frame}} and reports/bigru_lomo/manifest.json (timings, held-out frame accuracy,
checkpoint hashes).

  python src/exp_bigru_lomo.py prepare [workers]   DFL features (sim V3), PFF features, sequence index (CPU)
  python src/exp_bigru_lomo.py train SEED          the seven folds of one seed (GPU; seeds can run side by side)
  python src/exp_bigru_lomo.py collect             merge the per-fold predictions, write the manifest
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

import bench  # noqa: E402
import dfl  # noqa: E402
import model_v7  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "cache" / "bigru_lomo"
REPORT = ROOT / "reports" / "bigru_lomo"
SEEDS = (20261025, 20261026, 20261027)
SCALER_SEED = 20261024


def prepare(workers):
    from concurrent.futures import ProcessPoolExecutor
    import exp_pff as pff
    import exp_seq_pff as neural
    import gradient
    t = time.time()
    bench.features(model_v7.VERSION, "dfl@v3")
    print("DFL features ready, %.0f s" % (time.time() - t), flush=True)
    (pff.OUT / "features").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(pff.prepare_one, gradient.matches()))
    print("PFF features ready, %.0f s" % (time.time() - t), flush=True)
    (neural.OUT / "features").mkdir(parents=True, exist_ok=True)
    neural.prepare()
    print("sequence index ready, %.0f s" % (time.time() - t), flush=True)


def recipe():
    sel = json.loads((ROOT / "reports" / "sequence_pff" / "selection.json").read_text(encoding="utf-8"))
    return sel["selected"]["bigru"]["config"], int(sel["selected"]["bigru"]["epoch"])


def train(seed):
    import torch
    import exp_seq_pff as neural
    import seq_pff as seq
    config, epochs = recipe()
    rows = neural.read(neural.OUT / "index.json")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    neural.REPORT = REPORT
    for match in dfl.MATCHES:
        dest = OUT / ("pred_%s_s%d.pkl" % (match, seed))
        if dest.exists():
            continue
        t = time.time()
        data = neural.Data([r for r in rows if not (r["source"] == "dfl" and r["match"] == match)])
        neural.SEED = SCALER_SEED                    # the recipe's scaler is drawn with a fixed seed
        scaler = data.scaler()
        neural.SEED = seed
        ckpt = OUT / ("bigru_%s_s%d.pt" % (match, seed))
        neural.train_one(config, data, scaler, epochs, ckpt)
        net, sc, _ = neural.load_model(ckpt)
        preds, hits, live = {}, 0, 0
        for r in rows:
            if r["source"] == "dfl" and r["match"] == match:
                p = seq.predict_raw(net, sc, np.load(r["x"], mmap_mode="r"), r["full_length"]).astype(np.float32)
                y = np.load(r["y"])
                ok = y < 2
                hits += int(((p > 0.5) == (y == 0))[ok].sum())
                live += int(ok.sum())
                preds["%s/%s" % (match, r["key"].rsplit("_", 1)[-1])] = p
        del net
        torch.cuda.empty_cache()
        bench._dump(dict(preds=preds, accuracy=hits / max(live, 1), seconds=time.time() - t,
                         checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()), str(dest))
        print("seed %d held-out %s: native-label frame accuracy %.4f, %.0f s" % (
            seed, match, hits / max(live, 1), time.time() - t), flush=True)


def collect():
    probs, folds = {}, []
    for seed in SEEDS:
        probs[seed] = {}
        for match in dfl.MATCHES:
            d = bench._load(str(OUT / ("pred_%s_s%d.pkl" % (match, seed))))
            probs[seed].update(d["preds"])
            folds.append(dict(seed=seed, match=match, accuracy=d["accuracy"], seconds=d["seconds"],
                              checkpoint_sha256=d["checkpoint_sha256"]))
    bench._dump(probs, str(OUT / "dfl_lomo_probs.pkl"))
    config, epochs = recipe()
    manifest = dict(recipe=dict(config=config, epochs=epochs, seeds=SEEDS, scaler_seed=SCALER_SEED),
                    folds=folds, halves=sorted(probs[SEEDS[0]]),
                    mean_accuracy=float(np.mean([f["accuracy"] for f in folds])))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("collected %d folds, mean held-out native-label frame accuracy %.4f" % (
        len(folds), manifest["mean_accuracy"]), flush=True)


def main():
    stage = sys.argv[1]
    if stage == "prepare":
        prepare(int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1))
    elif stage == "train":
        train(int(sys.argv[2]))
    elif stage == "collect":
        collect()


if __name__ == "__main__":
    main()
