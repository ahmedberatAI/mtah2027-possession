"""Round 18 (literature review, agent A2): the shared BiGRU recipe with a spell-balanced loss (Colab GPU job).

Same recipe, data, scaler and seeds as the shared BiGRU and its leave-one-DFL-match-out copies (src/exp_bigru_lomo.py),
except the per-frame loss: every live frame of a true spell of L frames (25 fps) is weighted
min(cap, sqrt(100 / max(L, 6))), normalised to mean 1 per half (src/exp_sb.py spell_weight), and each window's
weighted loss is divided by its number of live frames, so windows around short spells weigh more. Caps 2 and 4 were
fixed in advance. Per seed and cap: the seven leave-one-DFL-match-out models (held-out DFL probabilities for the
evaluation) and one model on all DFL + PFF (for Metrica, SkillCorner and BAR-ATH, run locally on the CPU).
No BAR-ATH data on the VM; prints aggregates only.

  python src/exp_bigru_sb.py train SEED CAP      (after src/exp_bigru_lomo.py prepare, on the GPU)
  python src/exp_bigru_sb.py collect CAP         -> cache/bigru_sb/dfl_lomo_probs_sb<cap>.pkl
  python -I src/exp_bigru_sb.py members SRC CAP  local CPU inference of the all-data models
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import bench  # noqa: E402
import dfl  # noqa: E402
import exp_bigru_lomo as bl  # noqa: E402
from exp_sb import spell_weight  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "cache" / "bigru_sb"


def weighted_loss(logits, target, valid, source):
    """seq_pff.balanced_loss with per-frame weights carried in `valid` (0 = not live)."""
    import torch
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per = (losses * valid).sum(1) / (valid > 0).sum(1).clamp(min=1)
    return .5 * (per[source == 0].mean() + per[source == 1].mean())


def weighted_data(rows, cap):
    import exp_seq_pff as neural
    import seq_pff as seq
    import torch

    class Weighted(neural.Data):
        def __init__(self, rows):
            super().__init__(rows)
            self.w = [spell_weight(lab, cap)[::seq.STEP] for lab in self.full_y]

        def batch(self, specs, scaler):
            xx, yy, mm, ss = [], [], [], []
            for i, start, side, source in specs:
                xx.append(seq.window(self.x[i], start, side, scaler))
                lab = np.full(seq.CORE, 3, np.int8)
                w = np.zeros(seq.CORE, np.float32)
                a, b = max(0, start), min(len(self.y[i]), start + seq.CORE)
                if b > a:
                    lab[a - start:b - start] = self.y[i][a:b]
                    w[a - start:b - start] = self.w[i][a:b]
                yy.append((lab == side).astype(np.float32))
                mm.append(((lab < 2) * w).astype(np.float32))
                ss.append(source)
            return tuple(torch.from_numpy(a) for a in (np.stack(xx), np.stack(yy), np.stack(mm), np.asarray(ss)))
    return Weighted(rows)


def train(seed, cap):
    import torch
    import exp_seq_pff as neural
    import seq_pff as seq
    seq.balanced_loss = weighted_loss
    config, epochs = bl.recipe()
    rows = neural.read(neural.OUT / "index.json")
    OUT.mkdir(parents=True, exist_ok=True)
    neural.REPORT = OUT
    for match in list(dfl.MATCHES) + ["all"]:
        dest = OUT / ("pred_%s_s%d_sb%g.pkl" % (match, seed, cap))
        if dest.exists():
            continue
        t = time.time()
        data = weighted_data([r for r in rows if not (r["source"] == "dfl" and r["match"] == match)], cap)
        neural.SEED = bl.SCALER_SEED
        scaler = data.scaler()
        neural.SEED = seed
        ckpt = OUT / ("bigru_%s_s%d_sb%g.pt" % (match, seed, cap))
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
        print("cap %g seed %d held-out %s: native-label frame accuracy %.4f, %.0f s" % (
            cap, seed, match, hits / max(live, 1), time.time() - t), flush=True)


def collect(cap):
    probs, folds = {}, []
    for seed in bl.SEEDS:
        probs[seed] = {}
        for match in dfl.MATCHES:
            d = bench._load(str(OUT / ("pred_%s_s%d_sb%g.pkl" % (match, seed, cap))))
            probs[seed].update(d["preds"])
            folds.append(dict(seed=seed, match=match, accuracy=d["accuracy"], checkpoint_sha256=d["checkpoint_sha256"]))
    bench._dump(probs, str(OUT / ("dfl_lomo_probs_sb%g.pkl" % cap)))
    shared = {str(s): hashlib.sha256((OUT / ("bigru_all_s%d_sb%g.pt" % (s, cap))).read_bytes()).hexdigest()
              for s in bl.SEEDS}
    manifest = dict(cap=cap, recipe=dict(zip(("config", "epochs"), bl.recipe())), seeds=bl.SEEDS, folds=folds,
                    all_data_checkpoints=shared, mean_accuracy=float(np.mean([f["accuracy"] for f in folds])))
    (OUT / ("manifest_sb%g.json" % cap)).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("cap %g collected %d folds, mean held-out native-label frame accuracy %.4f" % (
        cap, len(folds), manifest["mean_accuracy"]), flush=True)


def lomo_members(cap):
    """Held-out DFL probabilities per half (bench.blocks order), one list per seed -- exp_verify.lomo_bigru's shape."""
    d = bench._load(str(OUT / ("dfl_lomo_probs_sb%g.pkl" % cap)))
    keys = ["%s/%s" % (b["group"], b["key"].split("/")[-1]) for b in bench.blocks("dfl@v3")]
    return [[np.asarray(d[s][k], np.float64) for k in keys] for s in bl.SEEDS]


def members(src, cap):
    """The three all-data models on src's halves (CPU), cached like exp_shared_blend.neural_probs."""
    import torch
    import exp_ctrl2 as e2
    import seq_pff as seq
    dest = OUT / ("%s_members_sb%g.pkl" % (src.replace("@", "-"), cap))
    if dest.exists():
        return bench._load(str(dest))["members"]
    nets = []
    for s in bl.SEEDS:
        ck = torch.load(OUT / ("bigru_all_s%d_sb%g.pt" % (s, cap)), map_location="cpu", weights_only=False)
        net = seq.Net(345, ck["config"]["kind"], ck["config"]["width"], ck["config"]["drop"])
        net.load_state_dict(ck["state"])
        nets.append((net.eval(), seq.Scaler(ck["mean"], ck["scale"])))
    out = [[] for _ in nets]
    for b in bench.blocks(src):
        raw = seq.views(bench._load(e2._feat_path(b)))
        for preds, (net, sc) in zip(out, nets):
            preds.append(seq.predict_raw(net, sc, raw, len(b["P"])))
    bench._dump(dict(members=out), str(dest))
    print("%s: cap %g members for %d halves" % (src, cap, len(out[0])), flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stage = sys.argv[1]
    if stage == "train":
        train(int(sys.argv[2]), float(sys.argv[3]))
    elif stage == "collect":
        collect(float(sys.argv[2]))
    elif stage == "members":
        members(sys.argv[2], float(sys.argv[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
