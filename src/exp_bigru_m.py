"""Round 21 (local GPU): the shared BiGRU recipe with the Metrica sample games in training.

The shared BiGRU (models/bigru_dfl_pff_v1) learned from DFL + PFF and never saw Metrica's tracking or its controlled-
possession convention, although Metrica is the provider of BAR-ATH and the closest open set. Here the three Metrica
games (their three simulated degradations, `ctrl` labels) join the optical side of the 50/50 DFL / PFF minibatches as
three more matches; recipe, scaler seed, seeds and features unchanged (src/exp_bigru_lomo.py). Optionally with the
round-18 spell-balanced loss (CAP, src/exp_bigru_sb.py). Every DFL match and every Metrica game gets a model that never
saw it (ten folds per seed); one model per seed on all data serves SkillCorner and BAR-ATH. The 'plain' mode drops the
Metrica rows (seven DFL folds + all; Metrica then from the all-data models): the Colab recipe retrained on this machine,
the control that separates a variant's effect from the training environment's. AGGREGATE output only.

  python -I src/exp_bigru_m.py index                         sequence rows of the Metrica games (after exp_bigru_lomo prepare)
  python -I src/exp_bigru_m.py train SEED [CAP] [plain]      held-out folds + the all-data model (GPU)
  python -I src/exp_bigru_m.py collect [CAP] [plain]         held-out DFL (/ Metrica) probabilities, all-data members
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

ROOT = HERE.parent
OUT = ROOT / "cache" / "bigru_m"
GAMES = ("g1", "g2", "g3")


def tag(cap, metrica=True):
    base = "m" if metrica else "p"
    return base if cap is None else "%s_sb%g" % (base, cap)


def _folds(metrica):
    return list(dfl.MATCHES) + (["metrica-%s" % g for g in GAMES] if metrica else []) + ["all"]


def index():
    """The shared recipe's index plus the Metrica halves (source 'dfl', match 'metrica-g<k>')."""
    import exp_ctrl2 as e2
    import exp_seq_pff as neural
    import seq_pff as seq
    rows = neural.read(neural.OUT / "index.json")
    labs = bench.labels("metrica@v3", "ctrl")
    for b, lab in zip(bench.blocks("metrica@v3"), labs):
        key = "metrica_%s" % b["key"].split("/", 1)[1].replace("/", "_")
        xp, yp = neural.OUT / "features" / (key + ".npy"), neural.OUT / "features" / (key + "_labels.npy")
        if not xp.exists():
            np.save(xp, seq.views(bench._load(e2._feat_path(b))))
        np.save(yp, np.asarray(lab).astype(np.int8))
        rows.append(dict(key=key, source="dfl", match="metrica-%s" % b["group"], x=str(xp), y=str(yp),
                         full_length=len(lab), split="train", bench_key=b["key"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(rows), encoding="utf-8")
    print("index: %d rows (%d Metrica halves)" % (len(rows), sum(r["match"].startswith("metrica") for r in rows)),
          flush=True)


def _held_key(r):
    """bench-style key of a held-out half: DFL '<match>/h<k>', Metrica the bench block key."""
    return r.get("bench_key") or "%s/%s" % (r["match"], r["key"].rsplit("_", 1)[-1])


def train(seed, cap=None, metrica=True):
    import torch
    import exp_seq_pff as neural
    import seq_pff as seq
    sb = None
    if cap is not None:
        import exp_bigru_sb as sb
        seq.balanced_loss = sb.weighted_loss
    config, epochs = bl.recipe()
    rows = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    if not metrica:
        rows = [r for r in rows if not r["match"].startswith("metrica")]
    neural.REPORT = OUT
    for match in _folds(metrica):
        dest = OUT / ("pred_%s_s%d_%s.pkl" % (match, seed, tag(cap, metrica)))
        if dest.exists():
            continue
        t = time.time()
        keep = [r for r in rows if r["match"] != match]
        data = neural.Data(keep) if sb is None else sb.weighted_data(keep, cap)
        neural.SEED = bl.SCALER_SEED
        scaler = data.scaler()
        neural.SEED = seed
        ckpt = OUT / ("bigru_%s_s%d_%s.pt" % (match, seed, tag(cap, metrica)))
        neural.train_one(config, data, scaler, epochs, ckpt)
        net, sc, _ = neural.load_model(ckpt)
        preds, hits, live = {}, 0, 0
        for r in rows:
            if r["match"] == match:
                p = seq.predict_raw(net, sc, np.load(r["x"], mmap_mode="r"), r["full_length"]).astype(np.float32)
                y = np.load(r["y"])
                ok = y < 2
                hits += int(((p > 0.5) == (y == 0))[ok].sum())
                live += int(ok.sum())
                preds[_held_key(r)] = p
        del net
        torch.cuda.empty_cache()
        bench._dump(dict(preds=preds, accuracy=hits / max(live, 1), seconds=time.time() - t,
                         checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()), str(dest))
        print("%s seed %d held-out %s: frame accuracy %.4f, %.0f s" % (tag(cap, metrica), seed, match,
                                                                      hits / max(live, 1), time.time() - t), flush=True)


def collect(cap=None, metrica=True):
    """{'dfl@v3': [members x halves], 'metrica@v3': [...], 'sc@v3': [...]} in bench.blocks order; Metrica held out
    when the games were in training, else (plain) from the all-data models like SkillCorner."""
    import torch
    import exp_ctrl2 as e2
    import seq_pff as seq
    out = {}
    held = {}
    for seed in bl.SEEDS:
        held[seed] = {}
        for match in _folds(metrica)[:-1]:
            held[seed].update(bench._load(str(OUT / ("pred_%s_s%d_%s.pkl" % (match, seed, tag(cap, metrica)))))["preds"])
    for src in ("dfl@v3", "metrica@v3")[:2 if metrica else 1]:
        keys = [b["key"] if src.startswith("metrica") else "%s/%s" % (b["group"], b["key"].split("/")[-1])
                for b in bench.blocks(src)]
        out[src] = [[np.asarray(held[s][k], np.float64) for k in keys] for s in bl.SEEDS]
    nets = []
    for s in bl.SEEDS:
        ck = torch.load(OUT / ("bigru_all_s%d_%s.pt" % (s, tag(cap, metrica))), map_location="cpu", weights_only=False)
        net = seq.Net(345, ck["config"]["kind"], ck["config"]["width"], ck["config"]["drop"])
        net.load_state_dict(ck["state"])
        nets.append((net.eval(), seq.Scaler(ck["mean"], ck["scale"])))
    for src in ("sc@v3",) if metrica else ("metrica@v3", "sc@v3"):
        out[src] = [[] for _ in nets]
        for b in bench.blocks(src):
            raw = seq.views(bench._load(e2._feat_path(b)))
            for preds, (net, sc) in zip(out[src], nets):
                preds.append(seq.predict_raw(net, sc, raw, len(b["P"])).astype(np.float64))
    bench._dump(out, str(OUT / ("members_%s.pkl" % tag(cap, metrica))))
    print("collected %s: DFL %d held-out halves, Metrica %d, SkillCorner %d halves" % (
        tag(cap, metrica), len(out["dfl@v3"][0]), len(out["metrica@v3"][0]), len(out["sc@v3"][0])), flush=True)


def members(src, cap=None, metrica=True):
    return bench._load(str(OUT / ("members_%s.pkl" % tag(cap, metrica))))[src]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stage, args = sys.argv[1], [a for a in sys.argv[2:] if a != "plain"]
    metrica = "plain" not in sys.argv[2:]
    if stage == "index":
        index()
    elif stage == "train":
        train(int(args[0]), float(args[1]) if len(args) > 1 else None, metrica)
    elif stage == "collect":
        collect(float(args[0]) if args else None, metrica)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
