"""Round 13: independently reproduce shared BiGRU and test its blend with v10.

Only open Metrica/SkillCorner matches are evaluated. Paired training seeds and
the six blend settings are frozen before inference. DFL is explicitly excluded:
all seven matches trained the shared neural checkpoints. No BAR-ATH or public
score is loaded. Frame probabilities stay in cache; reports contain scores only.

    python -I src/exp_shared_blend.py
    python -I src/exp_shared_blend.py --stage report
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import bench
import exp_ctrl2 as e2
import exp_smooth
import model_v7
import seq_pff as seq

OUT = ROOT / "cache/shared_blend_v1"
REPORT = ROOT / "reports/shared_blend"
MODELS = ROOT / "models/bigru_dfl_pff_v1"
SOURCES = ("metrica@v3", "sc@v3")
CONVS = {"metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp")}
PAIRS = ((0, 20261025), (1, 20261026), (2, 20261027))
GRID = tuple((method, weight) for method in ("probability", "logit")
             for weight in (.25, .5, .75))
DEC = (30., 8, 50)
EPS = 1e-4


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def v10_path(seed):
    suffix = "" if seed == 0 else "_seed%d" % seed
    return Path(bench.CACHE) / "combo" / ("probs_a2_bfree%s_wm2.pkl" % suffix)


def freeze():
    manifest = read(MODELS / "manifest.json")
    assert [m["seed"] for m in manifest["models"]] == [p[1] for p in PAIRS]
    for item in manifest["models"]:
        if sha(MODELS / item["file"]) != item["sha256"]:
            raise ValueError("Shared checkpoint hash mismatch")
    design = dict(
        date="2026-10-27", sources=SOURCES, conventions=CONVS, paired_seeds=PAIRS,
        grid=GRID, weight_means="BiGRU contribution; remainder is raw v10",
        smoothing="blend raw probabilities/logits, then centered +/-25-frame logit mean",
        controls=("v10 +/-25", "BiGRU raw", "BiGRU +/-25"),
        ensemble="fixed v10 seed 0 plus equal probability mean of all three BiGRU members",
        decoder=dict(pen_team=12, pen_dead=30, min_poss=8, min_dead=50, D0=175, c=48),
        dead="same seven-DFL learned-D for every candidate; oracle-D diagnostic",
        aggregation="average Metrica degradation draws within match before bootstrap",
        uncertainty="paired match bootstrap of seed-mean scores, 10000 draws, 95% CI",
        selection="worst regret across six external contexts; differences <0.005 keep simpler/current",
        limitations=["DFL cannot be evaluated without leave-match-out neural retraining",
                     "external sets have been reused in prior research",
                     "Metrica has only three independent matches",
                     "label conventions and dead modes are dependent views"],
        barath_or_public_used=False, default_change_authorized_by_this_test=False,
        features=model_v7.VERSION, manifest_sha256=sha(MODELS / "manifest.json"),
        v10_sha256={str(seed): sha(v10_path(seed)) for seed, _ in PAIRS},
        code={name: sha(HERE / name) for name in (
            "exp_shared_blend.py", "seq_pff.py", "bench.py", "decode2.py",
            "exp_ctrl2.py", "exp_smooth.py")},
        runtime=dict(torch=torch.__version__, numpy=np.__version__),
    )
    design = json.loads(json.dumps(design))
    path = REPORT / "design.json"
    if path.exists() and read(path) != design:
        raise ValueError("Frozen design or input artifacts changed; use a new experiment version")
    save(path, design)
    return manifest, design


def validate(probabilities, lengths):
    if len(probabilities) != len(lengths):
        raise ValueError("Wrong half count")
    for p, n in zip(probabilities, lengths):
        p = np.asarray(p)
        if p.shape != (n,) or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
            raise ValueError("Invalid or unaligned probability array")


def blend(a, b, method, weight):
    """Weight belongs to b (BiGRU); operate before the common smoothing step."""
    if method not in ("probability", "logit") or not 0 <= weight <= 1 or len(a) != len(b):
        raise ValueError("Invalid blend specification")
    validate(a, [len(p) for p in b])
    validate(b, [len(p) for p in a])
    if weight == 0:
        return [np.asarray(p, dtype=np.float64).copy() for p in a]
    if weight == 1:
        return [np.asarray(p, dtype=np.float64).copy() for p in b]
    if method == "probability":
        return [(1 - weight) * np.asarray(x, np.float64) + weight * np.asarray(y, np.float64)
                for x, y in zip(a, b)]
    return [e2.sig((1 - weight) * e2.logit(np.asarray(x, np.float64))
                   + weight * e2.logit(np.asarray(y, np.float64))) for x, y in zip(a, b)]


def mean_members(members):
    lengths = [len(p) for p in members[0]]
    for member in members:
        validate(member, lengths)
    return [np.mean(np.stack([m[i].astype(np.float64) for m in members]), axis=0)
            for i in range(len(lengths))]


def neural_probs(source, manifest, design, device):
    blocks = bench.blocks(source)
    lengths = [len(b["P"]) for b in blocks]
    identity = dict(source=source, blocks=[b["key"] for b in blocks], lengths=lengths,
                    manifest=design["manifest_sha256"], code=design["code"], device=device)
    dest = OUT / (source.replace("@", "-") + "_members.pkl")
    if dest.exists():
        cached = bench._load(str(dest))
        if cached["identity"] != identity:
            raise ValueError("Neural inference cache inputs changed")
        for member in cached["members"]:
            validate(member, lengths)
        print(source, "verified cached inference for three shared models", flush=True)
        return cached["members"]
    missing = [b for b in blocks if not Path(e2._feat_path(b)).is_file()]
    if missing:
        del missing
        features = bench.features(model_v7.VERSION, source, procs=2)
        del features
        gc.collect()
    networks = []
    for item in manifest["models"]:
        checkpoint = torch.load(MODELS / item["file"], map_location="cpu", weights_only=False)
        if (checkpoint["step"], checkpoint["core"], checkpoint["halo"]) != (seq.STEP, seq.CORE, seq.HALO):
            raise ValueError("Checkpoint window settings differ")
        config = checkpoint["config"]
        net = seq.Net(2 * len(manifest["feature_names"]) + 1,
                      config["kind"], config["width"], config["drop"]).to(device)
        net.load_state_dict(checkpoint["state"])
        networks.append((net.eval(), seq.Scaler(checkpoint["mean"], checkpoint["scale"])))
    members = [[] for _ in networks]
    for i, block in enumerate(blocks):
        features = bench._load(e2._feat_path(block))
        if features["fk"] + features["tk"] != manifest["feature_names"]:
            raise ValueError("Checkpoint feature order differs")
        raw = seq.views(features)
        for predictions, (net, scaler) in zip(members, networks):
            predictions.append(seq.predict_raw(net, scaler, raw, lengths[i]))
        del features, raw
        if (i + 1) % 5 == 0 or i + 1 == len(blocks):
            print(source, "inference halves", i + 1, "/", len(blocks), flush=True)
    del networks
    for member in members:
        validate(member, lengths)
    OUT.mkdir(parents=True, exist_ok=True)
    bench._dump(dict(identity=identity, members=members), str(dest))
    return members


def decode(pds, phs):
    validate(phs, [len(p) for p in pds])
    def one(pair):
        return e2.decode_all(DEC, [pair[0]], [pair[1]])[0]
    first = one((pds[0], phs[0]))
    with ThreadPoolExecutor(max_workers=int(os.environ.get("GATE_DECODE_THREADS", "6"))) as pool:
        return [first] + list(pool.map(one, zip(pds[1:], phs[1:])))


def by_match(blocks, labels, predictions):
    groups = bench.spell_by_group(blocks, labels, predictions)
    matches = {}
    for (match, _), score in groups.items():
        matches.setdefault(str(match), []).append(float(score[0]))
    return {m: float(np.mean(scores)) for m, scores in matches.items()}


def bootstrap(a, b):
    if set(a) != set(b) or not a:
        raise ValueError("Paired match scores required")
    delta = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(20261027)
    sampled = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    return dict(delta=float(delta.mean()), ci95=np.quantile(sampled, [.025, .975]).tolist(),
                probability_positive=float((sampled > 0).mean()), wins=int((delta > 1e-12).sum()),
                matches=len(delta))


def evaluate(source, manifest, design, device):
    result_path = REPORT / (source.replace("@", "-") + "_results.json")
    if result_path.exists():
        print(source, "completed results retained", flush=True)
        return
    blocks = bench.blocks(source)
    lengths = [len(b["P"]) for b in blocks]
    members = neural_probs(source, manifest, design, device)
    v10 = [bench._load(str(v10_path(seed)))[source] for seed, _ in PAIRS]
    for member in v10:
        validate(member, lengths)
    labels = {conv: bench.labels(source, conv) for conv in CONVS[source]}
    first = next(iter(labels.values()))
    assert all(all(np.array_equal(a == 2, b == 2) for a, b in zip(first, labs))
               for labs in labels.values())
    dead = dict(learned=bench.dead_ext("base", "v3", source),
                oracle=[np.where(lab == 2, 1 - EPS, EPS) for lab in first])
    if source == "metrica@v3":
        old_dead = bench.dead_probs(source, "base")
        delta = max(float(np.max(np.abs(a - b))) for a, b in zip(dead["learned"], old_dead))
        save(REPORT / "dead_consistency.json", dict(max_absolute_probability_difference=delta,
             same_as_v10_benchmark=delta < 1e-12))
        if delta >= 1e-12:
            raise ValueError("Published v10 and shared BiGRU used different dead channels")
    rows = []
    trials = [(str(seed), a, b) for (seed, _), a, b in zip(PAIRS, v10, members)]
    trials.append(("ensemble", v10[0], mean_members(members)))
    for trial, a, b in trials:
        candidates = [("v10", exp_smooth.smooth(a, 25)), ("bigru_raw", b),
                      ("bigru_smooth25", exp_smooth.smooth(b, 25))]
        candidates += [("%s_b%g" % (method, weight), exp_smooth.smooth(blend(a, b, method, weight), 25))
                       for method, weight in GRID]
        for mode, pds in dead.items():
            for name, phs in candidates:
                predictions = decode(pds, phs)
                for conv, labs in labels.items():
                    scores = by_match(blocks, labs, predictions)
                    rows.append(dict(source=source, conv=conv, dead=mode, trial=trial,
                                     model=name, spellF1=float(np.mean(list(scores.values()))),
                                     spell_by_match=scores))
                del predictions
            print(source, "trial", trial, mode, "nine candidates complete", flush=True)
        del candidates
    save(result_path, rows)
    bench._blocks.clear()
    bench._labels.clear()
    gc.collect()


def report():
    rows = [row for source in SOURCES
            for row in read(REPORT / (source.replace("@", "-") + "_results.json"))]
    assert len(rows) == 216
    prior = read(ROOT / "reports/external_bigru_ensemble/results.json")
    checks = []
    summary = []
    for source in SOURCES:
        for conv in CONVS[source]:
            for mode in ("learned", "oracle"):
                context = [r for r in rows if (r["source"], r["conv"], r["dead"]) == (source, conv, mode)]
                models = sorted({r["model"] for r in context})
                reference = next(r for r in context if r["model"] == "v10" and r["trial"] == "ensemble")
                seed_references = {r["trial"]: r for r in context if r["model"] == "v10" and r["trial"] != "ensemble"}
                ref_mean = {m: float(np.mean([r["spell_by_match"][m] for r in seed_references.values()]))
                            for m in reference["spell_by_match"]}
                for name in models:
                    seeds = [r for r in context if r["model"] == name and r["trial"] != "ensemble"]
                    ens = next(r for r in context if r["model"] == name and r["trial"] == "ensemble")
                    assert len(seeds) == 3
                    seed_mean = {m: float(np.mean([r["spell_by_match"][m] for r in seeds]))
                                 for m in reference["spell_by_match"]}
                    item = dict(source=source, conv=conv, dead=mode, model=name,
                                seed_mean=float(np.mean(list(seed_mean.values()))),
                                seed_sd=float(np.std([r["spellF1"] for r in seeds], ddof=1)),
                                seed_deltas=[r["spellF1"] - seed_references[r["trial"]]["spellF1"] for r in seeds],
                                versus_v10=bootstrap(seed_mean, ref_mean),
                                ensemble=ens["spellF1"],
                                ensemble_versus_v10=bootstrap(ens["spell_by_match"], reference["spell_by_match"]))
                    summary.append(item)
                    if name == "bigru_raw":
                        old = next(r for r in prior if (r["model"], r["source"], r["conv"], r["dead"])
                                   == ("bigru_ensemble", source, conv, mode))
                        error = max(abs(ens["spell_by_match"][m] - old["spell_by_match"][m]) for m in ens["spell_by_match"])
                        checks.append(dict(source=source, conv=conv, dead=mode,
                                           prior=old["spellF1"], reproduced=ens["spellF1"],
                                           max_match_score_difference=error, exact=error < 1e-12))
    regrets = []
    for name in sorted({r["model"] for r in summary}):
        contexts = [r for r in summary if r["model"] == name]
        worst = max(max(r["seed_mean"] for r in summary if (r["source"], r["conv"], r["dead"])
                        == (c["source"], c["conv"], c["dead"])) - c["seed_mean"] for c in contexts)
        ens_worst = max(max(r["ensemble"] for r in summary if (r["source"], r["conv"], r["dead"])
                            == (c["source"], c["conv"], c["dead"])) - c["ensemble"] for c in contexts)
        regrets.append(dict(model=name, seed_mean_worst_regret=worst, ensemble_worst_regret=ens_worst))
    regrets.sort(key=lambda r: r["seed_mean_worst_regret"])
    save(REPORT / "summary.json", summary)
    save(REPORT / "regret.json", regrets)
    save(REPORT / "verification.json", dict(cells=len(rows), external_reproduction=checks,
         all_prior_ensemble_scores_exact=all(c["exact"] for c in checks), checkpoints_verified=True,
         independent_matches=dict(metrica=3, skillcorner=20), dfl_evaluated=False,
         default_changed=False, public_used=False))
    lines = ["# v10 + shared BiGRU: round 13 external evaluation", "",
             "Frozen probability/logit weights 0.25, 0.5, 0.75; common +/-25-frame logit smoothing and B2.",
             "Three paired training-seed trials; bootstrap resamples matches after averaging noise/training seeds.",
             "Ensemble column uses the existing v10 seed 0 and the three equal-weight shared BiGRU members.", "",
             "| Context | D | Candidate | Three-seed mean | Delta vs v10 (95% CI) | Ensemble | Ensemble delta |",
             "|---|---|---|---:|---|---:|---:|"]
    for r in summary:
        d = r["versus_v10"]
        lines.append("| %s/%s | %s | %s | %.6f | %+.6f [%+.6f, %+.6f] | %.6f | %+.6f |" % (
            r["source"], r["conv"], r["dead"], r["model"], r["seed_mean"], d["delta"],
            *d["ci95"], r["ensemble"], r["ensemble_versus_v10"]["delta"]))
    lines += ["", "| Candidate | Worst regret, seed mean | Worst regret, ensemble |",
              "|---|---:|---:|"]
    for r in regrets:
        lines.append("| %s | %.6f | %.6f |" % (r["model"], r["seed_mean_worst_regret"], r["ensemble_worst_regret"]))
    lines += ["", "DFL is excluded: all shared checkpoints trained on all seven DFL matches.",
              "These external sets were previously inspected; results are exploratory, not a fresh holdout.",
              "Metrica has only three independent games. Label/dead contexts are correlated views.",
              "The match bootstrap does not include training-data or grid-selection uncertainty.",
              "No BAR-ATH data, public score, or automatic default change is part of this experiment.",
              "v10 remains the submission pending DFL leave-match-out neural validation."]
    (REPORT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in summary:
        if r["model"] in ("v10", "bigru_raw", regrets[0]["model"]):
            print(r["source"], r["conv"], r["dead"], r["model"],
                  "seed mean %.6f; ensemble %.6f; delta %+.6f" % (
                      r["seed_mean"], r["ensemble"], r["versus_v10"]["delta"]), flush=True)
    print("Worst-case regrets:", json.dumps(regrets), flush=True)
    print("Prior ensemble scores reproduced:", all(c["exact"] for c in checks), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "predict", "evaluate", "report"), default="all")
    parser.add_argument("--source", choices=("all",) + SOURCES, default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    if args.device == "cuda" and not Path("/content").is_dir():
        raise RuntimeError("GPU work must run in Google Colab")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest, design = freeze()
    sources = SOURCES if args.source == "all" else (args.source,)
    if args.stage in ("all", "predict", "evaluate"):
        for source in sources:
            if args.stage == "predict":
                neural_probs(source, manifest, design, args.device)
                bench._blocks.clear()
                gc.collect()
            else:
                evaluate(source, manifest, design, args.device)
    if args.stage == "report" or args.stage == "all" and args.source == "all":
        report()


if __name__ == "__main__":
    main()
