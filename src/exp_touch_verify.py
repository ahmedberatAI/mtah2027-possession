"""Round 13: three-seed confirmation of touch v2 after the DFL-evctrl screen.

Use existing leave-game/match-out predictions, never BAR-ATH/public scores.
Average simulation draws within each match before paired match bootstrap.
The original seed-0 screening result is explicitly not independent evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import bench
import exp_ctrl2 as e2
import exp_smooth
from exp_shared_blend import bootstrap, by_match, save, sha, v10_path

REPORT = HERE.parent / "reports/touch_v2_confirmation"
CONVS = {"dfl@v3": ("trk", "evctrl"), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp")}
SEEDS = (0, 1, 2)


def touch_path(seed):
    return Path(bench.CACHE) / "touch" / ("probs2%s.pkl" % ("_seed%d" % seed if seed else ""))


def decode(decoder, pds, phs):
    def one(pair):
        return e2.decode_all(decoder, [pair[0]], [pair[1]])[0]
    first = one((pds[0], phs[0]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        return [first] + list(pool.map(one, zip(pds[1:], phs[1:])))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    artifacts = {str(seed): dict(v10=sha(v10_path(seed)), touch_v2=sha(touch_path(seed))) for seed in SEEDS}
    save(REPORT / "design.json", dict(
        date="2026-10-27", training_seeds=SEEDS, artifacts=artifacts, sources=CONVS,
        protocol="fixed v10 training/decoder; add touch v2; smooth logits +/-25 frames",
        aggregation="equal matches, Metrica noise draws averaged within match first",
        uncertainty="paired match bootstrap of seed-mean differences, 10000 draws, 95% CI",
        selection_bias="seed 0 was screened among approximately 35 candidates; seeds 1/2 are confirmation",
        barath_public_used=False, default_changed=False,
    ))
    rows, summary = [], []
    for source, conventions in CONVS.items():
        blocks = bench.blocks(source)
        labels = {c: bench.labels(source, c) for c in conventions}
        labs = next(iter(labels.values()))
        dead = dict(learned=bench.dead_ext("base", "v3", source) if source.startswith("sc")
                    else bench.dead_probs(source, "base"),
                    oracle=[np.where(lab == 2, 1 - 1e-4, 1e-4) for lab in labs])
        decoder = (32., 5, 25) if source.startswith("dfl") else (30., 8, 50)
        for seed in SEEDS:
            base = exp_smooth.smooth(bench._load(str(v10_path(seed)))[source], 25)
            touch = exp_smooth.smooth(bench._load(str(touch_path(seed)))[source], 25)
            for mode, pds in dead.items():
                for name, phs in (("v10", base), ("touch_v2", touch)):
                    predictions = decode(decoder, pds, phs)
                    for conv in conventions:
                        scores = by_match(blocks, labels[conv], predictions)
                        rows.append(dict(source=source, conv=conv, dead=mode, seed=seed,
                                         model=name, spell_by_match=scores,
                                         spellF1=float(np.mean(list(scores.values())))))
                    del predictions
            print(source, "seed", seed, "complete", flush=True)
        bench._blocks.clear()
        bench._labels.clear()
        del blocks, labels, dead, base, touch
        gc.collect()
    for source, conventions in CONVS.items():
        for conv in conventions:
            for mode in ("learned", "oracle"):
                context = [r for r in rows if (r["source"], r["conv"], r["dead"]) == (source, conv, mode)]
                base = {r["seed"]: r for r in context if r["model"] == "v10"}
                touch = {r["seed"]: r for r in context if r["model"] == "touch_v2"}
                means = [{m: float(np.mean([family[s]["spell_by_match"][m] for s in SEEDS]))
                          for m in base[0]["spell_by_match"]} for family in (base, touch)]
                row = dict(source=source, conv=conv, dead=mode,
                           v10=float(np.mean(list(means[0].values()))),
                           touch_v2=float(np.mean(list(means[1].values()))),
                           seed_deltas=[touch[s]["spellF1"] - base[s]["spellF1"] for s in SEEDS],
                           versus_v10=bootstrap(means[1], means[0]))
                summary.append(row)
                print(source, conv, mode, "v10 %.6f -> touch v2 %.6f; delta %+.6f" % (
                    row["v10"], row["touch_v2"], row["versus_v10"]["delta"]), flush=True)
    save(REPORT / "results.json", rows)
    save(REPORT / "summary.json", summary)
    lines = ["# Touch v2: three-seed confirmation", "",
             "Seed 0 was screened on DFL-evctrl; seeds 1 and 2 are the confirmation runs.",
             "Equal-match scores, with Metrica noise draws averaged before bootstrap. No BAR-ATH/public selection.", "",
             "| Context | D | v10 | Touch v2 | Delta (95% CI) | Seed deltas |",
             "|---|---|---:|---:|---|---|"]
    for r in summary:
        d = r["versus_v10"]
        lines.append("| %s/%s | %s | %.6f | %.6f | %+.6f [%+.6f, %+.6f] | %s |" % (
            r["source"], r["conv"], r["dead"], r["v10"], r["touch_v2"], d["delta"], *d["ci95"],
            ", ".join("%+.6f" % delta for delta in r["seed_deltas"])))
    lines += ["", "Match-bootstrap intervals do not include training-data or candidate-screening uncertainty.",
              "v10 remains the submission; assess every context rather than the screened DFL-evctrl cell alone."]
    (REPORT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
