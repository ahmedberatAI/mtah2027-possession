# More than a Hack 2027 — possession labelling (working repo)

Per-frame `H` / `A` / `D` labels for BAR-ATH from the provided automatic tracking, the aligned
broadcast tracking and (optionally) models trained on open DFL data.

> **Data policy.** Match data lives in `data/` and every frame-level derivative in `cache/` and
> `submissions/` — all gitignored. Nothing derived from the match at frame level may be published,
> and no match data or extract may be sent to a hosted LLM (competition T&C). Analysis scripts in
> `src/` print aggregate statistics only.

## Ideas, experiments and results

Everything we tried and what it gave lives in one document: **[`docs/RAPOR.md`](docs/RAPOR.md)** (Turkish). It
opens with a summary of the current model and scores, then the permanent rules, the one-line-per-experiment log, the
idea pool with the round-by-round notes, and the early topic reports as appendices. Every new experiment is added
there (a row at the end of the log table in §3, details under the latest round in §4).

Fikirler, denemeler ve sonuçlar tek bir belgede: **[`docs/RAPOR.md`](docs/RAPOR.md)**. Her yeni deneme oraya eklenir.

## Submission

- **The file:** `submissions/candidate_v10.csv` (model_v10 without the whistle option), MD5
  `ca20c5b657a395d2f7c0e310cdaabd64`. `python src/make_submission.py` regenerates it.
- **Model file:** the team classifier is trained on open data only and kept in `models/v10_team.txt`, so the default
  run needs only the provided data (about a minute). `MTAH_RETRAIN=1` retrains it from the open data (about 7 min)
  byte for byte.
- **Technical document:** [`docs/TECHNICAL_DOCUMENT.md`](docs/TECHNICAL_DOCUMENT.md) (English).
- **Public repository content:** `python tools/export_public.py <empty folder>` copies the files that regenerate the
  submission (list in `tools/public/files.txt`), with `tools/public/README.md` as their README. The research log and
  `reports/` stay private.

## Layout

```
data/                     match bundle (gitignored): tactical/, broadcast/
external/more-than-a-hack-2027/   official repo clone (scorer, public GT, starter)
external_data/idsse/      open DFL data (Bassek et al., CC BY 4.0), optional
src/
  atd.py                  ATD parser + cache (video_frame = frame_id - 1)
  features.py             per-frame features (ball, nearest players, tempo, tags, camera)
  broadcast.py            broadcast -> tactical mapping (lag 4014 / 26920 frames per half)
  decode.py               min-duration Viterbi (numba)
  model_v1.py .. v4.py    rule/evidence models + decoding
  mlfeat.py, sim.py, dfl.py, train_dfl.py   learned emissions from degraded open DFL data
  bench.py, paired.py     shared CPU test bench for model ideas (DFL / Metrica / BAR-ATH aggregates)
  feat_*.py, decode2.py   idea plugins (features, generalised decoder); model_v7.py candidate
  evaluate.py, proxy.py   official metric on the public window + label-free full-match checks
  make_submission.py      end-to-end submission
  eda*.py, align_*.py     analysis (aggregate output only)
docs/RAPOR.md             the single log of ideas, experiments and results
docs/TECHNICAL_DOCUMENT.md   the technical document of the submission
models/v10_team.txt       the submitted team classifier (trained on open data only)
tools/export_public.py    builds the public repository's content in a separate folder
reports/                  raw aggregate outputs of the analysis scripts (txt / json / png)
```

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe numpy pandas scipy numba lightgbm kloppy==3.18.0 matplotlib
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into ./data (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/download_idsse.py            # open DFL data, CC BY 4.0, ~2.6 GB
.venv/Scripts/python.exe src/make_submission.py           # the submission: model_v10 -> submissions/candidate_v10.csv
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v10.csv
```

**The submission is `model_v10` without the referee-whistle option** (`submissions/candidate_v10.csv`, MD5
`ca20c5b657a395d2f7c0e310cdaabd64`). The model history below is kept for reference.

`model_v5` = alive/dead evidence from rules + the aligned broadcast (v3) and H/A probabilities from a
LightGBM team model trained on degraded open DFL data, decoded with a min-duration Viterbi per half.
`--model model_v3` reproduces the tracking-only baseline. The pipeline is deterministic (identical
MD5 on re-run, including retraining the DFL models from scratch).

Candidate `model_v7` (`--model model_v7 --out submissions/candidate_v7.csv`): same alive/dead evidence,
team model trained on DFL degraded with `sim.V3` (smooth tracking noise) with pressure-aware control,
set-piece formation, team shape and movement features, decoded with a soft minimum possession length
(`src/decode2.py`). Ideas, the shared CPU test bench (`src/bench.py`) and every result: `docs/RAPOR.md`.

Candidate `model_v8` (`--model model_v8`): v7 with the team model trained on open DFL plus the three Metrica
sample games (controlled-possession labels). Candidate **`model_v9`** (`--model model_v9`; with the referee whistle
`--params '{"wh_bonus": 30}'`): v8 plus ball-free short-horizon team interaction features (`src/bfree.py`) and its
team probabilities smoothed over ±1 s before the same decoder; **`model_v10`** is v9 with the Metrica games weighted
2 in training. On held-out external matches v9 / v10 beat v7 in every evaluation context (DFL-trk, Metrica,
SkillCorner; paired match bootstrap, `src/exp_boot.py`); v10 is the submission (the default of
`make_submission.py`; the whistle option `--params '{"wh_bonus": 30}'` is not used); v7 stays reproducible bit for bit. Option `model_v11`: v10 plus ball-touch
event features from the tracking (`src/touch.py`) — on a par with v10 over three LightGBM seeds (mean DFL +0.002 /
-0.000, Metrica-ctrl +0.001 / +0.000, SkillCorner +0.000…+0.002), inside the decision protocol's tie margin and not
significant under the paired bootstrap. Option `model_v12` (`--model model_v12`): v10's probabilities and decoder
plus a touch-aware switch cost — where the tracking's touch sequence changes team, a change of possession into that
team within ±6 frames costs 8 less (no retraining). Averaged over five model instances it adds ≈ +0.002 on
Metrica-ctrl and ≈ 0 elsewhere (the first, seed-0 instance looked like +0.003 everywhere), inside the tie margin.
v10 (without the whistle) stays the submission.

## Tools

- `tools/labeler.html` — local hand-labelling of H/A/D transitions for a validation set (allowed by
  the rules for training/validation only). The video never leaves the machine.
- `tools/transitions_to_gt.py` — marks -> 199,319-row `frame,label` file usable as `--gt` by the
  official scorer. Keep it in `cache/` (frame-level derived data).
- `colab/ball_detector_colab.py` — self-supervised ball detector (ATD balls projected through the
  homography as pseudo-labels) for a Colab GPU.
