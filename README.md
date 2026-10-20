# More than a Hack 2027 — possession labelling (working repo)

Per-frame `H` / `A` / `D` labels for BAR-ATH from the provided automatic tracking, the aligned
broadcast tracking and (optionally) models trained on open DFL data.

> **Data policy.** Match data lives in `data/` and every frame-level derivative in `cache/` and
> `submissions/` — all gitignored. Nothing derived from the match at frame level may be published,
> and no match data or extract may be sent to a hosted LLM (competition T&C). Analysis scripts in
> `src/` print aggregate statistics only.

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
  evaluate.py, proxy.py   official metric on the public window + label-free full-match checks
  make_submission.py      end-to-end submission
  eda*.py, align_*.py     analysis (aggregate output only)
reports/                  analysis outputs (aggregate)
```

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe numpy pandas scipy numba lightgbm kloppy==3.18.0 matplotlib
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into ./data (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/download_idsse.py            # open DFL data, CC BY 4.0, ~2.6 GB
.venv/Scripts/python.exe src/make_submission.py           # model_v5 -> submissions/baseline_v5.csv
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/baseline_v5.csv
```

`model_v5` = alive/dead evidence from rules + the aligned broadcast (v3) and H/A probabilities from a
LightGBM team model trained on degraded open DFL data, decoded with a min-duration Viterbi per half.
`--model model_v3` reproduces the tracking-only baseline. The pipeline is deterministic (identical
MD5 on re-run, including retraining the DFL models from scratch).

## Tools

- `tools/labeler.html` — local hand-labelling of H/A/D transitions for a validation set (allowed by
  the rules for training/validation only). The video never leaves the machine.
- `tools/transitions_to_gt.py` — marks -> 199,319-row `frame,label` file usable as `--gt` by the
  official scorer. Keep it in `cache/` (frame-level derived data).
- `colab/ball_detector_colab.py` — self-supervised ball detector (ATD balls projected through the
  homography as pseudo-labels) for a Colab GPU.
