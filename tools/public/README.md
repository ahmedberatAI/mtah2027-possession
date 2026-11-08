# More than a Hack 2027 — possession labelling (BAR-ATH)

Per-frame `H` / `A` / `D` labels for the BAR-ATH match of More than a Hack 2027 (Phase 1), produced end to end from
the provided data and open data:

- **Dead ball:** a small rule model on player tempo and ball state, a dead-ball classifier, a sequence model whose inputs are
  scaled by each half's own level, and the time-aligned broadcast recording's view; all weights and models come from open data.
- **Team in possession:** a LightGBM classifier on team-level tracking features, trained on open tracking (IDSSE /
  DFL, Metrica Sports sample games) degraded to look like automatic single-camera tracking, averaged in equal parts
  with a bidirectional GRU trained on open DFL and PFF World Cup 2022 tracking (`models/bigru_dfl_pff_v1`).
- **Short-possession specialist:** a ball-centred tracking model trained on open PFF World Cup 2022 data, with
  PFF's possessions in the challenge's controlled-possession reading (`models/kpu_c1`), then fine-tuned on the open
  DFL and Metrica games in the same reading (`models/kpu_j1`); its probability of control is half of the team
  probability.
- **Ball-free channel:** a model that reads only the players, never the ball: every player of both teams as a set per
  step, a bidirectional GRU, trained on open PFF World Cup 2022 data alone (`models/ballfree_v1`, mean of three seeds).
- **Stacker:** a small sequence model on all the channels' outputs, a path model in which possession passes to the other
  team only through its control state; trained on their held-out outputs on the open games only (`models/stack_v1`). It
  supplies half of the team evidence and half of the alive evidence.
- **Three seeds:** the team classifier, the specialist and the stacker each come in three seeds (`models/v10_seeds/`,
  `models/kpu_j1/`, `models/stack_v1/`).
- **Decoding:** the stacker is a path model whose samples are calibrated (as many possessions, of the same lengths, as
  the truth on open data). The labelling of each half is the metric's Bayes decision on those samples
  (`src/stack_decode.py`): 3,000 paths from the three stackers; candidates are their segments and a duration-aware
  Viterbi path; each possession is kept when its mean chance of matching over the IoU thresholds exceeds 0.5.

The approach, the validation protocol and the results are in **[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md)**.

## Reproduce the submission

```bash
# Python 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into ./data/{tactical,broadcast}  (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/make_submission.py           # -> submissions/candidate_v22.csv (a few minutes)
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v22.csv

# optional: retrain the team classifier from the open data (rewrites models/v10_team.txt, about 7 min on 24 cores)
.venv/Scripts/python.exe src/download_idsse.py            # open DFL / IDSSE data, CC BY 4.0, ~2.6 GB
.venv/Scripts/python.exe src/metrica_sample.py download   # Metrica Sports sample games, pinned commit, ~185 MB
MTAH_RETRAIN=1 .venv/Scripts/python.exe src/make_submission.py
```

- On Linux / macOS, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.
- `.gitattributes` keeps the LightGBM model files (`models/*.txt`, `models/**/*.txt`) byte for byte: LightGBM 4.7 cannot read them with
  the CRLF line endings a Windows checkout would otherwise write (the pipeline also reads them as text).
- `models/v10_team.txt` is the team classifier, `models/kpu_j1/kpu_J1_s20261102.pt` the specialist,
  `models/dead_multi_v2.txt` the dead-ball classifier, `models/tdead_v1/` the sequence dead-ball model and
  `models/dead_rules_open_v2.json` the rule weights, `models/ballfree_v1/` the ball-free channel and `models/stack_v1/`
  the stacker (three seeds each with `models/v10_seeds/` and `models/kpu_j1/`), all from open data only, so the default run needs
  only the provided data. On its first run it rebuilds the BiGRU's BAR-ATH probabilities on the CPU
  (`src/reproduce_shared_bigru.py`, checked against `models/bigru_dfl_pff_v1/manifest.json`) and runs the specialist,
  the sequence model, the ball-free channel and the stacker on the CPU.
- The run is deterministic: fixed simulation seeds, LightGBM deterministic mode with a fixed seed and thread count,
  and no GPU. Both the default run and a retrain of the team classifier from an empty `cache/` write the submitted
  file, MD5 `b4c1a9e50638da8a268b1fc0d0506ff0` (the decision's random draws use fixed seeds). `--model model_v21 --out
  submissions/candidate_v21.csv` writes the previous submission (MD5 `10ee3c08e6a5d5de1d3951652eca26f2`),
  `--model model_v20 --out submissions/candidate_v20.csv` v20 (MD5 `d909212952998c76568b09c17b76b5d5`), `--model model_v19 --out submissions/candidate_v19.csv`
  v19 (MD5 `45096061ebdc3aac6e2087d8a047246a`), `--model model_v15 --out submissions/candidate_v15.csv`
  v15 (MD5 `6081216bcf484049d8e6e84b27fc85dd`), `--model model_v14 --out submissions/candidate_v14.csv` v14 (MD5
  `5e7363fc83c6f56a923033894a28d3a1`) and `--model model_v13 --out submissions/candidate_v13.csv` v13 (MD5
  `ed8ad739207f367017bb8d8c8204c82a`).

## Layout

```
src/make_submission.py   end to end: provided data -> submissions/candidate_v22.csv
src/model_v22.py         the submitted model: v21 over three seeds and the Bayes decision
src/stack_decode.py      the Bayes decision: samples of the path models, candidates, values, dynamic programming
src/model_v21.py         the previous submission: v20's channels, the ball-free channel and the stacker
src/stack_model.py       the stacker: inputs, network, path model
src/exp_ballfree.py      the ball-free channel: inputs, network, training
src/model_v20.py         the previous submission: the specialist at 0.5, dead-ball evidence with the sequence model
src/exp_tdead.py         the sequence dead-ball model: network, per-half normalisation, training
src/dead_v6_fit.py       re-derives v20's rule weights and dead-ball models from open data on sim.V6's masks
src/model_v19.py         the previous submission: v17's team side, dead-ball evidence free of the public window
src/dead_rules_fit.py    re-derives the rule constants from open data
src/model_v18.py         the previous fusion: v3's rules + the broadcast + half the DFL dead model
src/model_v17.py         the specialist J1 in C1's place (on v16's recipe)
src/model_v16.py         model_v15 with its dead-ball boundaries timed by the learned dead model (not in v18)
src/dead_timing.py       the learned dead model's alive log-odds and own decode, v16's timing step
src/model_v15.py         the recipe under it: v14's blend mixed with a specialist + the verifier on that base
src/short_specialist.py  the specialist: inputs, network, inference, its candidates and verifier features
src/model_v14.py         the previous submission: 0.5 v10 + 0.5 BiGRU blend + the short-possession verifier
src/model_v13.py         model_v10 + the verifier (src/short_verifier.py)
src/model_v10.py         team classifier training + prediction, decoder settings
models/ballfree_v1/      the ball-free channel's three checkpoints (open PFF data only)
models/stack_v1/         the stacker's checkpoints, three seeds (the channels' outputs on the open games)
models/v10_seeds/        the team classifier's seeds 1 and 2
models/v10_team.txt      the team classifier (LightGBM, trained on the open data only)
models/short_verifier_*  the verifier's classifiers (LightGBM, open data only; *_j1m25s = v17's, *_c1m25s = v15's,
                         *_blend50 = v14's)
models/dead_multi_v2.txt the dead-ball classifier (LightGBM, open DFL, Metrica and SkillCorner data, equal provider weight, sim.V6)
models/tdead_v1/         the sequence dead-ball model's three checkpoints (the same open data, sim.V6)
models/dead_rules_open_v2.json  the rule weights (open data only, sim.V6)
models/dead_multi_v1.txt, models/dead_rules_open_v1.json   v19's
models/dead_base_v3.txt  the DFL dead-ball classifier of v16 to v18
models/bigru_dfl_pff_v1/ the shared BiGRU checkpoints; src/reproduce_shared_bigru.py and src/seq_pff.py run them
models/kpu_j1/           the specialist J1's checkpoint (C1 fine-tuned on the open DFL and Metrica games)
models/kpu_c1/           the specialist C1's checkpoint (open PFF data only) and its validation history
src/model_v2/3/5/7/9.py  dead-ball evidence, emission scores, the B2 soft-minimum decoder (decode2.py)
src/bench.py, sim.py     open-data inputs: parsing (dfl.py, metrica_sample.py), simulated degradation, features
src/mlfeat.py, feat_*.py, bfree.py   team-level features
src/atd.py, broadcast.py, features.py   provided-data parsing, broadcast alignment, per-frame features
docs/TECHNICAL_DOCUMENT.md   the technical document
```

## Data and licences

- No match data, and nothing derived from it at frame level, is in this repository. `data/`, `cache/` and
  `submissions/` are gitignored.
- Open data is fetched from its sources by the scripts above and is not redistributed:
  - IDSSE / DFL open data: Bassek et al. 2025, CC BY 4.0, figshare article 28196177.
  - Metrica Sports sample data: github.com/metrica-sports/sample-data, credited as its README asks.
  - SkillCorner open data (MIT): the dead-ball models and the stacker.
- The BiGRU checkpoints in `models/bigru_dfl_pff_v1/` were trained on the open DFL data and on PFF FC / Gradient Sports
  FIFA World Cup 2022 tracking, obtained through the provider's access form, the specialist in `models/kpu_c1/` on
  that PFF data alone (tracking and its human-labelled events), and `models/kpu_j1/` is that specialist fine-tuned on
  the open DFL and Metrica games, and the ball-free channel in `models/ballfree_v1/` on that PFF data alone; they contain no match data, and the PFF data itself is
  not needed to reproduce the submission.
