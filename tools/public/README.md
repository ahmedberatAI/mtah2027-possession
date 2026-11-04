# More than a Hack 2027 — possession labelling (BAR-ATH)

Per-frame `H` / `A` / `D` labels for the BAR-ATH match of More than a Hack 2027 (Phase 1), produced end to end from
the provided data and open data:

- **Dead ball:** a small rule model on player tempo and ball state, a dead-ball classifier and the time-aligned broadcast
  recording's view; all constants and models come from open data.
- **Team in possession:** a LightGBM classifier on team-level tracking features, trained on open tracking (IDSSE /
  DFL, Metrica Sports sample games) degraded to look like automatic single-camera tracking, averaged in equal parts
  with a bidirectional GRU trained on open DFL and PFF World Cup 2022 tracking (`models/bigru_dfl_pff_v1`).
- **Short-possession specialist:** a ball-centred tracking model trained on open PFF World Cup 2022 data, with
  PFF's possessions in the challenge's controlled-possession reading (`models/kpu_c1`), then fine-tuned on the open
  DFL and Metrica games in the same reading (`models/kpu_j1`); 0.25 of its probability of control enters the team
  probability.
- **Decoding:** a duration-aware Viterbi decoder per half.
- **Short possessions:** a verifier re-examines the decoder's short decisions with two LightGBM classifiers trained on
  open data labelled in the same controlled-possession reading; its inputs include the blend's and the GRU's margins and
  the specialist's control, change and touch outputs, and the specialist proposes candidates of its own.

The approach, the validation protocol and the results are in **[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md)**.

## Reproduce the submission

```bash
# Python 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into ./data/{tactical,broadcast}  (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/make_submission.py           # -> submissions/candidate_v19.csv (a few minutes)
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v19.csv

# optional: retrain the team classifier from the open data (rewrites models/v10_team.txt, about 7 min on 24 cores)
.venv/Scripts/python.exe src/download_idsse.py            # open DFL / IDSSE data, CC BY 4.0, ~2.6 GB
.venv/Scripts/python.exe src/metrica_sample.py download   # Metrica Sports sample games, pinned commit, ~185 MB
MTAH_RETRAIN=1 .venv/Scripts/python.exe src/make_submission.py
```

- On Linux / macOS, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.
- `.gitattributes` keeps the LightGBM model files (`models/*.txt`) byte for byte: LightGBM 4.7 cannot read them with
  the CRLF line endings a Windows checkout would otherwise write (the pipeline also reads them as text).
- `models/v10_team.txt` is the team classifier, `models/kpu_j1/kpu_J1_s20261102.pt` the specialist,
  `models/short_verifier_{ins,del}_j1m25s.txt` the verifier, `models/dead_multi_v1.txt` the dead-ball classifier and
  `models/dead_rules_open_v1.json` the rule constants, all from open data only, so the default run needs
  only the provided data. On its first run it rebuilds the BiGRU's BAR-ATH probabilities on the CPU
  (`src/reproduce_shared_bigru.py`, checked against `models/bigru_dfl_pff_v1/manifest.json`) and runs the specialist on
  the CPU.
- The run is deterministic: fixed simulation seeds, LightGBM deterministic mode with a fixed seed and thread count,
  and no GPU. Both the default run and a retrain of the team classifier from an empty `cache/` write the submitted
  file, MD5 `c89208a3a034ee73bb41f3fb28390f87`. `--model model_v15 --out submissions/candidate_v15.csv` writes the
  previous submission (MD5 `6081216bcf484049d8e6e84b27fc85dd`), `--model model_v14 --out
  submissions/candidate_v14.csv` the one before (MD5 `5e7363fc83c6f56a923033894a28d3a1`) and `--model model_v13
  --out submissions/candidate_v13.csv` v13 (MD5 `ed8ad739207f367017bb8d8c8204c82a`).

## Layout

```
src/make_submission.py   end to end: provided data -> submissions/candidate_v19.csv
src/model_v19.py         the submitted model: v17's team side, dead-ball evidence free of the public window
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
models/v10_team.txt      the team classifier (LightGBM, trained on the open data only)
models/short_verifier_*  the verifier's classifiers (LightGBM, open data only; *_j1m25s = v17's, *_c1m25s = v15's,
                         *_blend50 = v14's)
models/dead_multi_v1.txt the dead-ball classifier (LightGBM, open DFL, Metrica and SkillCorner data, equal provider weight)
models/dead_rules_open_v1.json  the rule constants (open data only)
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
- The BiGRU checkpoints in `models/bigru_dfl_pff_v1/` were trained on the open DFL data and on PFF FC / Gradient Sports
  FIFA World Cup 2022 tracking, obtained through the provider's access form, the specialist in `models/kpu_c1/` on
  that PFF data alone (tracking and its human-labelled events), and `models/kpu_j1/` is that specialist fine-tuned on
  the open DFL and Metrica games; they contain no match data, and the PFF data itself is
  not needed to reproduce the submission.
