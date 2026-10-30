# More than a Hack 2027 — possession labelling (BAR-ATH)

Per-frame `H` / `A` / `D` labels for the BAR-ATH match of More than a Hack 2027 (Phase 1), produced end to end from
the provided data and open data:

- **Dead ball:** interpretable evidence from the automatic tracking and the time-aligned broadcast recording.
- **Team in possession:** a LightGBM classifier on team-level tracking features, trained on open tracking (IDSSE /
  DFL, Metrica Sports sample games) degraded to look like automatic single-camera tracking, averaged in equal parts
  with a bidirectional GRU trained on open DFL and PFF World Cup 2022 tracking (`models/bigru_dfl_pff_v1`).
- **Decoding:** a duration-aware Viterbi decoder per half.
- **Short possessions:** a verifier re-examines the decoder's short decisions with two LightGBM classifiers trained on
  open data; its inputs include the blend's and the GRU's margins.

The approach, the validation protocol and the results are in **[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md)**.

## Reproduce the submission

```bash
# Python 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into ./data/{tactical,broadcast}  (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/make_submission.py           # -> submissions/candidate_v14.csv (a few minutes)
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v14.csv

# optional: retrain the team classifier from the open data (rewrites models/v10_team.txt, about 7 min on 24 cores)
.venv/Scripts/python.exe src/download_idsse.py            # open DFL / IDSSE data, CC BY 4.0, ~2.6 GB
.venv/Scripts/python.exe src/metrica_sample.py download   # Metrica Sports sample games, pinned commit, ~185 MB
MTAH_RETRAIN=1 .venv/Scripts/python.exe src/make_submission.py
```

- On Linux / macOS, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.
- `models/v10_team.txt` is the team classifier and `models/short_verifier_{ins,del}_g_blend50.txt` the verifier, all
  trained only on open data, so the default run needs only the provided data. On its first run it rebuilds the BiGRU's BAR-ATH
  probabilities on the CPU (`src/reproduce_shared_bigru.py`, checked against `models/bigru_dfl_pff_v1/manifest.json`).
- The run is deterministic: fixed simulation seeds, LightGBM deterministic mode with a fixed seed and thread count,
  and no GPU. Both the default run and a retrain of the team classifier from an empty `cache/` write the submitted
  file, MD5 `5e7363fc83c6f56a923033894a28d3a1`. `--model model_v13 --out submissions/candidate_v13.csv` writes the
  previous submission (MD5 `ed8ad739207f367017bb8d8c8204c82a`).

## Layout

```
src/make_submission.py   end to end: provided data -> submissions/candidate_v14.csv
src/model_v14.py         the submitted model: 0.5 v10 + 0.5 BiGRU blend + the short-possession verifier
src/model_v13.py         the previous submission: model_v10 + the verifier (src/short_verifier.py)
src/model_v10.py         team classifier training + prediction, decoder settings
models/v10_team.txt      the team classifier (LightGBM, trained on the open data only)
models/short_verifier_*  the verifier's classifiers (LightGBM, open data only; *_blend50 = v14's)
models/bigru_dfl_pff_v1/ the shared BiGRU checkpoints; src/reproduce_shared_bigru.py and src/seq_pff.py run them
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
  FIFA World Cup 2022 tracking, obtained through the provider's access form; they contain no match data, and the PFF
  data itself is not needed to reproduce the submission.
