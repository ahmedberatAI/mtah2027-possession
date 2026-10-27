# More than a Hack 2027 — possession labelling (BAR-ATH)

Per-frame `H` / `A` / `D` labels for the BAR-ATH match of More than a Hack 2027 (Phase 1), produced end to end from
the provided data and open data:

- **Dead ball:** interpretable evidence from the automatic tracking and the time-aligned broadcast recording.
- **Team in possession:** a LightGBM classifier on team-level tracking features. It is trained on open tracking (IDSSE /
  DFL, Metrica Sports sample games) degraded to look like automatic single-camera tracking.
- **Decoding:** a duration-aware Viterbi decoder per half.

The approach, the validation protocol and the results are in **[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md)**.

## Reproduce the submission

```bash
# Python 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into ./data/{tactical,broadcast}  (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/make_submission.py           # -> submissions/candidate_v10.csv (about a minute)
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v10.csv

# optional: retrain the team classifier from the open data (rewrites models/v10_team.txt, about 7 min on 24 cores)
.venv/Scripts/python.exe src/download_idsse.py            # open DFL / IDSSE data, CC BY 4.0, ~2.6 GB
.venv/Scripts/python.exe src/metrica_sample.py download   # Metrica Sports sample games, pinned commit, ~185 MB
MTAH_RETRAIN=1 .venv/Scripts/python.exe src/make_submission.py
```

- On Linux / macOS, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.
- `models/v10_team.txt` is the team classifier, trained only on the open data, so the default run needs only the
  provided data.
- The run is deterministic: fixed simulation seeds, LightGBM deterministic mode with a fixed seed and thread count,
  and no GPU. Both the default run and a retrain from an empty `cache/` write the submitted file, MD5
  `ca20c5b657a395d2f7c0e310cdaabd64`.

## Layout

```
src/make_submission.py   end to end: provided data -> submissions/candidate_v10.csv
src/model_v10.py         the submitted model (team classifier training + prediction, decoder settings)
models/v10_team.txt      the team classifier (LightGBM, trained on the open data only)
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
