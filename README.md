# More than a Hack 2027 — possession labelling (working repo)

> **Rules update, 2026-10-24.** The current official `RULES.md` prohibits human
> labels on the provided BAR-ATH game for **any** purpose, including training
> and validation. Earlier notes and the labeler description below are historical
> and superseded. Use the labeler only for other games. The public five minutes
> are a pipeline sanity check, never a tuning set.

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
  bench.py, paired.py     shared CPU test bench for model ideas (DFL / Metrica / BAR-ATH aggregates)
  feat_*.py, decode2.py   idea plugins (features, generalised decoder); model_v7.py candidate
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

Candidate `model_v7` (`--model model_v7 --out submissions/candidate_v7.csv`): same alive/dead evidence,
team model trained on DFL degraded with `sim.V3` (smooth tracking noise) with pressure-aware control,
set-piece formation, team shape and movement features, decoded with a soft minimum possession length
(`src/decode2.py`). Ideas, the shared CPU test bench (`src/bench.py`) and every result: `docs/FIKIRLER.md`,
`docs/DENEMELER.md`.

## PFF 64-match experiment

64 World Cup matches were aligned to 25 Hz, converted to conservative controlled
possession labels, and used to train three models with the fixed v7 features and
B2 decoder. The primary DFL+PFF model gains +0.0084 spellF1 on SkillCorner-pp and
is flat on Metrica-ctrl; v7 remains unchanged. Raw tracking only is used: the
provider's event-derived smoothed ball is excluded. Results, limitations and
reproduction commands: [PFF report](reports/pff/SONUC.md),
[full comparison](reports/pff/RESULTS.md), [method](reports/pff/METODOLOJI.md).
Dependencies are pinned in `requirements-gates.lock.txt`.

## Balanced TCN / BiGRU experiment

TCN and BiGRU use exact 50:50 DFL/PFF source sampling. Configuration selection
uses held-out training matches (2 DFL + 12 PFF), followed by full 7+64-match
refits. SkillCorner-pp spellF1 is 0.6265 / 0.6284 versus 0.5913 for PFF LightGBM;
Metrica-controlled remains near v7. Both are saved as separate candidates.
[Results](reports/sequence_pff/SONUC.md), [protocol and commands](reports/sequence_pff/METODOLOJI.md),
[paired comparisons](reports/sequence_pff/RESULTS.md). Code: `src/seq_pff.py`,
`src/exp_seq_pff.py`; final checkpoints: `cache/sequence_pff_v1/*_final.pt`.

## Three-seed source-balance control

Three fresh seeds repeat the frozen neural settings and train paired weighted /
unweighted LightGBM controls. SkillCorner-pp means: normal LGB 0.5911, balanced
LGB 0.5930, TCN 0.6243, BiGRU 0.6308. The neural gain largely persists after
source balancing; Metrica-controlled instead favors balanced LGB (0.8173 vs
TCN 0.8046 / BiGRU 0.8024). This qualifies the earlier single-seed result.
[Report](reports/seed_balance/SONUC.md), [all seed scores and intervals](reports/seed_balance/RESULTS.md),
[protocol](reports/seed_balance/METODOLOJI.md). Driver: `src/exp_seed_balance.py`;
12 model files are in `cache/seed_balance_v1/`. No best-seed selection or ensemble.

## BAR-ATH public-window comparison

With the actual BAR-ATH tracking and the fixed v7 rule-based dead-ball evidence
and B2 decoder, public spellF1 is **v7 0.6519**, **balanced LightGBM 0.6667**,
and **BiGRU 0.7770 ± 0.0080** across three training seeds. All seven full-match
prediction files were frozen before public scoring; the official evaluator
confirmed the scores. The public window has only 17 ground-truth possessions;
this does not establish the private full-match score.
[Report](reports/barath_public/SONUC.md), [scores and commands](reports/barath_public/RESULTS.md).
Driver: `src/exp_barath_public.py`; local CSVs: `submissions/barath_public_v1/`.

Equal-weight BiGRU probability ensemble: public spellF1 **0.7724** (+0.1206
over v7; below the separate-seed score mean of 0.7770). The validated full-match
CSV is `submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv` (199,319 rows).
[Ensemble report](reports/barath_bigru_ensemble/SONUC.md); driver:
`src/exp_barath_ensemble.py`. Prepared locally, not uploaded to the competition.

## Equal-weight ensemble on external matches

The same three BiGRU checkpoints and fixed 1/3 probability weights were tested
against the actual v7 artifact under identical external learned-D/B2 conditions.
SkillCorner-pp: **0.5829 → 0.6336**, phase: **0.5816 → 0.6221**, winning all
20 matches under both label conventions. Metrica-controlled: **0.8147 → 0.8071**;
Metrica-events: **0.6155 → 0.6452**. This supports a SkillCorner gain, not universal
superiority. The ensemble adds only about +0.0028 over the individual seed score
mean on SkillCorner-pp. [Report](reports/external_bigru_ensemble/SONUC.md),
[paired intervals](reports/external_bigru_ensemble/RESULTS.md).
Driver: `src/exp_external_ensemble.py`; no new training or parameter selection.

## Internal-selected BiGRU / balanced-LightGBM blend

Six new fits exclude the fixed 2 DFL + 12 PFF validation matches. Five probability
blend weights were selected only on that internal split. The winner is **100%
BiGRU / 0% LGB** (internal spellF1 0.6947 versus 0.6828 for 75/25); both source
means prefer pure BiGRU. The selected external result therefore equals the existing
ensemble; this experiment does not justify replacing it with a mixture.
Error diagnostics show extra fragmentation/unmatched predictions on Metrica-ctrl,
but fewer missed and unmatched possessions on SkillCorner.
[Report](reports/hybrid/SONUC.md), [error tables](reports/hybrid/HATA_ANALIZI.md),
[protocol](reports/hybrid/METODOLOJI.md). Driver: `src/exp_hybrid.py`.

## Controlled-label DFL BiGRU experiment

DFL native labels were replaced with the repository's tracking-derived `trk`
control proxy; PFF labels and all inputs remained unchanged. Two configurations
were selected only on held-out training matches; the existing 64-width setting
won and was refit with three seeds. Metrica-controlled spellF1 **0.8071 → 0.8176**
and the split proxy **4.91% → 2.07%**, but short possessions were missed more often.
SkillCorner-pp **0.6336 → 0.5778** (loss in 20/20 matches), phase **0.6221 → 0.5977**.
The current ensemble remains the candidate; this label conversion is not adopted.
[Report](reports/controlled_dfl_bigru/SONUC.md),
[protocol](reports/controlled_dfl_bigru/METODOLOJI.md).
Driver: `src/exp_controlled_dfl.py`; nine fits and 24 external cells audited.

## Tools

- `tools/labeler.html` — local hand-labelling of H/A/D transitions for other games only.
  The current rules prohibit hand labels on BAR-ATH for training or validation.
- `tools/transitions_to_gt.py` — marks -> 199,319-row `frame,label` file usable as `--gt` by the
  official scorer. Keep it in `cache/` (frame-level derived data).
- `colab/ball_detector_colab.py` — self-supervised ball detector (ATD balls projected through the
  homography as pseudo-labels) for a Colab GPU.
