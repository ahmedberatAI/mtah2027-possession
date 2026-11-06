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

## Ideas, experiments and results

Everything we tried and what it gave lives in one document: **[`docs/RAPOR.md`](docs/RAPOR.md)** (Turkish). It
opens with a summary of the current model and scores, then the permanent rules, the one-line-per-experiment log, the
idea pool with the round-by-round notes, and the early topic reports as appendices. Every new experiment is added
there (a row at the end of the log table in §3, details under the latest round in §4).

Fikirler, denemeler ve sonuçlar tek bir belgede: **[`docs/RAPOR.md`](docs/RAPOR.md)**. Her yeni deneme oraya eklenir.

## Submission

- **The file:** `submissions/candidate_v21.csv` (model_v21 = model_v20 + a channel that never sees the ball + a path-model stacker on the channels' outputs; model_v20 = the specialist J1 at weight 0.5 in the team probability and dead-ball
  evidence from open-data rules + a multi-source dead model + a sequence dead model with per-half normalised input + view terms,
  all on `sim.V6`'s masks, no verifier; model_v19 = model_v17's team side with dead-ball evidence free of the
  public window: open-data rules + a multi-source dead model + view terms; model_v18 = model_v17 with rules + broadcast + half
  the learned dead model; model_v17 = model_v16 with the short-possession specialist J1
  in C1's place; model_v16 = model_v15 + the dead-ball boundary timing of the learned dead model; model_v15 = v14's v10 /
  shared-BiGRU blend mixed with the short-possession specialist C1 + the short-possession verifier retrained on that
  base, no whistle), MD5 `d909212952998c76568b09c17b76b5d5`.
  `python src/make_submission.py` regenerates it; `--model model_v19 --out submissions/candidate_v19.csv` still writes
  v19 (MD5 `45096061ebdc3aac6e2087d8a047246a`), `--model model_v18 --out submissions/candidate_v18.csv`
  v18 (MD5 `b6101389cd877edec44bbe64cd00c4b3`), `--model model_v17 --out submissions/candidate_v17.csv`
  v17 (MD5 `55dfc08216c344f7077d4fac88c51a86`), `--model model_v16 --out submissions/candidate_v16.csv`
  v16 (MD5 `c89208a3a034ee73bb41f3fb28390f87`), `--model model_v15 --out submissions/candidate_v15.csv`
  v15 (MD5 `6081216bcf484049d8e6e84b27fc85dd`), `--model model_v14 --out
  submissions/candidate_v14.csv` v14 (MD5 `5e7363fc83c6f56a923033894a28d3a1`), `--model model_v13 --out
  submissions/candidate_v13.csv` v13 (MD5 `ed8ad739207f367017bb8d8c8204c82a`) and `--model model_v10 --out
  submissions/candidate_v10.csv` v10 (MD5 `ca20c5b657a395d2f7c0e310cdaabd64`).
- **Model file:** the team classifier is trained on open data only and kept in `models/v10_team.txt`, so the default
  run needs only the provided data (about a minute). `MTAH_RETRAIN=1` retrains it from the open data (about 7 min)
  byte for byte. v16's dead-ball classifier (all-DFL, open data) is kept the same way in `models/dead_base_v3.txt`.
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
.venv/Scripts/python.exe src/make_submission.py           # the submission: model_v21 -> submissions/candidate_v21.csv
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v21.csv
```

**The submission is `model_v21`** (`submissions/candidate_v21.csv`, MD5 `10ee3c08e6a5d5de1d3951652eca26f2`; rounds 61 and 65–69 in `docs/RAPOR.md`). v20's channels and dead-ball evidence, plus a channel that reads only the players (`src/exp_ballfree.py`, `models/ballfree_v1/`, open PFF data only, mean of three seeds) and a path-model stacker on the channels' outputs (`src/stack_model.py`, `models/stack_v1/`, open games only). Team: logit mix 0.5 stacker + 0.5 v20's smoothed blend. Alive evidence: 0.5 v20's + 0.5 the stacker's. Decoder: pen_team 6, B2 c 24. Held out, 3 seeds, v20 -> v21: DFL evctrl 0.8078 -> 0.8217, DFL tracking-derived 0.7886 -> 0.7997, Metrica ctrl 0.8694 -> 0.8930, SkillCorner evctrl / phases / player possessions 0.7604 / 0.7114 / 0.7020 -> 0.7848 / 0.7351 / 0.7210; worst-case regret on the target convention's contexts +0.0236 (90% [+0.0225, +0.0251]). SoccerTrack v2, which no model saw: 0.7936 / 0.7255 -> 0.8191 / 0.7589 (8/8). `--model model_v20 --out ...` writes v20.

**The previous submission is `model_v20`** (`submissions/candidate_v20.csv`, MD5 `d909212952998c76568b09c17b76b5d5`; rounds 45, 46, 49 and 51–55 in `docs/RAPOR.md`).
Team: J1's weight rises from 0.25 to 0.5 once Metrica's ball is de-glued for every channel. Dead ball: 0.25 x open-data rules + 0.375 x the
multi-source dead model + 0.375 x a sequence model (`src/exp_tdead.py`, `models/tdead_v1/`: convolution + bidirectional GRU at 5 Hz, every input
feature scaled by the half's own median and spread, three seeds) + view terms, all refitted on `sim.V6`, whose ball no longer reappears exactly on
the true restart frame (`src/sim.py`). No verifier, no boundary timing. Held out, V6 masks, Metrica de-glued everywhere, v19 -> v20, before v19's
verifier: DFL evctrl 0.796 -> 0.808, DFL tracking-derived 0.776 -> 0.789, Metrica ctrl 0.854 -> 0.869, SkillCorner evctrl / phases / player possessions
0.737 / 0.692 / 0.684 -> 0.760 / 0.711 / 0.702; worst-case regret on the target convention's contexts +0.015 (90% [+0.010, +0.019]). Sets no model
saw: SoccerTrack v2 0.691 -> 0.794 (provider's status 0.631 -> 0.725), raw PFF, dead ball only, 0.533 -> 0.548. The gain cannot be measured on BAR-ATH
itself; v20 differs from v19 on 3.97% of its frames.

**`model_v19`: `model_v17`'s team side with dead-ball evidence free of the public window**
(`submissions/candidate_v19.csv`, MD5 `45096061ebdc3aac6e2087d8a047246a`; rounds 43 and 44 in `docs/RAPOR.md`). The official rules say the public window
is not a tuning set, and the dead-ball rules up to v18 were written looking at it. In v19 the decoder's alive log-odds are 0.25 x
rules whose constants all come from the open sets (`models/dead_rules_open_v1.json`, `src/dead_rules_fit.py`) + 0.75 x a dead
model trained on DFL, de-glued Metrica and SkillCorner with equal provider weight (`models/dead_multi_v1.txt`) + the view terms,
with weights from SkillCorner's real flag. The verifier stays. Held out, before the verifier, v17's recipe / v18 / v19: DFL evctrl
0.682 / 0.784 / 0.794, Metrica ctrl 0.768 / 0.837 / 0.855, SkillCorner evctrl 0.642 / 0.702 / 0.731, PFF (independent) 0.535 / 0.523 /
0.533. Worst-case regret on the target convention's contexts +0.018 against v18. The gain cannot be measured on BAR-ATH itself; v19
differs from v18 on 3.95% of its frames.

**`model_v18`: `model_v17` with fused dead-ball evidence** (`submissions/candidate_v18.csv`, MD5
`b6101389cd877edec44bbe64cd00c4b3`; rounds 40 and 41 in `docs/RAPOR.md`). The decoder's alive log-odds are the rules + the aligned broadcast + 0.5 x
the alive log-odds of the learned dead model (`models/dead_base_v3.txt`); the boundary timing step is gone. The open sets'
rules used to lack the broadcast, so they understated BAR-ATH's; with SkillCorner's real view flag transplanted onto DFL and
Metrica (`src/exp_deadmix.py`) the rule formula is the same on every set. Held out, before the verifier, against v17's
recipe: DFL evctrl 0.685 -> 0.782, DFL tracking-derived 0.665 -> 0.761, Metrica ctrl 0.768 -> 0.837, SkillCorner evctrl /
phases / player possessions 0.642 / 0.605 / 0.599 -> 0.702 / 0.659 / 0.653 (every match better); worst-case regret on the
target convention's contexts +0.097. On BAR-ATH 6.95% of the frames change, and the structure of the changes lies in the
open sets' range. The gain cannot be measured on BAR-ATH itself.

**Its base, `model_v17`: `model_v16` with the short-possession specialist J1 in C1's place**
(`submissions/candidate_v17.csv`, MD5 `55dfc08216c344f7077d4fac88c51a86`; round 37 in `docs/RAPOR.md`). J1 (`models/kpu_j1`) is C1
fine-tuned on the ten open games in the challenge's controlled-possession reading (DFL event-derived, Metrica sample
games; `src/short_specialist.py` `finetune`). Metrica's sample ball sits on its owner, so its windows use 0.8 m of extra
ball noise, which puts the ball as far from the players as an automatic tracker's; half of C1's Metrica gain turned out
to come from that glued ball. Held out, against C1 on the same input, learned / true dead ball: DFL evctrl +0.0020 /
+0.0017, Metrica ctrl +0.0059 / +0.0051; worst-case regret on those contexts +0.0059 (90% [+0.0027, +0.0070]), worst
guard (DFL tracking-derived) −0.0050. The verifier is retrained on J1's outputs
(`models/short_verifier_{ins,del}_j1m25s.txt`). On BAR-ATH v17 differs from v16 on 0.43% of the frames.

**`model_v16`: `model_v15` with its dead-ball boundaries timed by the learned dead model**
(`submissions/candidate_v16.csv`, MD5 `c89208a3a034ee73bb41f3fb28390f87`; round 29 in `docs/RAPOR.md`). v15's dead
stretches (v3's rules + the aligned broadcast) stay; each stop and restart moves onto the nearest boundary of the same
kind in the learned dead model's own decode, within ±2 s and never across another boundary (`src/dead_timing.py`; the
all-DFL dead-ball classifier is kept in `models/dead_base_v3.txt`). With BAR-ATH-like dead ball on the open sets the
rules' boundaries are coarse (median error ~1 s) and the learned model's sharp: DFL evctrl +0.016, Metrica ctrl +0.005,
SkillCorner +0.003, and SoccerTrack v2, which was not used to design the step, +0.005. On BAR-ATH it moves 110 of 233
boundaries (1.07% of the frames).

**Its base, `model_v15`: v14's blend mixed with the short-possession specialist C1, plus the verifier
retrained on that base** (`submissions/candidate_v15.csv`, MD5 `6081216bcf484049d8e6e84b27fc85dd`; rounds 23–25 in
`docs/RAPOR.md`). C1 (`src/short_specialist.py`, `models/kpu_c1`) is a ball-centred tracking model trained only on open
PFF data, with PFF's possessions in the challenge's controlled-possession reading; 0.25 of its control probability
enters the team probability before smoothing and decoding, and its candidate / touch features and own insert candidates
feed the verifier (`models/short_verifier_{ins,del}_c1m25s.txt`), whose DFL and SkillCorner rows are labelled in the same
reading. On the target-convention contexts (Metrica ctrl, DFL evctrl) the worst-case regret falls by 0.0126 against v14
(90% [+0.0048, +0.0238]); no guard context loses more than 0.001.

The previous submission, **`model_v14`: an equal blend of `model_v10`'s team probability and the shared BiGRU's, plus
the short-possession verifier retrained on the blend's own decisions** (`submissions/candidate_v14.csv`, MD5
`5e7363fc83c6f56a923033894a28d3a1`; rounds 17 / 17b in `docs/RAPOR.md`). The BiGRU (`models/bigru_dfl_pff_v1`, open
DFL + PFF data) runs on the CPU (`src/reproduce_shared_bigru.py`, needs torch). The verifier (`src/short_verifier.py`,
`models/short_verifier_{ins,del}_g_blend50.txt`) re-examines the short decisions, the flips the decoder suppressed and
the decoded possessions under 4 s. A leave-one-DFL-match-out copy of the BiGRU (trained on Colab,
`src/exp_bigru_lomo.py`) let DFL judge the blend for the first time: the weight .5 is the minimax-regret point of the
grid fixed in round 13 (`src/exp_blend_weight.py`), and on the ten external contexts the worst-case regret falls from
0.0165 (v13, its verifier seeing a held-out BiGRU margin on DFL as on BAR-ATH) to 0.0039 (SkillCorner +0.010…+0.014,
Metrica +0.005…+0.006, DFL +0.002…+0.008). `model_v13` (round 15: v10 + the verifier) and the model history below are
kept for reference.

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

The three original BiGRU checkpoints are shared in
[`models/bigru_dfl_pff_v1/`](models/bigru_dfl_pff_v1/README.md), with SHA256
checks and a portable inference-only command:
`python src/reproduce_shared_bigru.py --device cuda --score`.
It reproduced the frozen BAR-ATH ensemble CSV byte-for-byte on 2026-10-27.
The participant's BAR-ATH assets are required; DFL/PFF training data are not.

- The hand-labelling tools (`tools/labeler.html`, `tools/transitions_to_gt.py`) were removed on 2026-10-29: the
  current rules forbid hand labels on BAR-ATH for any purpose, and they were not used. They remain in git history.
- `colab/ball_detector_colab.py` — self-supervised ball detector (ATD balls projected through the
  homography as pseudo-labels) for a Colab GPU.
