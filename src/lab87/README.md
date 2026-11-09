# Research harness of rounds 87–96 (`docs/RAPOR.md`)

Research code only; not part of the submission or of the public export. Every script reads the cached held-out outputs
under `cache/lab87/` (open data only) and prints aggregate scores, counts and ratios, never frame-level output. Until
9 November 2026 these scripts lived in a session scratchpad (`scratchpad lit87/...` in `docs/RAPOR.md`).

- `lab.py`: packs every open set once (truth of each reading, dead-ball evidence, the channels per seed) and scores
  arms (`score` = spellF1 per match).
- `foldnets.py`: trains and keeps the stacker's fold nets (`folds`, held out by match; `lopo`; `lnets`, provider held
  out) and the all-games nets; `STACK_OPTS` sets the width.
- `r88.py`, `r88f.py`: the Bayes decision on the fold nets' samples; the held-out path likelihood (NLL per step).
- `r92.py`, `r93.py`: error budgets (oracles) of v22.
- `r94.py`, `r94b.py`: round 94 (the wider stacker W3, B6), in-provider and provider held out.
- `r95op.py`: the target-condition yardstick (OPs: channels and dead-ball models trained without the scored provider,
  the stacker in-provider; also OPd, OPc, LP, OP).
- `r96.py`: round 96 (Hop and W3 under the lab + OPs rule); `v23tab.py`: v23 against v22 per context from its cells.
- `tgt94.py`, `cmp_target.py`, `cmp_two.py`: label-free checks on the target (aggregates only).
- `emit_st.py`, `crfk.py`: SoccerTrack emissions and the CRF baseline.
- `rapor.py`: appends rows and notes to `docs/RAPOR.md`, guarded by unique markers.

Run from the repository root, e.g. `.venv/Scripts/python.exe -I src/lab87/r96.py eval`.
