# Wide path models (rounds 94 and 96, `src/model_v23.py`, research code `src/exp_stack.py`)

The same stacker as `models/stack_v1/` (inputs, path model with the states home control, home free, away control,
away free, dead; training on the likelihood of the whole path; 2,000 steps of 60 s windows; equal provider weight;
random channel scales, dropped channels and a shift of the alive evidence), with a width of 128 instead of 64 in the
convolutions and the bidirectional GRU.

- Training data: the channels' held-out outputs on the open games only: 7 DFL matches (event-derived controlled labels),
  3 Metrica sample games (controlled labels, ball moved off its owner) and 20 SkillCorner matches (event-derived
  controlled labels), `sim.V6`.
- `stack_w128_s0.pt`, `stack_w128_s1.pt`, `stack_w128_s2.pt`: trained on all thirty games with seed indices 0, 1, 2 of
  the channels (their v10 and J1 seeds), with the same random seeds as `stack_v1`'s. `src/model_v23.py` draws the
  decision's samples from these three; v22's reference path still comes from `stack_v1`.
- Held out by match, the wider models give the whole path a higher likelihood on every open set and every seed, and
  the decision on their samples scores higher in the lab, in the target's condition (channels and dead-ball models that
  never saw the provider) and on SoccerTrack, which no model saw (rounds 94 and 96 in `docs/RAPOR.md`). They contain
  no match data and have never seen BAR-ATH.
