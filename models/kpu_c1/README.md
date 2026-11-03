# Short-possession specialist C1 (round 24, `src/short_specialist.py`)

A ball-centred tracking model of who controls the ball and when control changes, trained **only on the open PFF FC
World Cup 2022 data** (64 matches, 56 for training and 8 for early stopping; every half in two degraded copies,
`sim.V3` and `sim.V4`). Its labels are PFF's possessions in the challenge's controlled-possession reading
(`src/gradient.py` `controlled_reading`): a spell that directly follows the other team's spell counts only if the new
team shows control (a completed pass, a shot, or any of its events at least 0.5 s later). It never saw DFL, Metrica,
SkillCorner or BAR-ATH.

- Input per frame (25 fps) and side: ball state, the five players of each team nearest to the ball, context.
- Model: shared frame MLP + bidirectional GRU over +-2 s around 2 s cores, symmetric over the two sides.
- Outputs per frame (9): P(home control), P(away control), P(dead), gains / loses control per team, touches per team.
- `kpu_C1_s20261102.pt`: seed 20261102, 12,000 steps (Colab T4); `kpu_C1_s20261102.json`: its validation history.

`src/model_v15.py` mixes its live-normalised P(home control) into the team probability with weight 0.25 and gives the
short-possession verifier its candidate and touch features (`models/short_verifier_{ins,del}_c1m25s.txt`).
