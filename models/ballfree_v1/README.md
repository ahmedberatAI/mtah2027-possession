# Ball-free team channel (rounds 66, 68, `src/exp_ballfree.py`)

A channel that reads only the players, never the ball: per step (5 Hz) up to 14 players of each team as a set (position,
velocity, goalkeeper flag, nearest opponent and closing speed), one attention layer, a two-layer bidirectional GRU over
60 s windows. Outputs per step: home / away control and dead ball; an auxiliary head guesses where the ball is.

- Training data: only the open PFF FC World Cup 2022 data, under the challenge's controlled-possession reading
  (`gradient.controlled_reading`), degraded with `sim.V3`. It has never seen DFL, Metrica, SkillCorner, SoccerTrack or BAR-ATH.
- `bf_s<seed>.pt`: three seeds (20261106, 20261107, 20261108), batch 32, 12,000 steps; each file is that run's lowest
  validation loss on eight held-out PFF matches. The files contain no match data.

`src/model_v21.py` uses the mean of the three seeds as an input of the stacker (`models/stack_v1/`).
