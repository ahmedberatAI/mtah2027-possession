# Stacker of the team channels (rounds 61, 65-68, `src/stack_model.py`, research code `src/exp_stack.py`)

A small sequence model on the channels' outputs: per team at 5 Hz the logits of v10, the shared BiGRU, J1, their blend
and the ball-free channel, J1's gain / lose / touch heads, each team's distance to the ball; common to both the alive
evidence, J1's dead logit, ball speed, whether the ball is seen. A convolution and a bidirectional GRU give the emissions
of a path model with the states home control, home free, away control, away free, dead; a move to the other team enters
only its control state, and free is latent. Trained on the likelihood of the whole path.

- Training data: the channels' held-out outputs on the open games only: 7 DFL matches (event-derived controlled labels),
  3 Metrica sample games (controlled labels, ball moved off its owner) and 20 SkillCorner matches (event-derived
  controlled labels), `sim.V6`. 2,000 steps of 60 s windows, equal provider weight, with random channel scales, dropped
  channels and a shift of the alive evidence, so that it does not lean on their exact calibration.
- `stack_s0.pt`: trained on all thirty games with seed index 0 of the channels. It contains no match data and has never
  seen BAR-ATH.
