# v10's team classifier, seeds 1 and 2 (candidate v22, round 88)

`models/v10_team.txt` with two more LightGBM seeds: DFL native labels + the Metrica sample games' controlled labels (weight 2 per game), the ball-free team features, all degraded with `sim.V3` (`src/exp_combo.py`, seeds 1 and 2). The shipped `models/v10_team.txt` gives exactly the research seed-0 model's predictions on BAR-ATH's halves, so these are its siblings. Open data only; they never saw BAR-ATH.
