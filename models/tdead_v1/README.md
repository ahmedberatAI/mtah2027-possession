# Sequence dead-ball model TD6n (rounds 46, 49, 51, `src/exp_tdead.py`)

A small convolution + two-layer bidirectional GRU that reads the tracking features at 5 Hz and predicts the dead ball, with two auxiliary heads for stops and restarts during training.

- Inputs: the 59 `base` features of `src/bench.py`, averaged over 5 frames. Each feature is scaled by the half's own median and interquartile range (`half_norm`), so a match's absolute tempo does not move the model. Label-free and per half.
- Training: seven IDSSE / DFL matches, three Metrica Sports sample games (ball noise 0.8 m), twenty SkillCorner matches; equal provider weight in every batch. Tracking degraded by `sim.V6`, whose ball mask around stops and restarts follows aggregate, label-free shares (see `src/sim.py`). 4,000 steps.
- `tdead_v6n_s<seed>.pt`: the all-data model of each of three seeds; `src/model_v20.py` averages their P(dead) on the CPU. Each file holds the network's weights, and the mean and scale of the training features. No match data; none saw BAR-ATH.

Reproduce: `TD_V6=1 TD_NORM=1 TD_OUT=tdead6n python -I src/exp_tdead.py train <seed>`, then `python -I src/dead_v6_fit.py td`.
