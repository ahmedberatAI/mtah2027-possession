# Short-possession specialist J1 (round 37, `src/short_specialist.py` `finetune`)

The specialist of `models/kpu_c1/` fine-tuned on the ten open games that follow the challenge's controlled-possession
reading: the seven IDSSE / DFL matches (event-derived controlled labels) and the three Metrica Sports sample games.
Their tracking is degraded as in `sim.V3`. Metrica's sample ball sits on its owner, so its windows use `bench` variant
`v3g` (ball noise 0.8 m), which puts the ball as far from the players as an automatic tracker's.

- Fine-tuning: 2,000 steps at learning rate 3e-4; every batch is 32 windows of those games and 32 of C1's own PFF
  windows; the touch head has no targets on the ten games and keeps learning from the PFF windows only.
- `kpu_J1_s20261102.pt`: started from `models/kpu_c1/kpu_C1_s20261102.pt`, trained on all ten games. It contains no
  match data; it never saw BAR-ATH.
- Network, inputs and outputs are C1's (`models/kpu_c1/README.md`).

`src/model_v17.py` uses it in C1's three places: 0.25 of its live-normalised P(home control) in the team probability,
candidate and touch features of the short-possession verifier and its own insert candidates
(`models/short_verifier_{ins,del}_j1m25s.txt`).

- `kpu_J1_s20261103.pt`, `kpu_J1_s20261104.pt` (candidate v22, round 88): the same fine-tuning from C1's seeds 20261103 and 20261104, all ten games. They contain no match data; they never saw BAR-ATH.
