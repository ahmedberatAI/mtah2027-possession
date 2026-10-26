# BiGRU ensemble: external evaluation

Fixed equal probability mean of seeds 20261025/26/27; no new training or tuning.
v7 is the exact artifact used in the BAR-ATH comparison, not the older gate reference.
Same test inputs, label conventions, fixed learned-D and B2 for both.
Metrica: 3 matches, 3 noise seeds averaged within each match. SkillCorner: 20 matches.
Paired match bootstrap 10000 draws; Metrica intervals are fragile with only three matches.

| Test | D | v7 | Ensemble | Delta | Delta 95% CI | Wins / matches | Separate seed score mean |
|---|---|---:|---:|---:|---|---|---:|
| metrica@v3/ctrl | learned | 0.814711 | 0.807148 | -0.007563 | [-0.021047, +0.006763] | 1/3 | 0.802442 |
| metrica@v3/ctrl | oracle | 0.871499 | 0.863521 | -0.007978 | [-0.023927, +0.007061] | 1/3 | 0.858131 |
| metrica@v3/events | learned | 0.615542 | 0.645220 | +0.029678 | [-0.005678, +0.049899] | 2/3 | 0.646306 |
| metrica@v3/events | oracle | 0.659524 | 0.689024 | +0.029500 | [-0.005578, +0.047218] | 2/3 | 0.691036 |
| sc@v3/pp | learned | 0.582907 | 0.633614 | +0.050707 | [+0.043563, +0.058556] | 20/20 | 0.630812 |
| sc@v3/pp | oracle | 0.740998 | 0.788058 | +0.047060 | [+0.039473, +0.055903] | 20/20 | 0.784107 |
| sc@v3/phase | learned | 0.581579 | 0.622135 | +0.040556 | [+0.032884, +0.048931] | 20/20 | 0.619456 |
| sc@v3/phase | oracle | 0.745655 | 0.781473 | +0.035819 | [+0.026894, +0.045502] | 20/20 | 0.777385 |

Oracle-D uses true dead-ball labels and is diagnostic, not a deployable result.
The external protocol uses learned-D because BAR-ATH camera-specific rule-D is unavailable here.
These external sets have been used in prior experiments; this is not a new untouched holdout.
The ensemble is fixed; match-bootstrap intervals do not include training-data or model-selection uncertainty.
Label conventions within a source are alternative views of the same matches, not independent datasets.
No inference about the private full-match BAR-ATH score is possible.

Reproduce: `.venv/Scripts/python.exe src/exp_external_ensemble.py`
