# Hybrid — frozen internal selection and external evaluation

Selected weight: BiGRU 1.0; balanced LGB 0.0.
Three seeds averaged within each family. All settings fixed before external evaluation.

| Internal BiGRU weight | DFL native | PFF controlled | Equal-source mean |
|---|---:|---:|---:|
| 0.00 | 0.582279 | 0.694359 | 0.638319 |
| 0.25 | 0.594613 | 0.700810 | 0.647712 |
| 0.50 | 0.630575 | 0.713417 | 0.671996 |
| 0.75 | 0.640807 | 0.724785 | 0.682796 |
| 1.00 | 0.650774 | 0.738708 | 0.694741 |

| Test | D | v7 | BiGRU ensemble | Balanced LGB ensemble | Selected hybrid | Delta vs BiGRU (95% CI) |
|---|---|---:|---:|---:|---:|---|
| metrica@v3/ctrl | learned | 0.814711 | 0.807148 | 0.815985 | 0.807148 | +0.000000 [+0.000000, +0.000000] |
| metrica@v3/ctrl | oracle | 0.871499 | 0.863521 | 0.873510 | 0.863521 | +0.000000 [+0.000000, +0.000000] |
| metrica@v3/events | learned | 0.615542 | 0.645220 | 0.616039 | 0.645220 | +0.000000 [+0.000000, +0.000000] |
| metrica@v3/events | oracle | 0.659524 | 0.689024 | 0.661931 | 0.689024 | +0.000000 [+0.000000, +0.000000] |
| sc@v3/pp | learned | 0.582907 | 0.633614 | 0.592648 | 0.633614 | +0.000000 [+0.000000, +0.000000] |
| sc@v3/pp | oracle | 0.740998 | 0.788058 | 0.748425 | 0.788058 | +0.000000 [+0.000000, +0.000000] |
| sc@v3/phase | learned | 0.581579 | 0.622135 | 0.591667 | 0.622135 | +0.000000 [+0.000000, +0.000000] |
| sc@v3/phase | oracle | 0.745655 | 0.781473 | 0.752954 | 0.781473 | +0.000000 [+0.000000, +0.000000] |

Oracle-D is diagnostic. Main results use identical learned-D and fixed B2 for every model.
Metrica has 3 matches, noise draws averaged within match; SkillCorner has 20 matches.
Both internal validation and external benchmarks were used previously; these are not untouched holdouts.
DFL internal selection uses native labels as in the original neural selection, not the external controlled-label proxy.
No BAR-ATH scoring, CSV replacement, threshold search or external weight sweep was performed.
