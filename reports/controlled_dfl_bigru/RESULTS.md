# Controlled DFL BiGRU results

Selected config: {'name': 'bigru_64', 'kind': 'bigru', 'width': 64, 'drop': 0.15, 'lr': 0.001}.
All ensembles average seeds 20261025/26/27. Same features, PFF labels, balance and decoder.
DFL controlled labels are a tracking-derived proxy, not human ground truth.

| Internal config | DFL controlled | PFF controlled | Equal-source mean |
|---|---:|---:|---:|
| native reference | 0.786149 | 0.738708 | 0.762428 |
| bigru_64 | 0.836235 | 0.696262 | 0.766249 |
| bigru_96 | 0.835674 | 0.693309 | 0.764491 |

| Test | D | Native ensemble | Controlled fixed | Controlled selected | Selected delta [95% CI] | Wins |
|---|---|---:|---:|---:|---|---|
| metrica@v3/ctrl | learned | 0.807148 | 0.817565 | 0.817565 | +0.010418 [-0.006475, +0.025403] | 2/3 |
| metrica@v3/ctrl | oracle | 0.863521 | 0.879388 | 0.879388 | +0.015867 [-0.004099, +0.030172] | 2/3 |
| metrica@v3/events | learned | 0.645220 | 0.577091 | 0.577091 | -0.068129 [-0.092120, -0.036930] | 0/3 |
| metrica@v3/events | oracle | 0.689024 | 0.621838 | 0.621838 | -0.067186 [-0.089220, -0.035152] | 0/3 |
| sc@v3/pp | learned | 0.633614 | 0.577846 | 0.577846 | -0.055768 [-0.065628, -0.046255] | 0/20 |
| sc@v3/pp | oracle | 0.788058 | 0.733217 | 0.733217 | -0.054841 [-0.065323, -0.045107] | 0/20 |
| sc@v3/phase | learned | 0.622135 | 0.597724 | 0.597724 | -0.024411 [-0.034383, -0.013798] | 3/20 |
| sc@v3/phase | oracle | 0.781473 | 0.761525 | 0.761525 | -0.019948 [-0.031224, -0.007750] | 4/20 |

Main result: learned-D. Oracle-D is diagnostic. No external tuning.
Metrica: three matches, noise draws averaged within match. SkillCorner: twenty matches.
95% intervals are paired match bootstraps, conditional on fixed models.
The internal split and external sets were used previously; not fresh holdouts.
