# v10 + shared BiGRU: round 13 external evaluation

Frozen probability/logit weights 0.25, 0.5, 0.75; common +/-25-frame logit smoothing and B2.
Three paired training-seed trials; bootstrap resamples matches after averaging noise/training seeds.
Ensemble column uses the existing v10 seed 0 and the three equal-weight shared BiGRU members.

| Context | D | Candidate | Three-seed mean | Delta vs v10 (95% CI) | Ensemble | Ensemble delta |
|---|---|---|---:|---|---:|---:|
| metrica@v3/ctrl | learned | bigru_raw | 0.802442 | -0.020863 [-0.026309, -0.011734] | 0.807148 | -0.016586 |
| metrica@v3/ctrl | learned | bigru_smooth25 | 0.810156 | -0.013149 [-0.018659, -0.004328] | 0.814179 | -0.009554 |
| metrica@v3/ctrl | learned | logit_b0.25 | 0.826162 | +0.002857 [-0.005150, +0.008711] | 0.826386 | +0.002652 |
| metrica@v3/ctrl | learned | logit_b0.5 | 0.826610 | +0.003305 [-0.008915, +0.010889] | 0.824591 | +0.000857 |
| metrica@v3/ctrl | learned | logit_b0.75 | 0.821831 | -0.001474 [-0.007676, +0.005439] | 0.819691 | -0.004043 |
| metrica@v3/ctrl | learned | probability_b0.25 | 0.825500 | +0.002195 [-0.006080, +0.007724] | 0.826267 | +0.002533 |
| metrica@v3/ctrl | learned | probability_b0.5 | 0.826576 | +0.003271 [-0.007724, +0.009885] | 0.826058 | +0.002324 |
| metrica@v3/ctrl | learned | probability_b0.75 | 0.824251 | +0.000946 [-0.008365, +0.009219] | 0.823325 | -0.000409 |
| metrica@v3/ctrl | learned | v10 | 0.823305 | +0.000000 [+0.000000, +0.000000] | 0.823734 | +0.000000 |
| metrica@v3/ctrl | oracle | bigru_raw | 0.858131 | -0.024156 [-0.031949, -0.012326] | 0.863521 | -0.018172 |
| metrica@v3/ctrl | oracle | bigru_smooth25 | 0.866309 | -0.015978 [-0.022346, -0.004579] | 0.869762 | -0.011931 |
| metrica@v3/ctrl | oracle | logit_b0.25 | 0.884452 | +0.002165 [-0.005367, +0.006468] | 0.883696 | +0.002003 |
| metrica@v3/ctrl | oracle | logit_b0.5 | 0.887096 | +0.004809 [-0.006556, +0.011437] | 0.887828 | +0.006135 |
| metrica@v3/ctrl | oracle | logit_b0.75 | 0.880867 | -0.001420 [-0.009611, +0.006379] | 0.880625 | -0.001068 |
| metrica@v3/ctrl | oracle | probability_b0.25 | 0.883548 | +0.001261 [-0.006815, +0.005782] | 0.883346 | +0.001653 |
| metrica@v3/ctrl | oracle | probability_b0.5 | 0.887182 | +0.004895 [-0.004762, +0.011999] | 0.886934 | +0.005241 |
| metrica@v3/ctrl | oracle | probability_b0.75 | 0.883737 | +0.001450 [-0.008658, +0.010938] | 0.884303 | +0.002610 |
| metrica@v3/ctrl | oracle | v10 | 0.882287 | +0.000000 [+0.000000, +0.000000] | 0.881693 | +0.000000 |
| sc@v3/phase | learned | bigru_raw | 0.619377 | +0.022069 [+0.013484, +0.029960] | 0.622135 | +0.024540 |
| sc@v3/phase | learned | bigru_smooth25 | 0.619388 | +0.022080 [+0.014553, +0.029269] | 0.623566 | +0.025970 |
| sc@v3/phase | learned | logit_b0.25 | 0.606934 | +0.009625 [+0.006562, +0.013193] | 0.607886 | +0.010290 |
| sc@v3/phase | learned | logit_b0.5 | 0.615101 | +0.017793 [+0.013214, +0.022704] | 0.617041 | +0.019446 |
| sc@v3/phase | learned | logit_b0.75 | 0.619094 | +0.021786 [+0.014820, +0.028639] | 0.621081 | +0.023486 |
| sc@v3/phase | learned | probability_b0.25 | 0.605230 | +0.007922 [+0.005085, +0.011093] | 0.606531 | +0.008936 |
| sc@v3/phase | learned | probability_b0.5 | 0.614473 | +0.017165 [+0.013145, +0.021654] | 0.617827 | +0.020232 |
| sc@v3/phase | learned | probability_b0.75 | 0.618923 | +0.021614 [+0.015277, +0.028202] | 0.621181 | +0.023586 |
| sc@v3/phase | learned | v10 | 0.597308 | +0.000000 [+0.000000, +0.000000] | 0.597595 | +0.000000 |
| sc@v3/phase | oracle | bigru_raw | 0.777289 | +0.017956 [+0.009205, +0.026573] | 0.781473 | +0.022100 |
| sc@v3/phase | oracle | bigru_smooth25 | 0.779270 | +0.019937 [+0.011970, +0.027583] | 0.782813 | +0.023440 |
| sc@v3/phase | oracle | logit_b0.25 | 0.767936 | +0.008603 [+0.005628, +0.011726] | 0.768672 | +0.009299 |
| sc@v3/phase | oracle | logit_b0.5 | 0.775285 | +0.015952 [+0.010738, +0.021259] | 0.776029 | +0.016656 |
| sc@v3/phase | oracle | logit_b0.75 | 0.779654 | +0.020322 [+0.012787, +0.027618] | 0.780965 | +0.021592 |
| sc@v3/phase | oracle | probability_b0.25 | 0.766234 | +0.006901 [+0.004041, +0.009909] | 0.767319 | +0.007946 |
| sc@v3/phase | oracle | probability_b0.5 | 0.773627 | +0.014294 [+0.010008, +0.018750] | 0.775622 | +0.016249 |
| sc@v3/phase | oracle | probability_b0.75 | 0.778865 | +0.019532 [+0.012607, +0.026308] | 0.781429 | +0.022056 |
| sc@v3/phase | oracle | v10 | 0.759333 | +0.000000 [+0.000000, +0.000000] | 0.759373 | +0.000000 |
| sc@v3/pp | learned | bigru_raw | 0.630735 | +0.039021 [+0.031896, +0.046924] | 0.633614 | +0.041327 |
| sc@v3/pp | learned | bigru_smooth25 | 0.627454 | +0.035739 [+0.029912, +0.042281] | 0.629712 | +0.037425 |
| sc@v3/pp | learned | logit_b0.25 | 0.603407 | +0.011692 [+0.008308, +0.015439] | 0.603534 | +0.011247 |
| sc@v3/pp | learned | logit_b0.5 | 0.614280 | +0.022566 [+0.018021, +0.027524] | 0.615815 | +0.023528 |
| sc@v3/pp | learned | logit_b0.75 | 0.622496 | +0.030781 [+0.024960, +0.037205] | 0.623620 | +0.031334 |
| sc@v3/pp | learned | probability_b0.25 | 0.600783 | +0.009069 [+0.006265, +0.012334] | 0.601446 | +0.009159 |
| sc@v3/pp | learned | probability_b0.5 | 0.612292 | +0.020578 [+0.016521, +0.025241] | 0.614724 | +0.022437 |
| sc@v3/pp | learned | probability_b0.75 | 0.621214 | +0.029499 [+0.024089, +0.035740] | 0.622471 | +0.030185 |
| sc@v3/pp | learned | v10 | 0.591715 | +0.000000 [+0.000000, +0.000000] | 0.592287 | +0.000000 |
| sc@v3/pp | oracle | bigru_raw | 0.784014 | +0.036444 [+0.028821, +0.045071] | 0.788058 | +0.040167 |
| sc@v3/pp | oracle | bigru_smooth25 | 0.781157 | +0.033586 [+0.027398, +0.040377] | 0.782251 | +0.034359 |
| sc@v3/pp | oracle | logit_b0.25 | 0.759157 | +0.011587 [+0.008373, +0.015085] | 0.759316 | +0.011424 |
| sc@v3/pp | oracle | logit_b0.5 | 0.767857 | +0.020287 [+0.015846, +0.025150] | 0.768913 | +0.021022 |
| sc@v3/pp | oracle | logit_b0.75 | 0.775862 | +0.028292 [+0.022348, +0.034673] | 0.776315 | +0.028424 |
| sc@v3/pp | oracle | probability_b0.25 | 0.756012 | +0.008442 [+0.005791, +0.011289] | 0.756668 | +0.008776 |
| sc@v3/pp | oracle | probability_b0.5 | 0.765054 | +0.017484 [+0.013352, +0.021973] | 0.767360 | +0.019468 |
| sc@v3/pp | oracle | probability_b0.75 | 0.773904 | +0.026333 [+0.020956, +0.032240] | 0.775949 | +0.028057 |
| sc@v3/pp | oracle | v10 | 0.747570 | +0.000000 [+0.000000, +0.000000] | 0.747891 | +0.000000 |

| Candidate | Worst regret, seed mean | Worst regret, ensemble |
|---|---:|---:|
| logit_b0.75 | 0.008239 | 0.011743 |
| probability_b0.75 | 0.010110 | 0.012109 |
| logit_b0.5 | 0.016455 | 0.019145 |
| probability_b0.5 | 0.018959 | 0.020698 |
| bigru_smooth25 | 0.020873 | 0.018066 |
| logit_b0.25 | 0.027328 | 0.030080 |
| bigru_raw | 0.029050 | 0.024307 |
| probability_b0.25 | 0.029952 | 0.032167 |
| v10 | 0.039021 | 0.041327 |

DFL is excluded: all shared checkpoints trained on all seven DFL matches.
These external sets were previously inspected; results are exploratory, not a fresh holdout.
Metrica has only three independent games. Label/dead contexts are correlated views.
The match bootstrap does not include training-data or grid-selection uncertainty.
No BAR-ATH data, public score, or automatic default change is part of this experiment.
v10 remains the submission pending DFL leave-match-out neural validation.
