# Controlled DFL: error diagnostics

IoU=.5 diagnostic counts. Rates pool match counts after averaging Metrica noise seeds.
Split/merge are positive-overlap proxies, not exclusive causal diagnoses.

| Test | Model | Split proxy | Merge proxy | Missed GT | Unmatched predictions | <4s missed |
|---|---|---:|---:|---:|---:|---:|
| metrica@v3/ctrl | native_ensemble | 4.906% | 3.235% | 16.931% | 19.034% | 53.991% |
| metrica@v3/ctrl | controlled_fixed | 2.068% | 5.313% | 20.972% | 12.699% | 63.380% |
| metrica@v3/ctrl | controlled_selected | 2.068% | 5.313% | 20.972% | 12.699% | 63.380% |
| metrica@v3/events | native_ensemble | 0.844% | 19.597% | 47.074% | 14.768% | 80.961% |
| metrica@v3/events | controlled_fixed | 0.815% | 24.708% | 54.381% | 16.738% | 89.156% |
| metrica@v3/events | controlled_selected | 0.815% | 24.708% | 54.381% | 16.738% | 89.156% |
| sc@v3/pp | native_ensemble | 3.466% | 7.906% | 37.405% | 33.327% | 75.695% |
| sc@v3/pp | controlled_fixed | 1.960% | 15.089% | 48.185% | 31.943% | 87.097% |
| sc@v3/pp | controlled_selected | 1.960% | 15.089% | 48.185% | 31.943% | 87.097% |
| sc@v3/phase | native_ensemble | 5.577% | 6.553% | 36.037% | 37.038% | 76.254% |
| sc@v3/phase | controlled_fixed | 2.710% | 12.277% | 44.658% | 32.825% | 85.886% |
| sc@v3/phase | controlled_selected | 2.710% | 12.277% | 44.658% | 32.825% | 85.886% |
