# Eşit ağırlıklı BiGRU ensemble — 25 Ekim 2026

Üç seed (20261025/26/27) için kare bazındaki P(home) olasılıklarının aritmetik ortalaması alındı.
Ağırlıkların her biri 1/3. Aynı v7 ölü-top kanalı ve B2; ağırlık/dekoder araması yapılmadı.
Kaynak olasılıklar eski seed CSV’lerini yeniden üretti. Ensemble CSV’si skorlamadan önce donduruldu.

| Model / özet | Açık bölüm spellF1 |
|---|---:|
| v7 | 0.6519 |
| Üç ayrı BiGRU skorunun ortalaması | 0.7770 |
| En yüksek ayrı seed (karşılaştırma amaçlı) | 0.7862 |
| **Eşit ağırlıklı olasılık ensemble** | **0.7724** |

Ensemble − v7: **+0.1206**; ensemble − ayrı seed skor ortalaması: **-0.0046**. macroF1: **0.9433**.
Açık bölümde GT possession: 17; tahmin: 12.

Tam-maç dosyası: `submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv`.
199.319 veri satırı, sıfır tabanlı frame indeksi ve frame,label başlığı. Kare düzeyi içerik gitignored yereldir.
Bu skor yalnız yaklaşık beş dakikalık açık GT bölümünündür; tam maçın gizli skoru bilinmiyor.
Önceki public sonuçlar zaten görülmüştü. Bu deney kullanıcının istediği sabit 1/3 ağırlıklı birleşimi ölçer.
Submission dosyası hazırlanmıştır; yarışma platformuna yüklenmemiştir.

```powershell
.venv/Scripts/python.exe src/exp_barath_ensemble.py build
.venv/Scripts/python.exe src/exp_barath_ensemble.py score
.venv/Scripts/python.exe external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv
.venv/Scripts/python.exe external/more-than-a-hack-2027/evaluation/run_evaluation.py --gt public --results submissions/barath_bigru_ensemble --out cache/barath_bigru_ensemble/official --bootstrap 1000 --no-detail
```
