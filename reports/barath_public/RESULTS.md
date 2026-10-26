# BAR-ATH açık bölüm — 25 Ekim 2026

Sabit v7 ölü-top kanalı ve B2; yalnız takım modeli değişir. Açık bölümde ayar seçilmedi.
Tüm yedi tahmin dosyası skorlama başlamadan üretildi ve hash ile donduruldu.

| Model | spellF1 | Δ v7 | macroF1 | GT possession | Tahmin possession |
|---|---:|---:|---:|---:|---:|
| v7 | 0.6519 | +0.0000 | 0.9244 | 17 | 10 |
| bigru_s20261025 | 0.7724 | +0.1206 | 0.9440 | 17 | 12 |
| bigru_s20261026 | 0.7724 | +0.1206 | 0.9412 | 17 | 12 |
| bigru_s20261027 | 0.7862 | +0.1344 | 0.9434 | 17 | 12 |
| lgb_balanced_s20261025 | 0.6667 | +0.0148 | 0.9250 | 17 | 10 |
| lgb_balanced_s20261026 | 0.6667 | +0.0148 | 0.9244 | 17 | 10 |
| lgb_balanced_s20261027 | 0.6667 | +0.0148 | 0.9243 | 17 | 10 |

| Aile | 3-seed ortalama ± SS | Δ v7 | Min–maks |
|---|---:|---:|---|
| bigru | 0.7770 ± 0.0080 | +0.1252 | 0.7724–0.7862 |
| lgb_balanced | 0.6667 ± 0.0000 | +0.0148 | 0.6667–0.6667 |

Bu sonuç yalnız yaklaşık beş dakikalık açık GT bölümünündür; tam maç gizli skoru değildir.
Üç seed aynı açık bölümü ölçer; üç bağımsız maç sayılmaz. Seed ortalaması ensemble değildir.
Sabit ayarlar ve modeller: design.json; yerel CSV dosyaları: prediction_manifest.json.
Ham takip, özellikler, olasılıklar ve kare etiketleri yalnız gitignored yerel klasörlerde kalır.
Yeni elle etiket, public ayarı veya yarışmaya gönderim yapılmadı.

```powershell
.venv/Scripts/python.exe src/exp_barath_public.py prepare
.venv/Scripts/python.exe src/exp_barath_public.py predict
.venv/Scripts/python.exe src/exp_barath_public.py score
```
