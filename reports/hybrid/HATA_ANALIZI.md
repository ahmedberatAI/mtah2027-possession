# Possession hata analizi

Bu tablolar IoU=0.5 eşleşmesiyle tanısaldır. Ana spellF1 beş IoU eşiğinin ortalamasıdır.
Metrica gürültü seedleri önce maç içinde ortalanır; ardından sayımlar maçlar üzerinde toplanır.
Oranlar bu toplamların oranıdır; maçlara eşit ağırlıklı ana skorla aynı toplama işlemi değildir.
Eşleşmeyen tahmin, bütünüyle uydurma possession demek değildir: yetersiz zaman örtüşmesi de olabilir.

| Test | Model | <4sn kaçırma | Tüm possession kaçırma | Eşleşmeyen tahmin oranı | Birleştirme göstergesi | Bölme göstergesi |
|---|---|---:|---:|---:|---:|---:|
| metrica@v3/ctrl | bigru | 53.99% | 16.93% | 19.03% | 3.23% | 4.91% |
| metrica@v3/ctrl | lgb_balanced | 55.63% | 17.99% | 16.54% | 3.33% | 3.37% |
| metrica@v3/ctrl | v7 | 54.23% | 17.60% | 17.01% | 3.10% | 3.42% |
| metrica@v3/ctrl | hybrid | 53.99% | 16.93% | 19.03% | 3.23% | 4.91% |
| metrica@v3/events | bigru | 80.96% | 47.07% | 14.77% | 19.60% | 0.84% |
| metrica@v3/events | lgb_balanced | 84.60% | 49.99% | 15.91% | 21.73% | 0.93% |
| metrica@v3/events | v7 | 83.86% | 49.78% | 16.42% | 21.03% | 1.02% |
| metrica@v3/events | hybrid | 80.96% | 47.07% | 14.77% | 19.60% | 0.84% |
| sc@v3/pp | bigru | 75.70% | 37.40% | 33.33% | 7.91% | 3.47% |
| sc@v3/pp | lgb_balanced | 82.59% | 42.98% | 36.70% | 10.39% | 3.07% |
| sc@v3/pp | v7 | 82.59% | 42.63% | 38.82% | 9.79% | 3.54% |
| sc@v3/pp | hybrid | 75.70% | 37.40% | 33.33% | 7.91% | 3.47% |
| sc@v3/phase | bigru | 76.25% | 36.04% | 37.04% | 6.55% | 5.58% |
| sc@v3/phase | lgb_balanced | 81.81% | 40.77% | 39.24% | 8.56% | 4.73% |
| sc@v3/phase | v7 | 81.74% | 40.40% | 41.26% | 8.07% | 5.18% |
| sc@v3/phase | hybrid | 76.25% | 36.04% | 37.04% | 6.55% | 5.58% |

Birleştirme göstergesi: bir tahminin aynı takıma ait en az iki GT possession ile pozitif zaman örtüşmesi.
Bölme göstergesi: bir GT possession ile aynı takıma ait en az iki tahminin pozitif zaman örtüşmesi.
Bunlar tanısal göstergelerdir; birbirini dışlayan, kesin neden etiketleri değildir.

| Test | Model | Geçiş recall ±1sn | Geçiş precision ±1sn | Ortalama gecikme | Ortalama mutlak hata |
|---|---|---:|---:|---:|---:|
| metrica@v3/ctrl | bigru | 70.99% | 70.50% | +0.147 sn | 0.268 sn |
| metrica@v3/ctrl | lgb_balanced | 69.16% | 70.71% | +0.117 sn | 0.262 sn |
| metrica@v3/ctrl | v7 | 69.59% | 70.65% | +0.100 sn | 0.258 sn |
| metrica@v3/ctrl | hybrid | 70.99% | 70.50% | +0.147 sn | 0.268 sn |
| metrica@v3/events | bigru | 53.90% | 77.67% | +0.143 sn | 0.273 sn |
| metrica@v3/events | lgb_balanced | 51.15% | 75.88% | +0.113 sn | 0.271 sn |
| metrica@v3/events | v7 | 51.44% | 75.79% | +0.098 sn | 0.264 sn |
| metrica@v3/events | hybrid | 53.90% | 77.67% | +0.143 sn | 0.273 sn |
| sc@v3/pp | bigru | 44.86% | 46.96% | +0.062 sn | 0.351 sn |
| sc@v3/pp | lgb_balanced | 39.54% | 42.63% | +0.048 sn | 0.349 sn |
| sc@v3/pp | v7 | 39.01% | 40.88% | +0.018 sn | 0.350 sn |
| sc@v3/pp | hybrid | 44.86% | 46.96% | +0.062 sn | 0.351 sn |
| sc@v3/phase | bigru | 44.06% | 43.58% | +0.107 sn | 0.353 sn |
| sc@v3/phase | lgb_balanced | 39.59% | 40.32% | +0.084 sn | 0.349 sn |
| sc@v3/phase | v7 | 39.28% | 38.89% | +0.057 sn | 0.347 sn |
| sc@v3/phase | hybrid | 44.06% | 43.58% | +0.107 sn | 0.353 sn |

Geçişler ölü-top geçişlerini de içerir; hedef etiketi aynı olan geçişler ±25 karede birebir eşleştirilir.
Zamanlama yalnız eşleşen geçişler içindir. Kaçan geçişleri içermez; genel zamanlama başarısı diye okunmamalı.
Pozitif gecikme tahminin geç geldiğini belirtir. Bilinmeyen GT aralıkları arasında eşleşmeye izin verilmez.

| Test | İkisi doğru | Yalnız LGB doğru | Yalnız BiGRU doğru | İkisi de kaçırdı |
|---|---:|---:|---:|---:|
| metrica@v3/ctrl | 556.00 | 12.33 | 19.67 | 105.00 |
| metrica@v3/events | 555.00 | 17.67 | 51.00 | 521.33 |
| sc@v3/pp | 2993.00 | 149.00 | 456.00 | 1912.00 |
| sc@v3/phase | 2868.00 | 148.00 | 389.00 | 1687.00 |

Doğru burada GT possession’ın IoU=0.5 eşiğinde eşleşmesi anlamına gelir.
Metrica sayıları üç gürültü gerçekleşimi ortalandığından kesirli olabilir.
Tamamlayıcılık tablosu ideal seçimle kurtarılabilecek GT örneklerini gösterir; olasılık harmanının bunları otomatik kurtaracağını göstermez.
Kare bazlı veya possession başlangıç/bitiş verileri bu rapora yazılmamıştır.
