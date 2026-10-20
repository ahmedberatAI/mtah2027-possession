# `sim.degrade` top-kaybı düzeltmesi: önce / sonra

*20 Ekim 2026 · kod: `src/ball_gaps.py`, `src/simfix.py` (yeni); `src/sim.py`'ye yalnız ekleme
(`ball_dropout_v2`, `V2`, `V2_TACTICAL`; varsayılan v1 yolu bit-bit aynı, doğrulandı) · ham çıktılar
`reports/simfix/` · önbellek `cache/dfl_v2`, `cache/dfl_v2t`, `cache/simfix`. `cache/dfl`, `cache/metrica`,
`model_v5.py` ve `cache/dfl/models_all.pkl` değişmedi.*

> **Veri politikası.** BAR-ATH tarafında yalnız toplu istatistik üretildi: oranlar, medyanlar, sayılar,
> public-pencere skorları. Metrica örnek verisi ve açık DFL verisi kısıtsız.

## Özet

- **BAR-ATH'ın gerçek oranları (modelin gördüğü birleşik top, tam maç, v5 etiketleri):**
  - Top canlı oyunda karelerin **%6.6**'sında, ölü topta **%69**'unda yok. Yalnız tactical kamerada bu
    oranlar %10.7 ve %73.
  - Duraklamada kaybolan top, yeniden başlamaların yalnız **%22**'sinde (tactical %29) oyun başladıktan
    sonra da kayıp kalıyor. Bu taşmanın medyanı ~2 sn, p90'ı ~6 sn.
  - Toplam taşma canlı karelerin yalnız **%1.9**'u.
  - Ayrıca duraklamaların %24'ünde top, ölü toptan medyan 2.9 sn önce kayboluyor.
- **v1 simülasyonu:** canlıda %25 kayıp. Bunun %24 puanı taşma: yeniden başlamaların %71'inde top hâlâ
  yok, taşma medyanı 7.4 sn.
- **Düzeltme (`params=sim.V2`):**
  - Ölü topta başlayan boşluk yeniden başlamada kesiliyor. Kalibre edilmiş bir pay kısa bir süre devam
    ediyor.
  - Ölü top öncesi kayıp ve rastgele canlı boşluklar ekleniyor.
  - Hedefler her yarıda tam tutuyor. Sonuç profili BAR-ATH'ınkiyle neredeyse aynı (Bölüm 2).
- **DFL CV (aynı test verisinde, eşli):**
  - v2 modelleri BAR-ATH benzeri (v2) bozulmada spellF1'i **0.571 → 0.587**, canlı/ölü doğruluğunu
    0.969 → 0.980 çıkarıyor.
  - Takım modeli neredeyse aynı: doğruluk 0.8908 → 0.8914, log-loss 0.2445 → 0.2413, güven aynı.
  - Kazanç ölü-top modelinden geliyor. v5 bu modeli kullanmıyor. Bu model BAR-ATH'ta v2 ile de zayıf
    (public pencerede ham canlı/ölü doğruluğu 0.759 → 0.766).
- **"Takım modeli BAR-ATH'ta gereğinden az emin" hipotezi bu düzeltmeyle desteklenmiyor.**
  - v1 modeli v2 verisinde v2 modeli kadar emin (medyan |logit| 3.46 / 3.50) ve kalibre.
  - BAR-ATH'ta v2 modeli biraz daha az emin (2.93 → 2.78).
  - Public penceredeki ~2 puanlık düşük güven üç model setinde de aynı. Kaynağı top kaybı değil.
- **BAR-ATH v5 (toplu):** farklar gürültü bandında. v5 etiketleri %98 aynı.
  - pen 8'de public spellF1 0.566 → 0.542, pen 20'de 0.448 → 0.578.
  - Duran top proxy'si 0.815 → 0.926.
- **Metrica `pen_team` optimumu yerinde.** v2 modelleri ve v2 bozulmasıyla:
  - konvansiyon optimumları: events 5, ctrl 32 [24–32], tau3 32;
  - minimax yine **20**.
  - 20 ile 8 arasındaki fark ctrl'de +0.030 (9/9), events'te −0.019. Önceki koşuda bu farklar +0.037 ve
    −0.014'tü.
  - v2 modelleriyle "aynı çözülmüş hız" ölçütü 22–30'dan ~30–32'ye kayıyor.
- **Öneri:** `pen_team` önerisi (20, aralık 16–24) değişmiyor.
  - v2 modellerine geçmek ilkesel olarak daha doğru ama v5'e beklenen etkisi küçük. Metrica'da yalnız
    takım kanıtıyla (oracle) +0.001–0.003.
  - Karar ana oturumda; `models_all.pkl` değiştirilmedi.
  - DFL ölü-top modelini kullanacak her yeni sürüm v2 verisiyle eğitilmeli.

## 1. BAR-ATH'ta gerçek oranlar (görev 1)

Tanımlar:
- **Taşma:** ölü karede başlayan kayıp koşusunun kapsadığı canlı kareler.
- **Yeniden başlamada top yok:** D'den sonraki ilk canlı karede top yok.
- **D girişinde top yok:** D'den önceki son canlı karede top yok.

Bloklar devre başına; 2. yarı santradan başlıyor. Çıktı: `reports/simfix/barath_ball_profile.txt`.

**Birleşik top** (tactical + hizalanmış broadcast; `features_merged.pkl`, takım modelinin BAR-ATH girdisi):

| etiket kaynağı | canlıda kayıp | ölüde kayıp | taşma / canlı | yeniden başlamada top yok | taşma > 3 sn | taşma medyan (p90) sn | D girişinde top yok | önden kayıp medyan sn | görünür koşu medyan sn |
|---|---|---|---|---|---|---|---|---|---|
| public GT penceresi (5 dk, 4 duraklama) | 0.039 | 0.336 | 0 | 0/4 | 0/4 | – | 1/4 | 0.8 | 34.4 |
| v5 pen 8 (116 yeniden başlama) | **0.066** | **0.691** | **0.019** | **0.216** | 0.052 | 1.7 (5.8) | **0.237** | 2.9 | 15.9 |
| v5, top-varlığı ipuçları kapalı | 0.076 | 0.672 | 0.023 | 0.226 | 0.070 | 2.2 (5.8) | 0.239 | 2.7 | 15.9 |
| v3 canlı/ölü, top ipuçları kapalı | 0.091 | 0.681 | 0.019 | 0.235 | 0.070 | 2.2 (5.1) | 0.274 | 2.4 | 15.9 |
| v3, top + saha dışı ipuçları kapalı | 0.092 | 0.690 | 0.016 | 0.266 | 0.073 | 2.0 (5.0) | 0.315 | 2.4 | 15.9 |
| *tactical top, v5 pen 8* | *0.107* | *0.726* | *0.017* | *0.293* | *0.060* | *1.9 (5.4)* | *0.305* | *2.9* | *13.7* |

**Etiket yanlılığı kontrolü.**
- v5'in canlı/ölü kararı top ipuçlarını da kullanıyor. Bu yüzden v5 etiketleri canlıda kaybı aşağı çekebilir.
- Topu hiç görmeyen etiketler (son iki satır) daha gürültülü: public pencerede canlı/ölü uyumu
  0.935–0.950, v5'te 0.958. Etiket hatası canlı kaybı yukarı itiyor (ölü karelerin ~%70'i kayıp).
- Bütün varyantlar aynı resmi veriyor:
  - canlıda %6.5–9 kayıp,
  - yeniden başlamaların ~%22–27'sinde top yok, medyan ~2 sn,
  - taşma canlı karelerin ~%2'si.
- Public pencerede (4 duraklama) v5'in D başlangıçları ortalama 0.7 sn erken, bitişleri 0.9 sn geç. Yani
  taşma ve önden kayıp etiket gecikmesiyle açıklanamıyor; geç bitiş taşmayı olsa olsa küçük gösterir
  (n = 4, zayıf kanıt).

**Seçilen hedefler (`sim.V2`, birleşik top):** canlı %6.5, ölü %70, yeniden başlamada top yok %22 (taşma
medyan 2 sn, σ 0.8), D girişinde top yok %24 (öncül medyan 2.7 sn, σ 0.75). `sim.V2_TACTICAL`: canlı %12,
ölü %70 (v1 hedefleri), yeniden başlamada %30, D girişinde %30.

## 2. Düzeltme: `sim.ball_dropout_v2` (görev 2)

`sim.degrade(..., params=sim.V2)` ile açılıyor. Parametre verilmezse v1 aynen çalışıyor:
`sim_orig` çıktısıyla `np.array_equal` doğrulandı.

**Mekanizma:**
- Ölü topta başlayan lognormal boşluklar (medyan 10.7 sn) yeniden başlamada kesiliyor.
- Yeniden başlamaların bir payında top kısa bir lognormal süre daha kayıp kalıyor. Bu yalnız son ölü
  karede top yoksa oluyor.
- Duraklamaların bir payında top, ölü top başlamadan önce kayboluyor. Bu durumda ölü boşluk duraklamayla
  birlikte başlıyor.
- Kalan canlı kayıp, rastgele boşluklarla dolduruluyor. Havadaki topta bu boşluklar 2.5 kat daha olası.

**Kalibrasyon:**
- Rastgele sayılar bir kez çekiliyor (common random numbers).
- Boşlukların birleşimi her parametrede monoton. Bu yüzden oranlar ikiye bölme (bisection) ile, olay
  payları ise tam sayımla her yarıda tutturuluyor. Kaynağın kendi kayıpları da hesaba katılıyor.
- v2 modunda oyuncu bozulması v1'le aynı rastgele akışı kullanıyor. Böylece v1/v2 verileri yalnız top
  maskesinde farklı (eşli karşılaştırma).

| profil (gerçek etiketler) | canlıda kayıp | ölüde kayıp | taşma / canlı | yeniden başlamada top yok | taşma > 3 / > 10 sn | taşma medyan (p90) | D girişinde top yok | önden kayıp medyan | D başına koşu | görünür koşu medyan |
|---|---|---|---|---|---|---|---|---|---|---|
| BAR-ATH birleşik (v5 pen 8) | 0.066 | 0.691 | 0.019 | 0.216 | 0.052 / 0.000 | 1.7 (5.8) | 0.237 | 2.9 | 1.02 | 15.9 |
| **DFL, sim v2** | **0.066** | **0.700** | **0.016** | **0.223** | 0.067 / 0.003 | 2.2 (5.0) | **0.238** | 2.8 | 1.23 | **16.1** |
| BAR-ATH tactical (v5 pen 8) | 0.107 | 0.726 | 0.017 | 0.293 | 0.060 / 0.009 | 1.9 (5.4) | 0.305 | 2.9 | 0.95 | 13.7 |
| DFL, sim v2t | 0.120 | 0.700 | 0.019 | 0.306 | 0.089 / 0.009 | 2.1 (5.2) | 0.303 | 3.1 | 1.24 | 14.5 |
| DFL, sim v1 (`cache/dfl`) | 0.253 | 0.676 | **0.242** | **0.710** | 0.546 / 0.266 | 7.4 (25.3) | 0.232 | 8.9 | 1.81 | 7.5 |
| Metrica, sim v1 → v2 | 0.221 → 0.065 | 0.980 → 0.941 | 0.201 → 0.012 | 0.724 → 0.251 | | | 0.241 → 0.241 | | | 13.2 → 14.0 |

**Yarı bazında tutarlılık.**
- `cache/dfl`'de (v1) canlı kayıp yarıdan yarıya %17–35, ölü kayıp %58–73 arasında değişiyordu. v2'de
  canlı %5.7–7.3, ölü %69.5–71.1, yeniden başlamada top yok %21–24.
- Metrica'nın temiz takibinde top ölüyken zaten yok; bu yüzden ölü kayıp ~%94'ün altına inemiyor.

## 3. DFL CV, önce / sonra (görev 3)

**`train_dfl.cv()`, her varyant kendi bozulmasında.** Değerler 7 maçın ortalaması. Kaynaklar:
`reports/dfl_cv.txt`, `reports/simfix/dfl_cv_v2.txt`, `reports/simfix/dfl_cv_v2t.txt`.

| eğitim = test | kare doğruluğu | canlı/ölü | takım (canlı) | en iyi spellF1 (en kötü maç) | en iyi dekoder |
|---|---|---|---|---|---|
| v1 (önce) | 0.906 | 0.965 | 0.885 | 0.5492 (0.469) | pen_team 4, pen_dead 32 |
| **v2** | **0.924** | **0.980** | **0.891** | **0.5874** (0.515) | pen_team 4, pen_dead 64 (32'de 0.5868) |
| v2t | 0.917 | 0.974 | 0.889 | 0.5734 (0.516) | pen_team 4, pen_dead 64 |

Bu tablo iki etkiyi karıştırıyor: v2 test verisi daha kolay. Bu yüzden her kat modeli her bozulmada
değerlendirildi (`reports/simfix/dfl_cross.txt`). v1 kat modellerinin OOF'u `cache/dfl/oof.pkl` ile
birebir aynı.

| kat modelleri (eğitim) | test | kare | canlı/ölü | takım | takım, top yokken | takım log-loss | güven − doğruluk | medyan abs logit | spellF1 @4/32 | en iyi spellF1 |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | **v2** | 0.9138 | 0.9694 | 0.8908 | 0.798 | 0.2445 | +0.004 | 3.46 | 0.5714 | 0.5715 |
| v2t | **v2** | 0.9232 | 0.9795 | 0.8914 | 0.805 | 0.2425 | +0.003 | 3.51 | 0.5834 | 0.5841 |
| **v2** | **v2** | **0.9245** | **0.9802** | **0.8914** | 0.800 | **0.2413** | +0.002 | 3.50 | **0.5868** | **0.5874** |
| v1 | v1 | 0.9060 | 0.9649 | 0.8851 | 0.850 | 0.2607 | +0.002 | 3.13 | 0.5492 | 0.5492 |
| v2 | v1 | 0.8839 | 0.9366 | 0.8835 | 0.844 | 0.2635 | −0.003 | 3.01 | 0.5233 | 0.5244 |

- BAR-ATH benzeri veride v2 modelleri **+0.015 spellF1** kazandırıyor. Kazancın tamamına yakını ölü-top
  modelinden (canlı/ölü +0.011).
- Takım modeli değişmiyor: doğruluk +0.0006, log-loss −0.003, güven ve kalibrasyon aynı.
- v1'in taşma rejiminde (gerçekçi değil) v2 modelleri −0.026 kaybediyor.
- v1'de top yokken takım doğruluğu yüksek (0.85): kayıp kareler çoğunlukla yeniden başlamadan sonra ve
  takım belli. v2'de kayıp kareler duraklama öncesine kayıyor ve takım doğruluğu 0.80'e iniyor.

## 4. BAR-ATH'ta v5 (yalnız toplu; `reports/simfix/barath_v5.txt`)

- v5 dekoderi ve canlı/ölü kanıtı (v3) aynı; yalnız takım olasılıkları değişti.
- `model_v5.py` değiştirilmedi. Olasılıklar `model_v5.team_logp(f, path)` ile yüklendi.
- v1 olasılıkları `cache/dfl/barath_probs.pkl` ile birebir aynı.

| | v1 modelleri (şimdiki) | v2 | v2t |
|---|---|---|---|
| ham takım modeli, public canlı kareler: doğruluk / log-loss / güven − doğruluk | 0.915 / 0.211 / −0.020 | 0.912 / 0.213 / −0.020 | 0.913 / 0.213 / −0.021 |
| v5-canlı kareler, tam maç: medyan abs logit (top yokken) · p > 0.9 payı | 2.93 (1.25) · 0.612 | 2.78 (0.98) · 0.593 | 2.76 (1.13) · 0.589 |
| v5 pen 8: public spellF1 (tahmin/gerçek) · macroF1 · takım | 0.566 (12/17) · 0.901 · 0.929 | 0.542 (14/17) · 0.895 · 0.921 | 0.578 (10/17) · 0.891 · 0.915 |
| v5 pen 8: possession (canlı dk başına) · tag_team · setpiece_ok | 261 (4.94) · 0.866 · 0.815 | 277 (5.25) · 0.860 · 0.926 | 272 (5.14) · 0.865 · 0.889 |
| v5 pen 20: public spellF1 · macroF1 · possession (canlı dk başına) | 0.448 (8/17) · 0.885 · 218 (4.13) | 0.578 (10/17) · 0.892 · 236 (4.48) | 0.578 (10/17) · 0.891 · 222 (4.21) |

- **D tarafı aynı:** dead share 0.501/0.502, adF1 0.974; restart ve ballout proxy'leri aynı.
- **v5 pen 8 etiketleri:** v1 modelleriyle kare uyumu 0.980.
- **Public pencere:** 17 possession; bir possession ≈ 0.06 spellF1. Değişimler gürültü bandında.
- **pen 20'deki public düşüşü:** Metrica raporunun ana karşı-sinyali, v1'de 0.448. v2 modelleriyle
  kayboluyor (0.578). Kanıt zayıf, aynı 17 possession.
- **DFL ölü-top modeli BAR-ATH'a yine taşınmıyor** (`reports/simfix/barath_dead_model.txt`).
  - Ham kare bazında public pencerede canlı/ölü doğruluğu v1 0.759, v2 0.766, v2t 0.769. Log-loss
    1.00 / 0.99 / 0.90.
  - v3 kurallarının adF1'i 0.974. Alan farkı sürüyor.
  - v2'nin DFL'deki ölü-top kazancı burada görünmüyor (4 duraklama, zayıf kanıt).

## 5. Metrica `pen_team` kalibrasyonu (görev 4)

- `metrica_calibration` taraması yönlendirilmiş yollarla yeniden koşuldu. Çıktılar:
  `reports/simfix/metrica_<model>_<bozulma>/metrica_sweep.*` ve `reports/simfix/metrica_regret_paired.txt`.
- Aynı 3 maç × 3 tohum kullanıldı. v1 × v1 olasılıkları `cache/metrica` ile birebir aynı.
- Değerler: v5 dekoderi, öğrenilmiş ölü-top kanıtı; en iyi `pen_team` ve [en iyiden ≤ 0.005 aralık].

| modeller × Metrica bozulması | events | tau1 | tau2 | tau3 | ctrl | DFL kendi | minimax (öğrenilmiş / oracle) |
|---|---|---|---|---|---|---|---|
| v1 × v1 (önceki rapor) | 5 [3–8] | 6 [5–12] | 24 [16–32] | 32 [32] | 32 [20–32] | 2 | 20 / 20 |
| v1 × v2 | 5 [3–8] | 6 [4–12] | 20 [16–32] | 32 [32] | 32 [32] | 2 | 20 / 20 |
| v2 × v1 | 4 [3–8] | 8 [5–12] | 16 [12–24] | 32 [32–48] | 24 [24–48] | 3 | 20 / 20 |
| **v2 × v2** | **5 [3–12]** | 12 [5–16] | 16 [12–32] | 32 [32–48] | **32 [24–32]** | 3 | **20 / 20** |
| v2t × v2 | 6 [3–12] | 12 [6–12] | 24 [16–24] | 32 [32] | 32 [24–48] | 3 | 24 / 20 |

- **pen_team 20 − 8 (eşli, 9 koşu):**
  - v2 × v2: ctrl +0.0295 (9/9), tau2 +0.0103 (7/9), events −0.0192 (0/9).
  - Önceki v1 × v1: ctrl +0.0373 (9/9), tau2 +0.0136 (7/9), events −0.0140 (1/9).
- **Beklenen pişmanlık en iyileri (v2 × v2):** ctrl-ağırlıklı 32, eşit önsel 24, public-pencere ağırlıklı
  32, yalnız events 5. Önceki koşuda: 24 / 20 / 32 / 5.
  - Eşit önselde 16–32 arası farklar ≤ 0.0015.
  - ctrl-ağırlıklı ve public-pencere ağırlıklı önsellerde 32'nin pişmanlığı 20'ninkinden 0.006 / 0.009
    daha az.
- **Takım modelinin payı (v2 bozulmasında v1 → v2 modelleri):** ctrl optimumunda spellF1 öğrenilmiş
  kanıtla 0.800 → 0.808, oracle'da 0.865 → 0.867. Fark ölü-top modelinden.
- **Aynı çözülmüş hız ölçütü:**
  - Metrica ctrl optimumu (pen 24–32) her iki koşuda maç başına 227–234 spell üretiyor, canlı dk başına
    ~3.9–4.1.
  - BAR-ATH'ta v1 modelleriyle bu hıza ~22–30'da ulaşılıyor (20: 4.13, 24: 3.96, 32: 3.82; önceki rapor).
  - v2 modelleriyle ~30–32'de ulaşılıyor (24: 4.30, 28: 4.19, 32: 3.93).

## 6. Sonuç

1. Yan bulgu doğrulandı ve düzeltildi. Sorunun kaynağı ölü topta başlayan boşlukların canlı oyuna taşması.
   BAR-ATH'ta bu taşma canlı karelerin ~%2'si, v1'de ~%24'üydü. `sim.V2` BAR-ATH profilini canlı/ölü
   oranlarında, yeniden başlama ve giriş davranışında ve görünür koşu uzunluğunda yakalıyor.
2. Düzeltme DFL'de eşli **+0.015 spellF1** veriyor. Kazanç ölü-top modelinden geliyor. Takım modeli ve onun
   BAR-ATH güveni değişmiyor; "fazla zor rejim takım modelini çekingen yapıyor" etkisi yok.
3. `pen_team` önerisi değişmiyor: 20 (aralık 16–24). v2 modelleriyle hız ölçütü biraz yukarı (24–32)
   işaret ediyor. Elle etiketlenmiş doğrulama seti yine asıl karar yeri.
4. Ana oturuma kalan kararlar:
   - (a) v5 için `cache/dfl_v2/barath_probs.pkl`'yi kullanmak. Etki küçük ve gürültü bandında. Duran top
     proxy'si +0.11, pen 20 public düşüşü yok.
   - (b) `models_all.pkl`'yi v2 ile değiştirmek. Bu oturumda yapılmadı.
   - DFL ölü-top olasılığını kullanacak her sürüm v2 ile eğitilmeli. Ancak bu model BAR-ATH'ta v2 ile de
     zayıf (public pencerede ham doğruluk 0.77). v5'in v3 tabanlı canlı/ölü kanıtının yerini tutmuyor.

## 7. Yeniden üretim

```bash
.venv/Scripts/python.exe -I src/ball_gaps.py barath            # görev 1, BAR-ATH toplu profil (~15 sn)
.venv/Scripts/python.exe -I src/simfix.py build v2 v2t         # bozulmuş DFL öznitelikleri (~1 dk, 7 süreç)
.venv/Scripts/python.exe -I src/simfix.py cv v2                # train_dfl.cv -> cache/dfl_v2 (+ kat modelleri)
.venv/Scripts/python.exe -I src/simfix.py cv v2t
.venv/Scripts/python.exe -I src/simfix.py folds v1             # v1 kat modelleri (cache/dfl'ye yazmaz)
.venv/Scripts/python.exe -I src/simfix.py cross                # eşli çapraz tablo
.venv/Scripts/python.exe -I src/simfix.py models v2 v2t        # tüm-DFL modelleri -> cache/dfl_<v>/models_all.pkl
.venv/Scripts/python.exe -I src/simfix.py barath               # v5 x model seti, yalnız toplu
.venv/Scripts/python.exe -I src/simfix.py metrica_build        # Metrica v1/v2 bozulma x model setleri
.venv/Scripts/python.exe -I src/simfix.py metrica_sweep v2 v2  # (ve v1 v2, v2 v1, v2t v2)
.venv/Scripts/python.exe -I src/simfix.py metrica_summary
```

Her adım deterministik: sabit tohumlar ve LightGBM `deterministic=True`. v1 yolları mevcut önbelleklerle
birebir aynı.
