# Fikir havuzu ve deneme döngüsü (CPU)

GPU bir süre dolu → bütün fikirler CPU'da denenir. Döngü: **fikir üret → CPU'da dene → ölç → kaydet →
sonuçlara göre yeni fikir üret**. Denenen her şeyin sonucu `docs/DENEMELER.md`'ye de satır olarak geçer.

## Ortak ölçüt (`src/bench.py`)

Her fikir aynı düzenekle ölçülür; yalnızca toplu sayılar basılır.

| Set | İçerik | Neden |
|---|---|---|
| **DFL-CV** | 7 açık DFL maçı, `sim.V2` bozulması, maç-dışarıda-bırak | büyük, ama etiket konvansiyonu farklı |
| **DFL-trk** | aynı maçlar, **kontrollü possession** etiketi (`bench.trk_ctrl`: rakip spell'inden sonra gelen spell ancak takım içi pas ya da ≥1 sn top tutma varsa sayılır) | GT konvansiyonuna yakın: 240 spell/maç, canlı dk başına 4.56 (Metrica ctrl 4.01, public GT 4.23) |
| **Metrica-ctrl** | 3 Metrica örnek maçı × 3 bozulma tohumu, olaylardan "kontrollü" etiket | GT'ye en benzeyen konvansiyon; model DFL'de eğitilir (sızıntı yok) |
| **SkillCorner** (tur 4) | 20 A-League maçı (yayın takibi, 10→25 fps), `sim.V3` bozulması, etiket = "phases of play" (top oyundayken topa sahip takım; faz dışı = D) | alan dışı + GT'ye benzeyen konvansiyon: medyan spell 7.4 sn, <2 sn %13 (public GT 7.56 sn, %12); 4.83 spell/canlı dk |
| **BAR-ATH** | public 5 dk (17 possession; sadece akıl sağlığı kontrolü) + tam maç tanıları (possession hızı, Smart Tagging uyumu) | karar verdirmez |

Konvansiyon istatistikleri (spell/maç · canlı dk başına · medyan sn):
DFL native 437 · 8.30 · 3.7 | DFL tau2 288 · 5.47 · 6.8 | DFL tau3 237 · 4.50 · 8.7 | **DFL trk 240 · 4.56 · 8.5** |
Metrica events 382 · 6.62 · 4.4 | Metrica tau2 269 · 4.66 · 8.5 | **Metrica ctrl 231 · 4.01 · 10.3**.

## Fikirler

Değer = beklenen spellF1 kazancı × güven; maliyet = CPU zamanı + kod. Durum: ⏳ sırada, 🔬 deneniyor, ✅ işe yaradı, ❌ yaramadı.

### A. Hedef / etiket uyumu
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| A1 | Takım modelini DFL **trk** (kontrollü) etiketleriyle eğitmek | Model, kontrolsüz dokunuşta possession'ı çevirmemeyi doğrudan öğrenir; dekoder cezası daha az iş yapar | düşük | ≈ kare doğruluğu ↑, spellF1 aynı (tur 1) |
| A2 | Metrica örnek maçlarını (ctrl) eğitime eklemek | 7 → 10 maç, GT konvansiyonunda gerçek olay etiketleri | orta | ⏳ |
| A3 | SkillCorner open data (diskte **20 maç**; possession "phases of play"tan) | daha fazla maç + kontrol tabanlı possession; ama yayın takibi (10 fps, kısmi görüş) | yüksek | ✅ yeni bağımsız doğrulama seti (`src/skillcorner.py`, `src/exp_sc.py`); eğitime katmak (A3b) deneniyor |

### B. Dekoder / yapı (metrik farkında)
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| B1 | Yeniden başlama farkında dekoder: D durumunu "H'den sonra D" / "A'dan sonra D" diye ikiye bölmek, D→takım geçişine bağlama göre (top yan çizgiden / kale çizgisinden çıktı, faul) maliyet | duran toptan sonra topu kimin alacağı, duruştan önceki takımla ve çıkış yeriyle güçlü ilişkili | orta | ❌ ≤ +0.0014 (aynı takım yeniden başlatma: kale çizgisi 0.14–0.22, yan çizgi 0.59–0.60, içeride 0.52–0.56; kare kanıtı yanında önsel etkisiz) |
| B2 | Süre farkında dekoder (HSMM benzeri; negatif binom süre) | min-süre + geometrik kuyruk kaba; gerçek spell süre dağılımı lognormal | orta | ✅ **yumuşak minimum** (`src/decode2.py`): 7 sn'den kısa possession'dan çıkış 48·(1−d/175); V3'te Metrica-ctrl +0.009 (9/9), DFL-trk +0.013 (5/7); cezayı önemsizleştiriyor; native/events konvansiyonunda −0.03…−0.05 (konvansiyon tercihi) |
| B3 | MBR çözümleme: posterior örneklerine karşı beklenen spellF1'i en yükseltecek aday | metriği doğrudan optimize eder; parçalanma düzeyini maça göre seçer | orta-yüksek | ❌ çalıştırılmadı: maç başına kâhin ceza seçiminin tavanı bile +0.004 / +0.005 |
| B4 | Geçiş tipine özel cezalar (H→D, D→H, H↔A ayrı) | şu an simetrik | düşük | ❌ ≤ +0.0014 |
| B5 | Bağlama bağlı takım cezası: top görünmüyorken / havadayken daha yüksek | kanıt zayıfken sahte geçişleri önler | düşük | ❌ V2'de DFL-trk +0.006, V3'te tutmadı (+0.001/+0.002) |
| B6 | Sistematik gecikme düzeltmesi (geçiş tipine göre medyan gecikme kadar kaydırma) | pencereli özellikler geçişi geç yakalıyor olabilir | çok düşük | ❌ simülasyonda medyan gecikme 0–2 kare; ≤ +0.001 |

### C. Takım modeli özellikleri
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| C1 | Takım perspektifinden **duran top dizilimi**: kendi oyuncusu taç çizgisi dışında / korner bayrağında / kendi kalecisi 5.50'de ve rakipler ceza sahası dışında / santrada | top görünmezken bile yeniden başlatan takımı dizilim söyler (taç atan oyuncu çizginin dışındadır) | düşük-orta | ≈ DFL-native +0.010, trk +0.005, ctrl 0 (`feat_team2:c1`) |
| C2 | Top sahibi sürekliliği: en yakın oyuncu kimliğinin koşu uzunluğu, takım içi el değiştirme (pas) sayısı | "kontrol" = takım içinde dolaşan top | orta | ❌ +0.000–0.002 |
| C3 | Uzun bağlam pencereleri (±10 / ±20 sn) | kontrollü possession uzun süreli bir durum | düşük | ❌ ±0.001 |
| C4 | Takım şekli: genişlik/derinlik, dışbükey örtü alanı, hatlar arası mesafe | hücum eden takım açılır, savunan takım daralır | orta | ✅ trk +0.010, ctrl +0.005 |
| C5 | Takım bazında hareket yönü: hücum eden takım ileri / topdan uzağa (pas almak için), savunan takım topa ve kendi kalesine doğru koşar | dikkat bulgusunun takım versiyonu | düşük-orta | ✅ trk +0.009, ctrl +0.008 |
| **C6** | **Baskı altında kontrol** (`src/feat_ctrl.py`): takım başına top–oyuncu birlikte hareket (3 m içinde min \|v_top − v_oyuncu\|), yaklaşma hızı (baskı), topa dönük koşu, kesintisiz topa yakınlık süresi | hata analizi: kaçan possession'lar iki takımın da topa yakın olduğu kısa top kaybı spell'leri; kontrol eden topla birlikte hareket eder, baskı yapan topa koşar | düşük | ✅ eşli: DFL-trk +0.0085 (6/7), DFL-native +0.012 (6/7), Metrica-ctrl +0.006 (7/9) |

### D. Canlı / ölü
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| D1 | Dikkat sinyalleri (oyuncu hareketi topa kilitli mi), bozulmuş DFL'de son görülen topla, v3 kurallarıyla harman | AUC 0.929 → 0.974 (temiz DFL); canlı/ölü darboğaz değil ama D bölmeleri spell'i ikiye ayırıyor | orta | ⏳ (`src/feat_dead.py` hazır) |
| N1 | Kurallar + öğrenilmiş ölü-top harmanı | V3 ile öğrenilmiş model taşınıyor | düşük | ❌ public'te canlı/ölü F1 0.974→0.965, macroF1 0.901→0.873; ölü payını eşitleyince de düşüyor |
| D2 | Yeniden başlama anını dizilimden (duran oyuncular + ani ivme) yakalamak | D→canlı sınırı keskinleşir | düşük-orta | ⏳ |

### E. Sağlamlık / alan uyumu
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| E1 | Adversarial validation: bozulmuş DFL vs BAR-ATH özellikleri | hangi özellik alan farkı taşıyor → çıkar / simülasyonu düzelt | düşük | ✅ **iki hata bulundu** (tur 1) |
| E2 | Çok tohumlu / çok simülasyonlu eğitim ve topluluk (V1+V2+V2T) | varyans düşer, alan farkına dayanıklılık artar | düşük-orta | ⏳ |
| E3 | Top görünürlüğüne göre olasılık kalibrasyonu (sıcaklık) | top yokken model fazla emin olabilir | düşük | ⏳ |
| N3 | native + trk takım modeli topluluğu (logit ortalaması) | iki konvansiyonun güçlü yanları | çok düşük | ≈ DFL-trk +0.005, Metrica-ctrl +0.003; events'te kayıp |
| N9 | Takım modelinde top kaybı (H↔A) çevresindeki karelere ×3 ağırlık | hatalar kısa top kaybı spell'lerinde | çok düşük | ≈ native/events konvansiyonlarında + (DFL-native +0.018 7/7), kontrollüde değil (DFL-trk −0.003); public macroF1 en yüksek (0.912) |
| N4 | `own_half` düzeltmesi (V3) | alan farkı hatası | çok düşük | ✅ ilkeli; simülasyonda DFL-trk +0.004, Metrica-ctrl +0.006 |

### F. Smart Tagging (SSS: "Use it — that is why it is there")
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| F1 | Faz etiketlerinin takımını zayıf takım kanıtı olarak eklemek | v5 ile tam maçta %86.6 uyum; hatalar bağımsızsa birleşim kazandırır | düşük | ❌ (tur 1) |
| F2 | Set piece etiketleri = yeniden başlama: başlangıcı D→canlı sınırı, takımı yeniden başlatan takım | 18% duran topta takım/yeniden başlama kaçıyor (setpiece_ok 0.815) | düşük | ❌ zayıf (tur 1) |

### G. Metriğe özel son işlem
| # | Fikir | Hipotez | Maliyet | Durum |
|---|---|---|---|---|
| G1 | A-B-A kalıbında kısa B spell'ini A'ya katmak | kontrollü konvansiyonda kısa ara spell nadiren gerçek | çok düşük | ❌ ≤ +0.0006 (B2 aynı işi ilkeli yapıyor) |
| G2 | İki aynı takım spell'i arasındaki kısa / zayıf D'yi silmek | sahte D bir spell'i ikiye bölüp iki eşleşmeyi bozar | çok düşük | ❌ ≤ 0 |

## Tur kayıtları
(her tur sonunda: ne denendi, sonuç, karar, yeni doğan fikirler)

### Tur 1 — 21 Ekim
- **Düzenek** (`src/bench.py`) + DFL için takipten "kontrollü possession" etiketi (`trk`).
- **A1** (sim V2): trk ile eğitim → kare takım doğruluğu DFL-trk 0.903→0.915, Metrica-ctrl 0.940→0.950; ama
  spellF1 DFL-trk 0.756→0.749, Metrica-ctrl 0.808→0.802; BAR-ATH public 0.448 (8/17 spell, aşırı birleştirme).
  Doğal model + yüksek ceza aynı yumuşatmayı zaten yapıyor. V3'te ikisi eşit (0.744/0.741, 0.7985/0.7984).
- **F1/F2 Smart Tagging** (`src/exp_tags.py`): public'te faz etiketinin takımı GT ile %76 (v5 aynı karelerde
  %85); çeliştiklerinde etiket yalnız %25 haklı → takım kanıtı olarak değersiz. Set piece etiketi yeniden
  başlamadan medyan 5.3 sn önce başlıyor, 59'unun 27'sinde takım var → zayıf.
- **E1 adversarial validation** (`src/exp_advval.py`) → **iki hata**:
  1. `sim.degrade` her kareye bağımsız konum gürültüsü (oyuncu 0.15 m, top 0.35 m) ekliyordu; türev alınınca
     hız gürültüsü ~2.6 / ~6 m/s. Bozulmuş DFL'de medyan oyuncu hızı 3.7 m/s (temiz DFL 1.9–2.1, BAR-ATH canlı
     oyunda 1.95), top–oyuncu bağıl hızı 10.2 (temiz 1.5, BAR-ATH 3.5); ayırıcı AUC 1.000, kazancın %92'si hız.
     → `sim.V3`: aynı rastgele çekilişler zamanda alçak geçiren filtreden geçiyor (oyuncu 1 sn, top 0.5 sn);
     maskeler V2 ile birebir aynı, konum hatası aynı (0.17 m), hızlar gerçekçi.
  2. `mlfeat.own_half_frac_c12`: NaN konumlu (sahada olmayan) slotlar "kendi yarısında değil" sayılıyor;
     BAR-ATH 46 slot, DFL ~30 → farklı seyreltme (tek değişkenli AUC 0.017). → `src/feat_fix.py:own_half`.
  Kalan fark maçlar arası doğal fark düzeyinde (DFL'de bir maç vs diğer altı: AUC 0.93–0.96).
- **V3 ile** (BAR-ATH public, v5 kodlayıcı, pen 8): native model 0.579 / macroF1 0.905 / takım 0.934 (V2 modeli
  0.529 / 0.898 / 0.925); trk modeli 0.579 / 0.908 / 0.938 ve tam maçta 4.22 possession/canlı dk (public GT 4.23).
  **Öğrenilmiş ölü-top modeli artık taşınıyor**: public ham canlı F1 0.915 (önceki denemeler 0.77–0.88; kurallar 0.974).
- Yeni fikirler: **N1** kurallar + öğrenilmiş ölü-top harmanı (V3 sayesinde DFL/Metrica'da ayarlanabilir),
  **N2** adversarial validation ile simülasyon kalibrasyon döngüsü (sıradaki farklar: oyuncu sayısı, topun
  kaybolduğu yer `last_edge`/`next_edge`), **N3** native + trk takım modeli topluluğu, **N4** üretim
  özelliklerinde `own_half` düzeltmesi + V3 ile yeniden eğitim.

### Tur 2 — 21 Ekim
- **Hata analizi** (`src/err_spells.py`, public 17 GT spell, IoU 0.5): yeniden başlama ile başlayan 4/4 doğru;
  **top kaybıyla başlayan 12'nin yalnız 3'ü** doğru (5 yanlış takım, 3 birleşmiş, 1 ölü). Yanlış takım verilen
  spell'ler kısa (ort. 4.1 sn), top %98 görünür ama GT takımı topa daha yakın olduğu kare yalnız %29 (doğru
  spell'lerde %82): **iki takımın da topa yakın olduğu baskı anları**. Simülasyonda aynı hata türü daha seyrek
  (Metrica-ctrl'de top kaybı spell'lerinin %80'i doğru; yanlış takımlarda GT takımı %54 yakın).
- GT zamanlaması: "en yakın takım" ile uyum GT ~30–40 kare öne alınınca en yüksek; Metrica-ctrl / DFL-trk
  etiketlerinde −20 → GT bizim etiketlerden ~0.4–0.8 sn geç possession değiştiriyor (küçük etki; B6).
- ATD: takım başına >11 iz karelerin %5–8'inde (hakem / yanlış atama?) ama public'te bu karelerde takım
  modeli %100 doğru; ağır oyuncu kayıpları ölü topta (canlıda p5 = 20 oyuncu). Taktik kamera sabit değil
  (görünür genişlik 52–76 m, oyunu izliyor).
- Denenenler: N1 ❌, N3 ≈, N9 ≈ (native'e iyi, kontrollüye değil), N4 ✅ (ilkeli), **C6 ✅**.
- Yeni fikirler: **N11** C6'nın güçlü sinyali "kesintisiz yakınlık süresi" (tek başına AUC 0.92) için uzun /
  üstel azalan sürümler; **N12** temiz DFL'den oyuncu düzeyinde "topu kontrol eden" dedektörü (en yakın + birlikte
  hareket) öğrenip takım düzeyine toplamak; **N13** C6 + C1 + own_half birleşimi ve dekoder iyileştirmeleriyle
  v7 adayı.

### Tur 3 — 21 Ekim
- **Özellik hattı (alt ajan, `src/feat_team2.py`)**, V3, native etiket, eşli: en iyi set **c145** (dizilim + şekil +
  hareket) — pen 32: DFL-trk +0.0071 (5/7), Metrica-ctrl +0.0069 (8/9); pen 48: +0.0132 (6/7) / +0.0083 (6/9);
  kazanç yalnız yüksek cezalarda (32–48). BAR-ATH public takım doğruluğu 0.934 → 0.953.
  Yan bulgular: taç atan oyuncunun çizgi dışı mesafesi yeniden başlatan takımı AUC 0.97 ile söylüyor ama kare
  özelliği olarak katkı yok → B1'de D→takım geçiş kanıtı olarak denenmeli; takım genişliği/derinliği tek başına
  AUC 0.80 (oyunda), 0.95 (yeniden başlamada); c4 yavaş (yarı başına ~25 sn, yüzdelikler) → üretimden önce
  vektörleştirilmeli; en iyi ceza ızgara kenarında (48) → ızgaraya 64, 96 eklendi.
- **own_half + C6** (V3, native): DFL-native 0.6028 (+0.014), DFL-trk 0.7505 (+0.0065), Metrica-ctrl 0.8073
  (+0.009), Metrica-events 0.6523 (+0.004).
- **Ceza seçimi**: simülasyonda en iyi ceza, tahmin/GT spell oranı 0.90–0.95 olan yerde. BAR-ATH için GT oranı
  ~4.2/canlı dk alınırsa hedef 3.8–4.0/dk; own_half + C6 modelinde bu pen 40–48 (pen 32: 4.40, pen 48: 3.81).
  Mevcut v5 (pen 8) 4.94/dk → aşırı parçalı (simülasyonda oran 1.15–1.2 ≈ −0.02…−0.04 spellF1).
- **Dekoder hattı (alt ajan, `src/decode2.py`, `src/exp_decoders.py`)**: B1, B4, B5, B6, G1, G2 gürültü düzeyinde;
  **B2 yumuşak minimum** benimsendi (yukarıdaki tablo).
- **Sentez** (`src/exp_v7.py`, V3 taban + standart dekoder en iyi cezasına göre eşli): taban+B2 +0.015 / +0.009;
  own_half+C6+B2 +0.011 / +0.018 (9/9); c145+B2 +0.019 (7/7) / +0.0165; **birleşim+B2 +0.016 (7/7) / +0.016 (9/9)**
  (DFL-trk / Metrica-ctrl). B2 ile özellik kazançları kısmen örtüşüyor; birleşim en sağlamı.
- trk etiketle birleşim yine geride (DFL-trk 0.749 vs 0.757, Metrica-ctrl 0.800 vs 0.809) → native kalıyor.
- **v7 adayı** (`src/model_v7.py`, `submissions/candidate_v7.csv`, MD5 48dbf986…): v3 kuralları + broadcast
  canlı/ölü, sim V3 + own_half + C6 + c145 takım modeli (native), B2 dekoder, pen_team 12. Temiz önbellekten
  yeniden üretim birebir (1 dk 41 sn). Public: spellF1 **0.652 [0.43–0.81]**, macroF1 **0.924**, canlı/ölü F1
  0.979, takım 0.948 (v5: 0.566 [0.36–0.76] / 0.901 / 0.974 / 0.929). Tam maç: 227 possession (~4.3/canlı dk),
  <2 sn 1; v5 ile kare uyumu %96.3. Public hata dökümü: top kaybı spell'leri 4/12 doğru (v5 3/12), eşleşmeyen
  tahmin 1 (v5 4). Güven aralıkları örtüşüyor → asıl dayanak simülasyondaki eşli kazanç.
- Kalan / yeni fikirler: **A2** Metrica maçlarını son eğitime katmak; **A3** SkillCorner; **N2** simülasyon
  kalibrasyonu (topun kaybolduğu yer, takım başına >11 iz); **N11** yakınlık süresi varyantları; c4'ü vektörleştirmek;
  kale çizgisi / taç atan oyuncu ipucunu takım modeline yeniden başlama özelliği olarak vermek.
  **En değerli sonraki adım:** elle etiketlenmiş birkaç 5 dakikalık pencere (`tools/labeler.html`) — public
  17 possession'la v5/v7 farkı istatistiksel olarak ayrılamıyor.

### Tur 4 — 21 Ekim
- Kullanıcı notu: **veri indirme yok** (kendi internetine bağlı); yalnız diskteki veriler kullanılıyor.
- **SkillCorner (A3)**: diskteki 20 maç `src/skillcorner.py` ile DFL düzenine çevrildi (25 fps ara değerleme, fazlardan
  H/A/D). Bozulmuş kinematik DFL/BAR-ATH ile uyumlu (canlı oyunda oyuncu hızı medyan 2.05 m/s; top 5.8 m/s, BAR-ATH 6.1).
  DFL'de eğitilen modeller SC'de (alan dışı, `src/exp_sc.py`): mutlak spellF1 ~0.57–0.58 — simülasyon içi
  DFL/Metrica'dan (0.75–0.81) çok düşük, **BAR-ATH public'e (0.57–0.65) çok yakın** → SC, BAR-ATH zorluğunu daha iyi
  yansıtan vekil. Eşli kazançlar (V3 taban std en iyi 0.5665'e göre): taban+B2 +0.0066 (14/20); own_half+C6+B2
  +0.0159 (17/20); c145+B2 +0.0178 (17/20); **birleşim+B2 (v7) +0.0167 (15/20)**; std ile birleşim +0.0075 (15/20).
  → v7'nin kazançları bağımsız bir veri setinde de tutuyor. DFL ölü-top modeli SC'de kare doğruluğu 0.903 (DFL'de 0.98).
- **SC'de hata kaynağı**: gerçek D etiketleriyle (oracle) v7 0.582 → **0.746**. SC'deki düşük skorun çoğu DFL ölü-top
  modelinin SC'ye taşınamamasından (birleşen spell'ler, 2132'ye karşı 884 sahte spell). BAR-ATH'ta D için kurallar +
  broadcast kullanıldığından takım fikirleri SC'de oracle D ile de ölçülüyor. Oracle D ile (V3 taban std 0.7326'ya göre):
  taban+B2 +0.0044 (9/20); own_half+C6 ±0.001 (C6 SC'de takım tarafında etkisiz); c145+B2 +0.0125 (15/20);
  **birleşim+B2 (v7) +0.0137 (17/20)**; trk ≈ 0. Kalan ana hata yine top kaybı spell'lerinde yanlış takım (%20).
- **A3b** SC'yi eğitime katmak (DFL + SC, 4 katlı): SC katlar dışı +0.025 (18/20, alan içi), Metrica-ctrl −0.005 (4/9),
  BAR-ATH public aynı (0.652; macroF1 0.929), tam maç 227 → 202 possession → ≈ kararsız; v7 DFL ile kalıyor.
- **SC'de yeniden sınama**: A1 trk eğitimi (B2 ile +0.003 vs native +0.0066) ❌, N9 (+0.0057) ❌, birleşim trk (+0.014 vs
  +0.0167) ❌ — simülasyondaki kararlar SC'de de geçerli.
- **B2 parametre taraması** (`src/exp_b2tune.py`, 64 ayar × SC/Metrica/DFL): varsayılan (D0 175, c 48, pen 12) düz bir
  platoda; en iyi ortalama sıra (175, 64, 8) yalnız +0.002/+0.0004/+0.004 ve BAR-ATH'ta fark yok → varsayılan kalıyor.
- **C7 dokunuş olayları** (`src/feat_touch.py`): top hız sıçraması + yakında oyuncu = dokunuş; takım başına son dokunan,
  ardışık dokunuş, takım içi pas, dokunuştan beri geçen süre. Sabit eşik alan bağımlı (3 m/s'de DFL 38, SC 24, BAR-ATH
  14 dokunuş/canlı dk; BAR-ATH topu çok pürüzsüz) → eşik her yarıda dakikada 10 dokunuşa uyarlanıyor (artık 17.5–19/canlı
  dk her yerde; dokunan takım = possession %85–87). Değerlendiriliyor.
- **C7 sonucu**: v7'ye eklenince GT benzeri setlerin hepsinde hafif kayıp (B2 ile DFL-trk 0.7600→0.7550, Metrica-ctrl
  0.8147→0.8128, SC oracle 0.7463→0.7425, SC 0.5832→0.5799); yalnız Metrica-events +0.019 → dokunuşlar hızlı el
  değişimlerini yakalıyor, GT'nin kontrollü tanımı bunları saymıyor. ❌ v7'ye alınmadı.
- **Sıcaklık (temp_t) B2 ile**: tutarlı yön yok (SC yüksek, DFL düşük ister; Metrica ve SC-oracle 1.0) → 1.0 kalıyor.
- **N19 topluluk** (birleşim + c145 + own_half/C6 modellerinin logit ortalaması): Metrica +0.0034 (7/9), DFL-trk +0.0054
  (5/7), SC +0.0013 (12/20) ama SC-oracle −0.0025 (8/20) → ≈ karışık; v7 tek model kalıyor.
- (Tur 4 burada kullanıcı isteğiyle durduruldu, 21 Ekim; kullanıcı "devam et" deyince tur 5 başladı.)

### Tur 5 — 21 Ekim
- **N20 zamansal yığınlama** (`src/exp_stack.py`; ikinci LightGBM, v7 olasılıklarının ±5 sn kaymaları ve pencere
  istatistikleri): DFL-trk −0.005 (2/7), Metrica-ctrl −0.006 (2/9), SC +0.0005, SC-oracle −0.010 → ❌ (kazancın %81'i
  zaten z0'dan; B2 ile çakışan bir yumuşatıcı).
- **Geçiş kaydırma** (takım değişimi / yeniden başlama ±0.5–1 sn): SC ve Metrica'da her kaydırma kaybettiriyor (SC'de
  yeniden başlamayı 0.5 sn kaydırmak −0.012…−0.023) → zamanlama bu konvansiyonlarla hizalı; ölü-top sınır hassasiyeti çok
  önemli. ❌
- **N21 SkillCorner'da BAR-ATH ölü-top yaklaşımı** (`src/exp_scdead.py`): SC'nin "görüntü sahayı gösteriyor" bayrağı
  BAR-ATH broadcast geçerliliği gibi davranıyor (canlı %94, ölü %25 geçerli; BAR-ATH ~%96.5 / %24). SC'de: öğrenilmiş DFL
  ölü-top 0.582, v3 kuralları + görüntü işaretleri 0.573, **kurallar + 0.5 × öğrenilmiş 0.619 (+0.038, 20/20)**,
  gerçek D 0.746. Ama BAR-ATH'ta (v7 ile) harman public macroF1'i 0.924 → 0.84–0.90 düşürüyor, vekiller çelişkili
  (ballout_cov ↑, setpiece_ok ↓). SC'de taktik kamera yok (tekrar sırasında oyuncu verisi de yok) → kurallar orada
  zayıf; BAR-ATH'ta sürekli taktik takip var. ≈ BAR-ATH'ta benimsenmedi.
- **N22 başlangıç türüne göre yumuşak minimum**: GT süre dağılımları yeniden başlama / top kaybı spell'leri için aynı
  (SC medyan 7.4 / 7.3 sn) → denenmeden ❌.
- **N23 takım modeli hiperparametreleri** (B2 ile, V3 taban std'ye göre; DFL-trk / Metrica-ctrl / SC-oracle / SC):
  v7 varsayılan +0.016 / +0.016 / +0.014 / +0.017; yavaş öğrenme (lr 0.025, 800 tur) +0.015 / +0.019 / +0.012 / +0.018;
  büyük ağaç (127 yaprak) +0.016 / +0.020 / +0.013 / +0.017; güçlü düzenlileştirme +0.015 / +0.015 / +0.008 / +0.019;
  her kare (stride 1) +0.021 / +0.017 / +0.008 / +0.015 → hepsi ±0.005 içinde, her biri başka sette önde → ≈ varsayılan kalıyor.
- **N24 v7 + hakem düdüğü** (v6'nın mantığı: düdükten sonra geç başlayan D'ye girişi düdüğe çekmek; `model_v7`
  `wh_bonus` seçeneği, varsayılan kapalı): public macroF1 0.9244 → **0.9314**, canlı/ölü F1 0.9788 → 0.9819, spellF1
  aynı; tam maçta karelerin %0.1'i değişiyor; vekiller aynı. Yalnız BAR-ATH'ta ölçülebiliyor (sesi olan tek veri).
  `submissions/candidate_v7w.csv` (MD5 00e1fab6…). Düdük skoru videodan yeniden üretilebilir (ffmpeg, kurulu pakette).
- **Tur 5 sonucu**: denenen 6 fikirden yalnız N24 (düdük) küçük bir artı (public macroF1 +0.007); diğerleri ≈/❌.
  CPU'da model tarafında getiri azaldı: son iki turda hiçbir fikir v7'yi tutarlı biçimde geçmedi. Kalan en değerli
  adımlar CPU deneyi değil: (1) elle etiketlenmiş birkaç 5 dakikalık pencere (v5 / v7 / v7w kararı), (2) teslim
  hattının (v7 veya v7w) varsayılan yapılması ve teknik dokümana işlenmesi.
