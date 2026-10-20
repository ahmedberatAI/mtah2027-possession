# More than a Hack 2027 — Faz 1 (Top Sahipliği) · Analiz, Araştırma ve Baseline Raporu

*Hazırlanma: 20 Ekim 2026 · Veri: FC Barcelona – Athletic Club (LaLiga 2024-25, 2. hafta)*

> **Veri politikası notu.** Yarışma şartları, verinin veya herhangi bir parçasının bulut tabanlı bir
> LLM'e yüklenmesini yasaklıyor. Bu nedenle tüm analizler lokal scriptlerle yapıldı; bu rapordaki her
> sayı **toplu istatistik** (dağılım, oran, skor). Ham satır, video karesi veya kare-bazlı türev yok.
> Bu dosyayı herkese açık bir yere koymayın; frame-level türev veri paylaşımı da yasak.

---

## 1. Yarışmanın özü

| | |
|---|---|
| Görev | 199,319 karenin her biri için `H` (Barça topa sahip, oyun canlı) / `A` (Athletic) / `D` (ölü top) / `X` (cevap yok) |
| Değerlendirilen kareler | 1. yarı 11,185–90,651 · 2. yarı 110,808–190,197 (158,857 kare ≈ 106 dk). 110,808–113,741 arası (2. yarı santrasından önce) **tamamı D** |
| Sıralama metriği | **spellF1**: possession = kesintisiz H veya A koşusu (D böler). Tahmin/gerçek possession'lar aynı takım şartıyla, temporal IoU ile birebir açgözlü eşleşir; F1, IoU 0.3/0.4/0.5/0.6/0.7'de hesaplanıp ortalanır |
| Eşitlik bozucu | macroF1 (H/A/D kare bazlı) |
| `X` | spellF1'de "önceki etiketi tekrarla" demek (ileri doldurulur); macroF1'de hata. Değerlendirilen karelerde kullanmanın faydası yok |
| Hassasiyet | 1 kare gecikme bedava; 8 kare → 0.893; **1 sn gecikme → 0.654**. Tam maçta <0.04 fark gürültü |
| Yayınlanan taban | `baseline_smooth50` spellF1 **0.2433** (tam maç) |
| Public GT | Sadece ilk 5 dk (17 possession). Public'te bootstrap aralığı **0.27–0.62** — model seçimi için kullanılamaz |
| Kurallar | Her yöntem serbest (dış veri, hazır modeller dahil). Etiketler algoritmadan çıkmalı; az miktarda elle etiket **eğitim/doğrulama için serbest**. Repo dosyayı birebir yeniden üretmeli. Teslim: 17 Kasım 2026 |

GT, Metrica'nın **düzeltilmiş** takip verisinden mekanik olarak türetilmiş: top ölüyse D, değilse
düzeltilmiş verinin "topa sahip takım" alanı. Bize verilen ise **otomatik, düzeltilmemiş** takip.

## 2. Veri durumu

| Dosya | Durum |
|---|---|
| tactical ATD (txt+xml), homografi, Smart Tagging | ✅ indirildi, doğrulandı (199,319 satır, frame ID 1..199,319 kesintisiz) |
| tactical video (5.2 GB) | ✅ indirildi; 1920×1080, 25 fps, 2:12:52.76 = 199,319 kare, **AAC stereo ses var** |
| broadcast ATD, homografi | ✅ (164,488 satır) |
| broadcast video (13 GB) | ⏳ Drive indirme kotası doluydu; tekrar denenecek |
| Open DFL / IDSSE (7 Bundesliga maçı, CC BY 4.0) | dış eğitim/doğrulama verisi |

Kendi parse'ımla resmi 4 baseline'ı **sıfır fark**la yeniden ürettim → hizalama doğru.

## 3. Veri analizi — temel bulgular

### 3.1 Takip verisi
- Top, değerlendirilen karelerin **%59.3**'ünde görünür. Boşluklar **uzun**: 150 boşluk, medyan **10.7 sn**,
  kayıp karelerin %44'ü 30 sn'den uzun boşluklarda; 1 sn'den kısa boşluk neredeyse yok.
- Görünür periyotlar medyan 13.8 sn.
- Public pencerede top görünürlüğü: **H %99.9 · A %85 · D %66** → kayıp top büyük ölçüde ölü top ve bazı canlı fazlar.
- Kayıp topun ~%29'u top çizgiye ≤2 m yakınken başlıyor (oyun dışı), ~%30'u ceza sahasında bitiyor (aut vuruşu vb.).
- Oyuncu: kare başına ort. 20.5 takip (p5=14); 1,219 tracklet, medyan 35 sn; her takımda **1 kaleci slotu** (`position_type=Goalkeeper`, %93 doluluk).
- **Hız kanalı m/s** (XML "kph" diyor ama konumdan türetilen hızla oran 1.006, korelasyon 0.90).
- **Top hız kanalı anlamsız**: sadece 20,971 karede var ve konum-türevli hızla korelasyonu −0.10 → kullanılmamalı.
- Top saha dışında: değerlendirilen top-görünür karelerin %4.2'si; GT-D'de %10.7, canlıda ~%0.

### 3.2 Public GT'ye karşı
- "En yakın oyuncunun takımı = GT takımı": görünür canlı karelerde %77.8; mesafe ≤1 m iken %90.6.
- "Son yakın takım" (R metre, k ardışık kare) kuralları R/k'dan bağımsız **~%85–87 takım doğruluğunda tavan** yapıyor.
- Takım hatalarının **tamamı GT geçişlerinin 8 sn yakınında** (%36'sı ±1 sn içinde) → sorun zamanlama ve kısa possession'lar.
- **Oracle analizi**: canlı/ölü mükemmel olsa spellF1 0.461→0.473; **takım mükemmel olsa 0.889**. Darboğaz H↔A kararı.
- Ölü topta oyuncu hızı belirgin düşük (D ort. 1.1–2.0 m/s vs canlı 2.1–3.4); oyuncular hakeme **~1.5–2 sn gecikmeyle** tepki veriyor.

### 3.3 Smart Tagging (638 etiket)
- Faz etiketleri **çiftler halinde**: PROGRESSION ↔ DEFENDING IN MIDDLE THIRD, BUILD UP ↔ DEFENDING IN ATTACKING THIRD, CREATING CHANCES ↔ DEFENDING IN DEFENSIVE THIRD, ATTACKING ↔ DEFENSIVE TRANSITION (DEFENDING etiketinin takımı = savunan takım).
- Public GT ile takım uyumu: BUILD UP %91, LONG BALL %90, PROGRESSION %73, ATTACKING TRANSITION %67.
- 59 duran top: 26 taç (hepsi Team=N/A), 12 korner, 10 aut, 5 serbest vuruş, 6 santra. Tipik maçta ~100–110 duraklama olduğundan etiketler eksik.
- Duran top penceresi, gerçek yeniden başlamadan 3–12 sn önce açılıyor; birçok etiket sabit süreli (30–35 sn).

### 3.4 Homografi / kamera
- Tactical homografi %100 geçerli; görüntü merkezi topa uzak (medyan 23.6 m) → kamera merkezi zayıf top vekili.

### 3.5 Broadcast — **hizalama çözüldü**
- Yayın gerçek zamanlı kaydedilmiş, devre arasında kesilmiş: **1. yarı `tactical = broadcast + 4014`, 2. yarı `+ 26920`** (2'şer dakikalık bloklarda ±1–2 kare sabit; eşleşen top konumlarında medyan fark 0.5–1 m; oyuncu konumlarında medyan 0.30 m).
- **Ölü top sinyali**: public GT-D karelerinin **%76'sında broadcast homografisi geçersiz** (tekrar/yakın çekim), canlıda sadece ~%3.5 (olabilirlik oranı ≈ 22).
- Broadcast topu, tactical'da kayıp olan canlı-oyun karelerinin %40–60'ını dolduruyor (tüm maçta top kapsaması %59.3 → %63.1).
- İki kameranın top konumu >2.5 m ayrışınca (top havada → zemine izdüşüm paralaksı) en-yakın-takım doğruluğu %89.5'ten ~%71'e düşüyor → güvenilirlik sinyali.

### 3.6 Ses
- Her iki santrada GT başlangıcından ~1.1 sn önce güçlü düdük tepesi var. Ancak basit bant-tonalite detektörü tüm maçta tribün ıslıkları yüzünden ayırt edici değil → önceden eğitilmiş ses olayı modeli (AudioSet "Whistle") gerekli.

## 4. Araştırma özeti (benzer problemler / en güçlü yaklaşımlar)

- **2026 edisyonu** açık uçlu GenAI hackathonuydu → doğrudan kullanılabilir çözüm yok; 2027 reposunda henüz issue yok.
- **Yapısal çözümleme** (HSMM / min-süreli Viterbi, sınır yakalama) segment metriklerinde en büyük kaldıraç (MS-TCN/ASRF literatürü, F1@k ≈ spellF1).
- **Ölü top**: Lang et al. 2022 — sadece oyuncu verisiyle, ±0.4…80 sn zaman kaydırmalı bağlam: %92 kare doğruluğu, ≥2 sn duraklamalarda F1 0.80; formasyon tetikleyicileri (taç: çizgi dışında oyuncu, korner: köşede oyuncu, aut: kale alanında oyuncu, santra: herkes kendi yarısında).
- **Sahiplik**: Vidal-Codina 2022 (possession zone 0.5–1 m, duel zone 1 m, top yön/hız değişimi ile kontrol), Link & Hoernig 2017; topsuz takım sahipliği: PathCRF (%92.4), ballradar (%85.9), TranSPORTmer.
- **Dış etiketli veri**: Open DFL/IDSSE — 7 maç, 25 Hz, kare bazında `BallPossession` + `BallStatus` (CC BY). Metrica örnek maçları (3 maç, olaylarla H/A/D türetilebilir).
- **Video**: top tespiti için WASB (MIT, futbol ağırlıkları), YOLO + SAHI; ATD topunu homografiyle piksele izdüşürüp **sözde-etiket** olarak kullanmak (113k kare).
- Beklentiler: maç başına ~108 duraklama (ort. 18.7 sn), LaLiga efektif oyun ~53 dk → tam maçta D payı ~%45–50; ~200–350 possession.

## 5. Baseline'lar ve sonuçlar

Public pencere = 17 possession (gürültülü, sadece doğruluk kontrolü). Tam maç tanıları etiketsizdir.
Vekil metrikler: `tagA` Smart Tagging faz-takım uyumu (az possession'ı ödüllendirir, dikkat), `spOK` takımlı duran
topların doğru takım ile yeniden başlaması, `rst` herhangi bir yeniden başlama, `bout` top-dışarı olaylarının D kapsaması.

| Model | macroF1 | spellF1 (public) | Possession (tam maç) | D payı | spOK | rst |
|---|---|---|---|---|---|---|
| baseline_smooth50 (resmi) | 0.621 | 0.326 | 594 | 0.32 | 0.48 | 0.80 |
| v1 kurallar + Viterbi | 0.709 | 0.525* | 214 | 0.41 | 0.52 | 0.81 |
| v2 ayrık canlı/ölü + takım kanıtı | 0.816 | 0.448 | 234 | 0.48 | 0.59 | 0.90 |
| v3 = v2 + broadcast (+ top doldurma, kamera uyuşmazlığı, 1.5 sn tepki gecikmesi) | 0.878 | 0.459 | 221 (113 D segmenti) | 0.47 | 0.70 | 0.92 |
| **v5 = v3 canlı/ölü + DFL'de öğrenilmiş takım olasılıkları (önerilen baseline)** | **0.901** | **0.566** | 261 (119 D segmenti) | 0.50 | **0.82** | **0.93** |

v5 public pencerede: alive/dead F1 0.974, canlı karelerde takım doğruluğu 0.929 (v3: 0.972 / 0.901).
Resmi skorlayıcı (public, 1000 bootstrap): v5 macroF1 0.9010 / spellF1 0.5655 / geçiş-F1 (±1 sn) 0.389, ortalama
gecikme +1.1 kare, Barça topa sahip olma payı hatası 0.003. Public pencerede spellF1 farkları <0.41 gürültü sayılır;
macroF1 ve tüm vekil metriklerdeki tutarlı iyileşme daha güvenilir sinyaldir.

v3 public pencerede: alive/dead F1 **0.972**, takım doğruluğu 0.901 (ikisi de canlı karelerde). Tam maçta 113 ölü
top segmenti ve %46.8 D payı, literatür (~108 duraklama, ~%45–50) ve DFL maçları (maç başına ~113 segment, %45–48)
ile uyumlu.

\* v1'in ceza ayarı public pencerede seçildi — iyimser.

## 6. Yol haritası (öncelik sırasıyla)

1. **Video top tespiti** — `colab/ball_detector_colab.py`: ATD topunu homografiyle piksele izdüşürüp ~113k
   sözde-etiket, YOLO11s ince ayar (640 karo), ATD'de topun kayıp olduğu karelerde tespit → sahaya geri izdüşüm.
   Hedef: top kapsaması %59 → %80+; takım kararı en çok bundan kazanır (oracle: takım mükemmel olsa public
   spellF1 0.46 → 0.89). Yerel GPU şu an başka bir işle dolu → Colab.
2. **Elle etiketlenmiş doğrulama seti (20–30 dk)** — kurallara uygun. `tools/labeler.html` (tarayıcıda lokal video,
   klavye kısayolları, CSV) + `tools/transitions_to_gt.py` → resmi skorlayıcıda `--gt <yol>`. 17 possession'lık
   public pencere yerine ~100+ possession ile model seçimi.
3. **Düdük tespiti** — önceden eğitilmiş ses olayı modeli (PANNs/YAMNet "Whistle"); basit bant detektörü tribün
   ıslıklarından dolayı yetersiz kaldı.
4. **Ölü-top modelini BAR-ATH'a uyarlamak** — DFL simülasyonunu (topun saha dışındaki konumu, top kaybolma
   örüntüsü) iyileştirmek veya v3 kurallarıyla öğrenilen modeli birleştirmek; Smart Tagging duran top
   pencerelerini ek kanıt olarak eklemek.
5. **Metrica örnek maçları** (3 maç, olaylardan H/A/D türetilebilir) — Metrica konvansiyonuna daha yakın ek
   eğitim/doğrulama; takım-geçiş cezasını onlarla ayarlamak.
6. Broadcast videosu (13 GB) — Drive kotası açılınca; tekrar/yakın çekim tespiti zaten homografi geçerliliğiyle
   yapılıyor, opsiyonel.

## 7. DFL deneyleri (dış veri ile öğrenme)

**Kurulum.** Open DFL/IDSSE (7 maç, CC BY 4.0, DOI 10.6084/m9.figshare.28196177) → BAR-ATH ATD'ye
benzetecek bozulma (`src/sim.py`): top canlıda ~%12–15, ölüde ~%70 kayıp (uzun boşluklar, havadaki topta daha
olası), havadaki top için yan kamera paralaksı, oyuncu izlerinde ~%6 parçalanma + gürültü. Aynı öznitelikler
(`src/mlfeat.py`: çok ölçekli/kaydırmalı oyuncu temposu, formasyon tetikleyicileri, top öznitelikleri, takım
perspektifli kontrol/şekil) → LightGBM ölü-top modeli + takım-perspektifli H/A modeli (yer değiştirme
augmentasyonu) → maç-dışarıda-bırak CV → min-süreli Viterbi.

**DFL yapısı.** Maç başına ~113 ölü top segmenti, %45–48 D — BAR-ATH v3 tahminiyle neredeyse aynı (113, %46.8).
Ancak DFL'nin sahiplik bayrağı daha sık dönüyor: maç başına ~456 possession, medyan 3.6 sn (public GT medyan
7.6 sn, ort. 14.2 sn) → **konvansiyon farkı**; takım-geçiş cezası DFL'den doğrudan aktarılmamalı.

**CV (bozulmuş DFL, 7 kat, 3,061 possession).**

| | değer |
|---|---|
| kare doğruluğu | 0.89–0.92 |
| canlı/ölü doğruluğu | 0.96–0.97 (Lang 2022: 0.92) |
| takım doğruluğu (canlı) | 0.85–0.91 |
| spellF1 (DFL konvansiyonu) | 0.549 (en iyi: pen_team=4, pen_dead=32) |
| sınır yakalama (snapping) | etkisi ±0.003 → kullanılmadı |

**BAR-ATH'a aktarım.** DFL modeli takım kararında belirgin daha iyi (public canlı karelerde takım doğruluğu
0.937–0.940 vs v3 0.901), fakat ölü-top tespitinde alan farkı yüzünden zayıf (alive/dead F1 0.880 vs 0.972).
→ **v5 hibrit**: canlı/ölü için v3 (kural + broadcast), H/A için DFL olasılıkları; doğru olasılıksal birleştirme
`S[H]=a/2+log p_H` (ilk denemedeki `±e/2` formu canlı durumlara bedava bonus verip D payını bozuyordu).
