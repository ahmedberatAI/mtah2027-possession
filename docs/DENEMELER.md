# Deneme günlüğü — ne denendi, ne işe yaradı, ne yaramadı

Amaç: aynı hataları tekrarlamamak. Her yeni denemede buraya bir satır ekleyin (tarih, ne, sonuç, karar).
Skorlar: public = ilk 5 dk (17 possession, spellF1 bootstrap aralığı ~0.27–0.62 → tek başına karar verdirmez).
"Vekiller" = `src/proxy.py` (Smart Tagging duran top / takım uyumu, top-dışarı kapsaması) + tam maç sayıları.

## Kalıcı kurallar / tuzaklar
- **Veri hiçbir bulut LLM'e gösterilmez** (T&C). Scriptler sadece toplu istatistik basar; video karesi, ham satır,
  ses, ham veri grafiği yok. Analiz raporları Artifact olarak yayınlanmaz.
- **Public pencerede parametre ayarlamayın.** Resmi örnek: smooth125 public'te kazanıyor, tam maçta kaybediyor.
  Bizde de v1'de ceza seçimi public'e göre yapılınca iyimser çıktı (0.525*).
- **Hız kanalı m/s** (XML "kph" diyor). **Top hız kanalı çöp** (corr −0.10) → konumdan türet.
- **ATD frame_id − 1 = video karesi.** Kendi parse'ımızla resmi baseline'lar birebir üretildi (hizalama doğru).
- **Broadcast hizası:** 1. yarı `tactical = broadcast + 4014`, 2. yarı `+ 26920` (top eşleşmesiyle; takım
  merkezleriyle denendi → başarısız, çünkü broadcast sadece görünen oyuncuları içeriyor).
- **Simülasyonu simülasyon içi CV ile doğrulamayın.** DFL-CV / Metrica eğitim ile testi aynı bozulmadan
  geçirir; bozulma gerçekçi değilse (V2'deki her-kare gürültü gibi) bunu göremez. Her yeni bozulmayı
  `src/exp_advval.py` (BAR-ATH'a karşı adversarial validation) ve kinematik istatistiklerle kontrol edin.
- **Skor birleştirmede ±e/2 hatası:** `S[H]=a/2+e/2, S[A]=a/2−e/2` en iyi canlı duruma |e|/2 bedava bonus verir,
  D payını bozar. Doğrusu `S[H]=a/2+log p_H`, `S[A]=a/2+log p_A` (log p ≤ 0).
- **Takım kanıtı sıfır-toplamlı olmalı:** v1'de takım kanıtı canlı durumları da ödüllendirince duran topta
  topun yanında duran oyuncu "canlı H" ürettirdi.
- Google Drive "Kota aşıldı": klasörü **zip olarak indirmek** çalıştı; tek dosya / "kopya oluştur" videoda çalışmadı.
  Yerleşik tarayıcı paneli kapatılınca indirme iptal oluyor.

## Modeller (kronolojik)

| # | Ne | Public macroF1 / spellF1 | Tam maç | Karar |
|---|---|---|---|---|
| resmi | en yakın oyuncu + çoğunluk oyu | 0.621 / 0.326 | 594 poss | taban |
| v1 | kural kanıtı + min-süreli Viterbi | 0.709 / 0.525* | 214 | ceza public'e göre seçildi → güvenilmez |
| v2 | canlı/ölü (a) ve takım (e) ayrı; sıfır-toplamlı takım | 0.816 / 0.448 | 234, %48 D | ✅ yapı doğru |
| v3 | v2 + broadcast geçerliliği/canlılığı + broadcast topu ile boşluk doldurma + kamera uyuşmazlığı ağırlığı + hız penceresi 1.5 sn ileri | 0.878 / 0.459 | 221, 113 D seg | ✅ canlı/ölü F1 0.972 |
| v4 | olay tabanlı takım takibi (kontrol epizodları, teyit kuralları) | 0.84 / 0.39–0.47 | 206–266 | ❌ v3'ü geçmedi, bırakıldı |
| DFL-saf | DFL'de eğitilen LightGBM ölü-top + takım, BAR-ATH'a doğrudan | 0.74–0.75 / 0.37–0.42 | %54 D | ❌ ölü-top alan farkı (F1 0.880) |
| **v5** | v3 canlı/ölü + DFL LightGBM takım olasılıkları, pen_team 8 | **0.901 / 0.566** | 261, 119 D seg, %50 D | ✅ **mevcut baseline** |
| v6 | v5 + düdük şablonu (D başlangıcını öne çekme) | 0.908 / 0.566 | aynı | ✅ küçük, düşük riskli; henüz varsayılan değil |
| **v7** (aday) | v3 canlı/ölü + **sim V3** ile eğitilen takım modeli (own_half düzeltmesi + baskı altında kontrol C6 + dizilim/şekil/hareket c145) + **B2 yumuşak min. dekoder**, pen 12 | **0.924 / 0.652** [0.43–0.81] | 227 poss (~4.3/dk), 118 D seg | ✅ simülasyonda eşli +0.016 (DFL-trk 7/7, Metrica-ctrl 9/9); `submissions/candidate_v7.csv`; varsayılan yapılması kullanıcı onayında |

## Tek tek denemeler

| Deneme | Sonuç | Karar |
|---|---|---|
| "Son yakın takım" kuralları (R 0.75–3 m, k 1–25 kare) | takım doğruluğu ~%85–87'de tavan | tek başına yetmez |
| Oracle (public): canlı/ölü mükemmel / takım mükemmel | 0.461→0.473 / 0.461→0.889 | darboğaz takım kararı |
| Hız penceresini ileri kaydırma (0–50 kare) | alive/dead F1 0.962→0.974 (50); 37 seçildi (fiziksel gerekçe) | ✅ |
| Kamera uyuşmazlığı (>2.5 m) ile takım kanıtını zayıflatma | macroF1 +0.012 | ✅ v3'te |
| Basit bant-tonalite düdük detektörü (`whistle.py`) | tüm maçta lift ~1 (tribün ıslıkları) | ❌ |
| PANNs CNN14 AudioSet "Whistle" (3 varyant) | lift 1.0–1.9, santraları kaçırıyor | ❌ |
| Santra düdüğü şablonu ile arama (`whistle_ref.py`) | D başl. ±1 sn lift 3.2 (p=4e-4); santraları 5./9. sırada buluyor | ✅ v6 |
| Düdük bonusunu her düdüğe vermek | D segmenti 119→160, spellF1 düşüyor | ❌ sadece "v5 geç kaldıysa" |
| Sınır yakalama (snapping) DFL'de | ±0.003 | ❌ gereksiz |
| DFL bozulma simülasyonu v1 (canlı %12, ölü %70 kayıp) | çalışıyor; ölü boşluklar canlıya taşıyordu | ✅ `sim.V2` (opt-in): boşluklar yeniden başlamada kesiliyor (canlı %6.6, ölü ~%70); varsayılan değişmedi; pen_team önerisi (20) değişmedi |
| Metrica örnek maçları kalibrasyonu | "kontrollü possession" konvansiyonu GT'ye uyuyor; pen_team 20 öneriliyor (8 public'te daha iyi) | ⏳ elle etiketle karar verilecek |
| Kamera merkezi top vekili | medyan 23.6 m hata | ❌ |
| Takım merkezleriyle broadcast hizalama | tutarsız | ❌ → top ile çözüldü |
| WASB (futbol, hazır) tam kare 1920×1080→512×288, sıfır-atış (600 kare) | 40 px içinde **%0**, medyan hata 548 px; top 2–3 px'e iniyor | ❌ tam kareyi küçültme |
| WASB hazır, topu içeren **doğal çözünürlükte** 512×288 kırpıntı (oracle konum) | medyan 33 px, 10 px içinde %28; çok yanlış pozitif (kafa/tozluk) | ❌ ince ayar şart |
| WASB vs ATD zaman/uzay ofseti | en iyi 0/+1 kare (hizalı); WASB merkezi ATD'nin ~5–7 px **yukarısında** (ATD = topun zemin noktası) | sözde-etiket y'sini ~6 px yukarı kaydır |
| CPU hızı (24 çekirdek) | WASB çıkarımı ~0.16 sn/üçlü → tüm maç ~3 sa (örtüşmesiz üçlü, tek ölçek); ince ayar CPU'da pratik değil | GPU (Colab) |

| sim.V2 ile yeniden eğitilen takım modeli + v5 (pen_team 8/12/20) | public spellF1 0.542/0.552/**0.578**, macroF1 ~0.89–0.90; tam maç 277/258/236 poss; v5 ile kare uyumu %98 | aday: **v2 model + pen 20** (Metrica'nın önerisiyle tutarlı, public'i de cezalandırmıyor); karar elle etiketle |
| sim.V2 ile ölü-top LightGBM | DFL CV'de iyi (0.980), BAR-ATH public'te hâlâ zayıf (0.766) | ❌ kurallar kalıyor |

| YOLO11s ince ayar (ATD sözde-etiketleri, 640 doğal karolar, 16k+5.6k kırpıntı) | 18 epoch'ta val mAP50 ~0.08'de takıldı | ❌ durduruldu (ağırlık `cache/ball_yolo/train/weights/last.pt`) |
| **Top izinin değeri (GPU'suz, DFL CV, eşli)**: oyun içi top eksikliği %6.5 → %3.3 → %0 | takım doğruluğu 0.8914 → 0.8933 → 0.8942; spellF1 (pen20) 0.558 → 0.563 → 0.568; ayarlı 0.587 → 0.597 → 0.606 | ❌ mükemmel top bile <+0.02 → **video top tespiti ertelendi**; darboğaz top izi değil |
| ATD kısa boşluk doldurma şüphesi | 27–128 karelik boşluk sadece 1; ~%3 görünür kare olağandışı pürüzsüz, orada iki kamera farkı 1.22 m (normal 0.74 m) | ATD kısa boşlukları tahminle dolduruyor; sözde-etiket kullanılırsa pürüzsüz koşular + boşluk kenarları çıkarılmalı |

| Resmi skorlayıcı, public 5 dk, 1000 bootstrap (21 Eki) | v6 0.908/0.566 [0.36–0.76]; v5 0.901/0.566 [0.36–0.76]; v5 pen20 0.885/0.448 [0.21–0.65]; v3 0.878/0.459; resmi 0.621/0.326 [0.21–0.45] | v5/v6 resmi baseline'dan net iyi; v5 / v5-pen20 / v6 farkları güven aralığı içinde |
| Dikkat sinyalleri (oyuncu hareketi topa dönük mü) DFL'de ölü top ayrımı | AUC: hız 0.929 → hız+dikkat 0.944 (doğrusal), 4 pencere + LightGBM 0.974 (hız tek başına 0.950) | ✅ umut verici; v5'e eklenmesi bekliyor |

| **Ortak CPU düzeneği** `src/bench.py` (21 Eki): DFL-CV (7 maç), DFL **trk** (takipten kontrollü possession: 240 spell/maç, 4.56/canlı dk), Metrica-ctrl (3 maç × 3 tohum), BAR-ATH public + tam maç | V2 tabanı: DFL-trk 0.756 (pen 48), Metrica-ctrl 0.808 (pen 32), DFL-native 0.592 (pen 4) | ✅ fikirler bununla ölçülüyor (`docs/FIKIRLER.md`) |
| Takım modelini trk (kontrollü) etiketle eğitmek (A1) | kare doğruluğu ↑ (0.903→0.915), spellF1 ≈/↓ (0.756→0.749; Metrica 0.808→0.802) | ❌ tek başına; dekoder cezası aynı işi yapıyor |
| Smart Tagging faz takımı / set piece (F1, F2) | public'te etiket %76 vs v5 %85; çelişkide etiket %25 haklı; set piece başlangıcı restarttan ~5 sn önce | ❌ değersiz / zayıf |
| **Adversarial validation (E1)**: bozulmuş DFL vs BAR-ATH | AUC 1.000 — **simülasyon hatası**: her-kare bağımsız gürültü hızları 2× şişiriyordu (medyan 3.7 vs 1.95 m/s); ayrıca `own_half_frac` NaN hatası | ✅ `sim.V3` (pürüzsüz gürültü), `feat_fix.own_half`; **tüm öğrenilmiş modeller bozuk hızlarla eğitilmişti** |
| sim.V3 ile ölü-top LightGBM | BAR-ATH public ham canlı F1 0.915 (V2: 0.766) | ✅ artık kullanılabilir ikinci sinyal (kurallar 0.974) |
| sim.V3 ile takım modeli + v5 | public pen 8: 0.579 / 0.905 / takım 0.934 (V2 0.529 / 0.898 / 0.925); trk modeli 0.579 / 0.908 / 0.938, 4.22 poss/dk | ✅ aday; DFL/Metrica simülasyon içi olduğu için V2–V3 farkını göremiyor |

| Kurallar + öğrenilmiş ölü-top harmanı (N1, V3) | DFL/Metrica'da kurallar zayıf (broadcast yok; canlı/ölü 0.88 vs 0.98); BAR-ATH public: canlı/ölü F1 0.974→0.965, macroF1 0.901→0.873 (w=0.25); ölü payı eşitlenince de düşüş | ❌ v3 kuralları kalıyor |
| native + trk topluluğu (N3) | DFL-trk +0.005, Metrica-ctrl +0.003, events −0.02 | ≈ |
| Top kaybı çevresine ×3 örnek ağırlığı (N9) | DFL-native +0.018 (7/7), Metrica-events +0.009; DFL-trk −0.003, Metrica-ctrl +0.0025; public macroF1 0.912 | ≈ GT konvansiyonuna yaramıyor |
| `own_half` NaN düzeltmesi (N4) | DFL-trk +0.004, Metrica-ctrl +0.006, public aynı | ✅ ilkeli, tutuluyor |
| **Baskı altında kontrol özellikleri (C6, `src/feat_ctrl.py`)** | eşli: DFL-trk +0.0085 (6/7), DFL-native +0.012 (6/7), Metrica-ctrl +0.006 (7/9), events 0; public pen 8 0.594 / 0.900 | ✅ ilk tutarlı kazanç; birleşime alınacak |
| Public hata analizi (`src/err_spells.py`) | top kaybı spell'lerinin 3/12'si doğru (simülasyonda %66–80); kaçanlar kısa ve iki takım da topa yakın | darboğaz: baskı altındaki kısa possession'lar |

| Dekoder fikirleri B1/B4/B5/B6/G1/G2 (alt ajan) | hepsi ≤ +0.002 | ❌ |
| **B2 yumuşak min. possession süresi** | V3: Metrica-ctrl +0.009 (9/9), DFL-trk +0.013; native/events'te −0.03…−0.05 | ✅ v7'de (kontrollü konvansiyon tercihi) |
| Özellik grupları c1–c5 (alt ajan) | c145 en iyi (+0.007…+0.013, pen 32–48); c2, c3 ≈ 0 | ✅ c145 v7'de |
| Birleşim + B2 (sentez) | DFL-trk +0.016 (7/7), Metrica-ctrl +0.016 (9/9) | ✅ v7 |

| **SkillCorner 20 maç, alan dışı doğrulama** (DFL'de eğitilen modeller, `src/exp_sc.py`) | mutlak ~0.57–0.58 (BAR-ATH public düzeyi); v7 (birleşim+B2) +0.0167 (15/20), c145+B2 +0.0178 (17/20), own_half+C6+B2 +0.0159 (17/20) | ✅ v7 bağımsız veride de doğrulandı |

| SC gerçek D ile (oracle) takım karşılaştırması | v7 0.582 → 0.746 (fark ölü-top modelinin SC'ye taşınamaması); v7 +0.0137 (17/20), c145+B2 +0.0125, own_half+C6 ±0.001 | ✅ v7 en iyi; C6'nın SC katkısı takım tarafında değil |
| SC'yi eğitime katmak (A3b) | SC alan içi +0.025 (18/20), Metrica −0.005, BAR-ATH public aynı | ≈ v7 DFL ile kalıyor |
| B2 parametre taraması (64 ayar) | düz plato; (175, 64, 8) +0.002 düzeyi, BAR-ATH'ta fark yok | ✅ varsayılan (175, 48, 12) kalıyor |
| Dokunuş dedektörü sabit eşikle | DFL 38, SC 24, BAR-ATH 14 dokunuş/canlı dk → alan farkı | ⚠️ eşik dokunuş oranına uyarlanmalı (C7'de yapıldı) |

| Zamansal yığınlama (N20) | DFL-trk −0.005, Metrica −0.006, SC-oracle −0.010 | ❌ |
| Geçişleri kaydırmak (SC, Metrica) | her kaydırma kaybettiriyor; SC'de yeniden başlamayı 0.5 sn kaydırmak −0.012…−0.023 | ❌ zamanlama hizalı |
| BAR-ATH ölü-top yaklaşımı SC'de (N21) | kurallar + 0.5 × öğrenilmiş: SC +0.038 (20/20); BAR-ATH'ta public macroF1 düşüyor | ≈ SC'de taktik kamera yok; BAR-ATH'ta benimsenmedi |

| v7 + hakem düdüğü (N24, `wh_bonus` 30) | public macroF1 0.9244 → 0.9314, canlı/ölü F1 0.9788 → 0.9819, spellF1 aynı; karelerin %0.1'i değişiyor | ✅ küçük, düşük riskli; `candidate_v7w.csv` (yalnız BAR-ATH'ta ölçülebilir) |
| Takım modeli hiperparametreleri (N23: yavaş öğrenme, büyük ağaç, düzenlileştirme, her kare) | hepsi v7 varsayılanının ±0.005'i içinde, tutarlı kazanan yok | ≈ varsayılan kalıyor |

| **Oracle merdiveni + hata bütçesi** (`src/exp_oracle.py`) | kusursuz ölü top +0.022/+0.057/+0.164, restart takımı +0.006…+0.014, zamanlama +0.005…+0.007, kaçan <4 sn possession'lar +0.040…+0.111, sahteler +0.050…+0.086 (DFL-trk/Metrica/SC) | ✅ fikir sırası buna göre |
| Gecikme düzeltmesi (fikir 1) | bulunan değişimlerde medyan +1…+3 kare; resmi medyan 0 | ❌ potansiyel yok |
| El değişimi modeli + zamana bağlı ceza (fikir 2) | AUC 0.88–0.95 ama spellF1 +0.001…+0.003 (iyimser seçimle) | ≈ |
| Kısa possession doğrulama (fikir 3) | ekleme adaylarının %4–9'u gerçek, eklemek kaybettiriyor; silme ±0.003 | ≈/❌ |
| Restart spell'lerini B2'den muaf tutmak (fikir 4 yan dalı) | −0.001…−0.010 | ❌ |
| Görünürlüğe göre sıcaklık (7) / konsensüs (10) | ±0.003 / etkisiz | ≈ / ❌ |
| BiGRU takım modeli, Colab T4 (11) | gerçek D ile −0.04…−0.10 | ❌ |
| sim.V4 ön yumuşatma (N26) | SC +0.004, Metrica −0.004, BAR-ATH yapışkanlık aynı | ≈ |
| **ATD uydurma top karelerini temizlemek (8)** | iki kamera farkı 1.66 m vs 0.75 m; public 0.652 → 0.696, takım 0.948 → 0.968; SC'de dış doğrulama yok | ✅ seçenek (`clean_ball`), varsayılan değil |

## Dış veri
- IDSSE/Open DFL (7 maç, CC BY): ölü-top istatistikleri çok benzer; possession bayrağı GT'den ~2× sık dönüyor.
- Metrica örnek (3 maç, kaynak belirtilmeli), SkillCorner open data (20 maç, MIT; tur 4'ten beri doğrulama seti,
  `src/skillcorner.py`; kaynak belirtilmeli).
- WASB futbol ağırlıkları (MIT) — top tespiti için.

## Açık sorular
- ~~pen_team 8 mi 20 mi?~~ → **dış doğrulamayla karara bağlandı (22 Eki, v7 + B2):** GT'ye benzeyen dört dış set
  (DFL-trk, Metrica-ctrl, SkillCorner öğrenilmiş D, SkillCorner gerçek D) üzerinde en kötü durum pişmanlığı
  (en iyiye uzaklık) pen 4: 0.0127, **8: 0.0068**, **12: 0.0016**, 20: 0.0017 → **pen 12** (v7 varsayılanı). Tam maç
  possession hızı pen 12'de 4.26/canlı dk (hedef ~3.8–4.2; simülasyonda en iyi tahmin/GT oranı 0.9–1.0). Public
  pencere pen 4–32 arasında birebir aynı skoru veriyor → karar verdiremez. Kaynakların eskiden farklı değer
  göstermesi konvansiyon farkıydı (DFL native sık döner → düşük ceza; kontrollü konvansiyonlar → yüksek) ve standart
  dekoderin dik eğrisiydi; B2 eğriyi düzleştirdi.
- Top tespiti: hazır WASB yetmiyor; ATD sözde-etiketleriyle ince ayar (doğal çözünürlükte karolar) Colab'da yapılacak.
