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

## Dış veri
- IDSSE/Open DFL (7 maç, CC BY): ölü-top istatistikleri çok benzer; possession bayrağı GT'den ~2× sık dönüyor.
- Metrica örnek (3 maç, kaynak belirtilmeli), SkillCorner open data (20 maç, MIT, henüz kullanılmadı).
- WASB futbol ağırlıkları (MIT) — top tespiti için.

## Açık sorular
- pen_team 8 mi 20 mi? → elle etiketlenmiş doğrulama seti.
- Top tespiti: hazır WASB yetmiyor; ATD sözde-etiketleriyle ince ayar (doğal çözünürlükte karolar) Colab'da yapılacak.
