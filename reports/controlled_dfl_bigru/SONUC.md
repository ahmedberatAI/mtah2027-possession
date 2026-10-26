# DFL kontrollü etiketli BiGRU — 25 Ekim 2026

**Metrica'da bölünme azaldı; fakat daha fazla possession birleşti ve dış
performansın geneli geriledi. Mevcut BiGRU ensemble korunmalı.**

## Dış sonuçlar

Her iki sütun da aynı üç seed'in eşit olasılık ensemble'ıdır. Özellikler,
PFF etiketleri, 50:50 DFL/PFF dengesi, scaler, ağ ayarı, epoch sayısı,
ölü-top kanalı ve B2 aynı; yalnız DFL eğitim etiketleri farklıdır.

| Test | Mevcut ensemble | Kontrollü DFL | Fark | Kazanılan maç |
|---|---:|---:|---:|---:|
| Metrica kontrollü | 0.8071 | **0.8176** | +0.0104 | 2/3 |
| Metrica olay | **0.6452** | 0.5771 | −0.0681 | 0/3 |
| SkillCorner oyuncu possession | **0.6336** | 0.5778 | −0.0558 | 0/20 |
| SkillCorner faz | **0.6221** | 0.5977 | −0.0244 | 3/20 |

Metrica kontrollü farkının %95 eşli maç-bootstrap aralığı
**[−0.0065,+0.0254]**; yalnız üç maç nedeniyle artış belirsiz.
SkillCorner-pp farkı **[−0.0656,−0.0463]**, faz farkı
**[−0.0344,−0.0138]**: bu sette düşüş tutarlı.

## Metrica'daki bölünmeler azaldı mı?

Evet, tanısal bölünme göstergesi üç maçın tamamında azaldı. IoU=.5 hata
incelemesinde, Metrica kontrollü etiketleri için:

| Ölçüm | Mevcut ensemble | Kontrollü DFL |
|---|---:|---:|
| Bir GT possession'ın birden fazla aynı-takım tahminiyle örtüşmesi | %4.91 | **%2.07** |
| Eşleşmeyen tahmin oranı | %19.03 | **%12.70** |
| Bir tahminin birden fazla aynı-takım GT possession ile örtüşmesi | **%3.23** | %5.31 |
| Tüm GT possession'ları kaçırma | **%16.93** | %20.97 |
| 4 sn'den kısa possession'ları kaçırma | **%53.99** | %63.38 |

Bölünme göstergesinde yaklaşık **%58 göreli azalma** var. Buna karşılık
birleştirme ve kaçırma üç maçın tamamında arttı. Dolayısıyla modelin daha
tutucu davranması bazı yanlış bölünmeleri giderirken gerçek el değiştirmeleri
de siliyor. Bu yorum, aynı anda gözlenen bölme/merge ve kaçırma sonuçlarına
dayanıyor; her tekil hatanın nedenini kesin olarak belirlemiyor.

SkillCorner-pp'de aynı ödünleşme daha ağır: birleştirme göstergesi
**%7.91 → %15.09**, kısa possession kaçırma **%75.70 → %87.10**.
Ölü-top etiketi kusursuz verilse bile SkillCorner-pp spellF1
**0.7881 → 0.7332** düşüyor. Dolayısıyla sorun yalnız ölü-top tahminine
bağlanamıyor; takım/possession kararında da var.

Bu oranlar tanısal IoU=.5 incelemesidir; ana spellF1 beş IoU eşiğinin
ortalamasıdır. Bölme/birleştirme göstergeleri pozitif zaman örtüşmesine
dayanır; birbirini dışlayan kesin hata sınıfları değildir. Oranlar Metrica
gürültü seedleri maç içinde ortalandıktan sonra sayımlar toplanarak hesaplanır.

## Eğitim ve iç seçim

DFL etiketleri, repodaki mevcut `bench.trk_ctrl` kuralıyla temiz takipten
yeniden türetildi. Önceki cache ile birebir eşitlik doğrulandı. Canlı karelerin
**%6.68'i** değişti; canlı oyun içindeki takım değişimleri **2.351 → 972**.
Ölü-top ve bilinmeyen kareler korunuyor. Bu etiketler resmî insan GT'si değil,
takip verisinden türetilen kontrollü possession vekilidir.

İki ayar üçer seed ile 5 DFL + 52 PFF eğitim maçında eğitildi; 2 DFL +
12 PFF maçı yalnız ayar seçimine ayrıldı. Seçim kontrollü DFL ve kontrollü
PFF etiketlerinde, kaynaklara ve maçlara eşit ağırlıklı oracle-D spellF1 ile yapıldı.

| İç referans / aday | DFL kontrollü | PFF kontrollü | Dengeli ortalama |
|---|---:|---:|---:|
| Eski native DFL modeli, aynı doğrulama etiketlerinde | 0.7861 | **0.7387** | 0.7624 |
| Kontrollü DFL, genişlik 64 | **0.8362** | 0.6963 | **0.7662** |
| Kontrollü DFL, genişlik 96 | 0.8357 | 0.6933 | 0.7645 |

Seçilen ayar mevcut ayarla aynı: 64 genişlik, dropout .15, öğrenme oranı
.001, 11 epoch. Seçim dış değerlendirmeden önce donduruldu; seçilen model
tüm 7 DFL + 64 PFF üzerinde üç seed ile yeniden eğitildi. Böylece dış
karşılaştırma ayar değişimiyle karışmayan bir DFL etiket müdahalesidir.
İç doğrulamada da DFL kazanırken PFF geriliyordu; kaynak ortalamasındaki
küçük artış bu gerilemeyi gizlememeli.

## Karar ve sınırlar

Bu **etiket dönüşümü + sabit B2** birleşimi genel yükseltme olarak
uygulanmamalı. Mevcut ensemble güçlü aday olarak korunuyor. Metrica'daki
bölünmeyi azaltabiliyoruz, ama bu ayarla bedeli daha fazla gerçek possession
kaçırmak ve SkillCorner performansını düşürmek.

Bu deney bütün kontrollü etiket tanımlarının başarısız olduğunu göstermez.
DFL kuralı yalnız yakınlık ve sahiplik süresine dayanır; topa iki oyuncunun
yakın görünmesi tamamlanmış pas kanıtı değildir, 25 sahiplik karesi de zorunlu
olarak ardışık değildir. Ayrıca native eğitimden gelen B2 ayarları sabit
tutuldu; farklı bir iç doğrulama dekoder seçimi bu deneyin kapsamında değildir.

İç ayrım ve dış testler daha önce kullanılmıştır; yeni bağımsız holdout
değillerdir. Metrica yalnız üç maçtır. Bootstrap aralıkları eğitim/model
seçimi belirsizliğini kapsamaz. Dış sonuçlara bakarak ayar değiştirilmedi.

**Doğrulama:** 9 eğitim, 24 dış skor hücresi, eğitim/doğrulama ayrımı,
yalnız eğitimden hesaplanan scaler ve native sabit-ayar kontrolü denetlendi.
Dört yeni birim testi geçti. Önceki ensemble'ın maç skorları birebir
yeniden üretildi. BAR-ATH bu turda ölçülmedi, mevcut submission değiştirilmedi.

[Tüm skorlar ve aralıklar](RESULTS.md) · [Hata oranları](ERRORS.md) ·
[Hata farkları](error_contrasts.json) · [Protokol](METODOLOJI.md) ·
[Bağımsız kontrol](independent_audit.json)
