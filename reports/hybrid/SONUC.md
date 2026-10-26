# Hata analizi + BiGRU / dengeli LightGBM — 25 Ekim 2026

**İç doğrulama LightGBM eklemeyi seçmedi. En iyi ağırlık %100 BiGRU, %0 LGB.**
Mevcut üç-seed BiGRU ensemble korunuyor. Bu sonuç yeni bir hibrit kazancı
değildir; denenen sabit olasılık karışımı bu seçim ölçütünde fayda vermedi.

## Eğitim içindeki ağırlık seçimi

Önceki ayrım korunarak 5 DFL + 52 PFF üzerinde üç BiGRU ve üç dengeli
LightGBM yeniden eğitildi. Ayrılmış 2 DFL + 12 PFF maçı altı modelin hiçbirinin
eğitimine veya BiGRU standardizasyonuna girmedi. Her ailenin üç seed olasılığı
eşit ortalandı; yalnız aşağıdaki beş aile ağırlığı denendi.

| BiGRU / LGB ağırlığı | DFL doğrulama | PFF doğrulama | Kaynaklara eşit ağırlıklı spellF1 |
|---|---:|---:|---:|
| 0 / 100 | 0.5823 | 0.6944 | 0.6383 |
| 25 / 75 | 0.5946 | 0.7008 | 0.6477 |
| 50 / 50 | 0.6306 | 0.7134 | 0.6720 |
| 75 / 25 | 0.6408 | 0.7248 | 0.6828 |
| **100 / 0** | **0.6508** | **0.7387** | **0.6947** |

Her iki kaynağın ortalamasında da saf BiGRU en iyi. Seçim dış değerlendirmeden
önce `selection.json` ile donduruldu; Metrica avantajını kurtarmak için sonradan
ağırlık değiştirilmedi. İç ölçüt önceki mimari seçimiyle aynı: gerçek ölü-top
maskesi ve sabit B2 ile DFL native / PFF kontrollü etiketleri üzerinde spellF1.

## Dış test

Seçilen tek ağırlık, önceki tüm 7 DFL + 64 PFF ile eğitilmiş modellere
uygulandı. Her model için aynı girdiler, öğrenilmiş ölü-top kanalı ve B2 kullanıldı.
LGB sütunu, üç dengeli LGB seed'inin olasılık ensemble'ıdır; ayrı seed skorlarının
ortalaması değildir.

| Dış test | v7 | Dengeli LGB ensemble | Mevcut BiGRU ensemble | Seçilen sonuç |
|---|---:|---:|---:|---:|
| Metrica kontrollü | 0.8147 | 0.8160 | 0.8071 | **0.8071** |
| Metrica olay | 0.6155 | 0.6160 | 0.6452 | **0.6452** |
| SkillCorner oyuncu possession | 0.5829 | 0.5926 | 0.6336 | **0.6336** |
| SkillCorner faz | 0.5816 | 0.5917 | 0.6221 | **0.6221** |

Seçilen sonuç mevcut ensemble ile kare bazında aynı olasılıklardan gelir;
ek kazanç **0.0000**. SkillCorner'daki önceki v7 kazancı korunur: pp +0.0507
(%95 [+0.0436,+0.0586]), faz +0.0406, her ikisinde 20/20 maç. Metrica
kontrollüde v7'ye −0.0076, olayda +0.0297; üç maçlık aralıklar sıfırı kapsar.

**%75/%25 karışımın dış skoru bu deneyde ölçülmedi.** İç ölçüte göre kaybeden
ağırlıkları dış sette deneyip sonradan kazanan seçmek yapılmadı. Bu nedenle
"bütün karışımlar dış sette kötü" sonucu da çıkarılamaz.

## Hata analizi ne gösterdi?

Tanısal sayımlar IoU=0.5'tir; ana spellF1 beş eşik ortalamasıdır.

**Metrica kontrollü:** BiGRU daha fazla gerçek possession yakalıyor, fakat
daha çok eşleşmeyen tahmin üretiyor. Tüm GT possession'ları kaçırma oranı
v7'de %17.60, BiGRU'da %16.93, LGB'de %17.99. Buna karşılık eşleşmeyen
tahmin oranı v7'de %17.01, BiGRU'da %19.03, LGB'de %16.54. GT possession'ı
birden fazla aynı-takım tahminiyle bölme göstergesi BiGRU'da %4.91,
v7'de %3.42, LGB'de %3.37. BiGRU'nun kaybı yalnız "kısa possession'ları
göremiyor" diye açıklanamaz; ekstra parçalama ve yanlış/eşleşmeyen tahminler
de önemlidir.

**SkillCorner-pp:** v7 → BiGRU karşılaştırmasında:

- 4 saniyeden kısa GT possession'ları kaçırma: **%82.59 → %75.70**.
- Tüm possession'ları kaçırma: **%42.63 → %37.40**.
- Eşleşmeyen tahmin oranı: **%38.82 → %33.33**.
- Bir tahminin birden fazla aynı-takım GT possession ile örtüşme göstergesi:
  **%9.79 → %7.91**.
- ±1 saniyede geçiş yakalama recall: **%39.01 → %44.86**.

Kazanç yalnız zaman kaydırması değildir. Eşleşebilen geçişlerde ortalama
mutlak zaman hatası v7 ve BiGRU'da yaklaşık **0.35 sn**, ama BiGRU daha fazla
geçiş yakalıyor. Bu gecikme yalnız eşleşen geçişlere koşulludur.

Çok kısa possession'lar hâlâ sorun: SkillCorner-pp'de BiGRU, 2 saniyeden
kısaların yaklaşık %93.6'sını IoU=0.5'te eşleştiremiyor.

**Tamamlayıcılık var ama otomatik kazanç değil.** SkillCorner-pp'de yalnız
LGB'nin eşleştirdiği 149 GT possession, yalnız BiGRU'nun eşleştirdiği 456
possession var. İkisi de 1.912 possession'ı kaçırıyor. Hangi anlarda hangi
modelin doğru olduğunu bilmeden olasılık ortalaması almak, yalnız LGB'nin
kurtardığı örnekleri seçerek eklemek anlamına gelmiyor. Bu beş ağırlıklı
deneyde karışımın iç doğrulama getirisi olmadı; bunun bütün olası koşullu
birleştirme yöntemlerini dışladığı iddia edilmiyor.

## Karar ve kısıtlar

Mevcut BiGRU ensemble adayını koru; bu deneyin seçtiği karışımla değiştirmek
zaten aynı çıktıyı verir. Metrica kontrollü kaybı çözülmüş değildir. Sonraki
ayrı deney için bulgu, etiket tanımı ve gereksiz bölünme davranışının
incelenmesini destekler; yalnız sabit gecikme düzeltmesi yeterli görünmüyor.

İç doğrulama aynı zamanda önceki mimari seçimi için de kullanılmıştı. DFL
native etiketleri kontrollü possession ile aynı değildir; oracle ölü-top
seçim koşulu dış learned-D koşulundan farklıdır. Dış setler de önceki
deneylerde incelendi. Bunlar sonucun genellenebilirliğini sınırlar; bu turda
dış ağırlık seçimi yapılmadı. Metrica yalnız üç maçtır.

Altı modelin hash'i, eğitim/doğrulama ayrımı, yalnız eğitimden yeniden
hesaplanan scaler, LGB satır hizası ve 50/50 kaynak ağırlıkları bağımsız
audit ile kontrol edildi. Seçim tekrar hesaplandı, 32 dış hücre doğrulandı;
v7 ve mevcut ensemble'ın önceki maç skorları birebir yeniden üretildi.
Altı yeni birim testi geçti.

BAR-ATH bu deneyde yeniden puanlanmadı; mevcut **0.7724** açık bölüm skoru ve
submission CSV'si değişmedi. Gizli tam maç skoru hâlâ bilinmiyor.

[Tüm skorlar](RESULTS.md) · [Hata tabloları](HATA_ANALIZI.md) ·
[Protokol ve komutlar](METODOLOJI.md) · [Bağımsız audit](independent_audit.json)
