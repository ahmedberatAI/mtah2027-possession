# PFF eğitimi ve v7 karşılaştırması — 24 Ekim 2026

> Sonraki çalışma tamamlandı: kaynakları dengeli TCN/BiGRU ve eğitim içi
> maçlarla ayar seçimi. [Sıra modellerinin sonuçları](../sequence_pff/SONUC.md).

**64 maç 25 fps'e hizalandı, kontrollü possession etiketleri üretildi ve üç model
eğitildi. Ana DFL+PFF-controlled adayının kazancı SkillCorner'da küçük ama olumlu;
Metrica'da mevcut v7 seviyesinde. v7'yi genel olarak geçtiği gösterilmedi.**

## Ana sonuç

Tablo, aynı dış test düzeneğinde sabit öğrenilmiş ölü-top kanalı ve B2 ile
maç ortalaması spellF1'dir. Yalnız takım modeli değişir.

| Dış test / etiket | v7 | DFL+PFF kontrollü | Fark | Eşli %95 aralık | Artan maç |
|---|---:|---:|---:|---|---:|
| Metrica kontrollü | 0.8147 | 0.8144 | −0.0003 | [−0.0077, +0.0088] | 1/3 |
| Metrica olay | 0.6155 | 0.6100 | −0.0056 | [−0.0150, +0.0092] | 1/3 |
| SkillCorner oyuncu possession | 0.5829 | 0.5913 | +0.0084 | [+0.0023, +0.0144] | 14/20 |
| SkillCorner faz | 0.5816 | 0.5897 | +0.0081 | [+0.0018, +0.0142] | 14/20 |

Metrica yalnız üç bağımsız maçtır; üçer bozulma tohumu maç içinde ortalanmıştır.
Aralıklar maç bootstrap'ıdır, eğitim tekrarları belirsizliğini içermez. Oracle-D
tanısında da SkillCorner kazancı korunur: pp +0.0061, phase +0.0069. Dolayısıyla
gözlenen kazanç yalnız ölü-top modelinin hatalarıyla açıklanmıyor.

Kare düzeyinde takım doğruluğu Metrica-ctrl %93,97→%94,40, SkillCorner-pp
%90,41→%91,56. Kare doğruluğunun artması Metrica possession bölümü skorunu
artırmaya yetmedi; sınır/zamanlama ve etiket tanımı önemini koruyor.

## Veri miktarı mı, kontrollü etiket mi?

| Eğitim | Metrica ctrl | Metrica events | SC pp | SC phase |
|---|---:|---:|---:|---:|
| DFL7 — v7 | 0.8147 | 0.6155 | 0.5829 | 0.5816 |
| DFL7 + PFF64 aktör-takım vekili | 0.8043 | 0.6152 | 0.5888 | 0.5853 |
| DFL7 + PFF64 kontrollü | 0.8144 | 0.6100 | 0.5913 | 0.5897 |
| Yalnız PFF64 kontrollü | 0.8123 | 0.6022 | 0.5891 | 0.5856 |

Kontrollü etiketlerin aynı karelerdeki aktör-takım vekiline göre katkısı
Metrica-ctrl'de **+0.0101** (3/3 maç), SC-pp'de **+0.0025**, SC-phase'de
**+0.0044**. SC-pp etiket katkısının aralığı [−0.0005,+0.0060] olduğundan
bu küçük farkı kesin kazanım sayamayız. Metrica-events ise **−0.0053**:
etiket tanımını değiştirmenin etkisi değerlendirme konvansiyonuna bağlı.
Eşli ayrıntılar [label_ablation.json](label_ablation.json) içinde.

**Karar:** PFF kontrollü model aday olarak saklandı; v7 değiştirilmedi.
10,65 kat örnek bu sabit LightGBM düzeninde büyük sıçrama getirmedi. Bu sonuç,
64 maçla bir TCN/transformer'ın yararsız olacağını göstermez; sıra modelleri
bu çalışmada eğitilmedi. Sonraki deneyin kaynak dengelemesini ve etiket
belirsizliğini eğitim içi ayrılmış maçlarda sınaması anlamlı olur. Bu dış
testleri tekrar tekrar ayar seçmek için kullanmamak gerekir.

## Üretilen veri ve modeller

- **64 maç, 134 devre, 9.884.621 adet 25 Hz kare.** En kötü devre hizalama
  artık-hatası p95=32,86 ms; bütün devreler kontrolden geçti.
- Ortak geçerli canlı eğitim verisi **59,33 saat**. Stride 2 ve iki takım
  perspektifiyle DFL+PFF **5.892.500 satır**, aynı **172 özellik / 400 ağaç**.
- Sekme, başarısız kontrol ve kontrolle sonuçlanmayan düello takım değiştirmez.
  D ve X ayrı tutulur. Aktör-takım vekiline göre ortak canlı karelerin %2,18'i
  değişti; bilinen canlı takım geçişleri 15.249→12.473.
- **Smoothed top kullanılmadı:** sağlayıcının olaylardan oluşturduğu bu alan
  etiket sızıntısı riski taşıyor. Yalnız ham tracking kullanıldı.
- 134 etiket dosyası: `cache/pff25/raw_control_v2/labels/*.csv.gz`.
- Ana model: `cache/pff_experiment/raw_control_v2/dfl_pff_controlled.txt`.
  Tanı modelleri aynı klasörde `pff_controlled.txt`, `dfl_pff_native.txt`.

Bu etiketler yarışma tanımına yaklaşan muhafazakâr vekillerdir; Metrica'nın
özel GT üretim mekanizması hâlâ doğrulanmış değil. BAR-ATH sonucu ölçülmedi.
Ham veri, modeller ve büyük ara dosyalar `D:\barca` altında kaldı.

17 birim testi geçti; 25 Hz aralıklar, tekrarlar, kamera kesmeleri, sekme/düello,
yeniden başlama ve Smoothed topun dışlanması denetlendi. 32 değerlendirme
hücresi tamamlandı; 24 v7'ye karşı eşli karşılaştırma üretildi.

[Tüm skorlar ve komutlar](RESULTS.md) · [Yöntem ve sınırlamalar](METODOLOJI.md) ·
[Veri sayımları](data_summary.json) · [Deney protokolü](design.json)
