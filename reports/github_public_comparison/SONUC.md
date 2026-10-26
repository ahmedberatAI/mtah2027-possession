# GitHub sonuçlarına göre dürüst değerlendirme — 25 Ekim 2026

**Yayımlanmış başlangıç modellerine göre güçlü bir yöntem; yarışma sıralaması
ve gizli tam maç için henüz kanıt yok.** Aynı açık 7.500 kareyi, aynı resmî
skorlayıcıyla yeniden ölçtük. Modeller, ağırlıklar ve dekoder değişmedi.

Resmî repo HEAD'i `f8f2629ff1e0985f81d7e59d8413678514ee1530`, proje repo HEAD'i
`681a731517bcb0ee1399a899afaba52189835eed`; uzaktaki HEAD'lerin yerel sürümlerle
aynı olduğu 25 Ekim'de doğrulandı. Dört resmî baseline, önceki yedi model CSV'si
ve ensemble aynı resmî CLI çağrısında, public GT ve 1.000 bootstrap ile ölçüldü.

## Aynı veri ve metrikler

| Model | spellF1 | macroF1 | transF1 ±1 saniye | Tahmin possession |
|---|---:|---:|---:|---:|
| Resmî raw | 0.0378 | 0.5972 | 0.1410 | 131 |
| Resmî smooth25 | 0.2868 | 0.6200 | 0.2034 | 36 |
| Resmî smooth50 | 0.3256 | 0.6207 | 0.0816 | 26 |
| Resmî smooth125 | 0.4242 | 0.5812 | 0.1667 | 16 |
| v7 | 0.6519 | 0.9244 | 0.5455 | 10 |
| **BiGRU ensemble** | **0.7724** | **0.9433** | **0.6286** | **12** |

GT'de 17 possession var. spellF1, possession aralıklarının IoU 0.3/0.4/0.5/0.6/0.7
eşiklerinde hesaplanan F1 ortalaması; macroF1 kare bazında H/A/D F1 ortalaması.
transF1, burada aynı hedef etikete giden geçişlerin ±25 kare / ±1 saniyede
eşleşmesidir. Bunlar farklı hata türlerini ölçer; spellF1 yarışmanın ana metriğidir.

Ensemble, en iyi public resmî baseline'a **+0.3482 spellF1**, v7'ye **+0.1206**
ekliyor. macroF1 ve geçiş F1 da yükseldi; kazanım yalnız tek metrikten ibaret değil.
Bu dört resmî yöntem aynı en-yakın-oyuncu + yumuşatma algoritmasının farklı
pencere boylarıdır; rakip takımların güncel skor listesi değildir.

## Karıştırılmaması gereken skorlar

GitHub'ın öne çıkardığı **0.2433**, smooth50'nin **gizli tam maç** skorudur.
Bizim 0.7724 ile kıyaslanamaz. Aynı modelin public skoru 0.3256'dır. Resmî
örnekte smooth125 public'te 0.4242 ile smooth50'yi geçerken, tam maçta 0.2242
ile onun 0.2433 skorunun gerisine düşüyor. Public sıralama tersine dönebiliyor.

Projenin GitHub'daki tarihsel deney günlüğünde v5/v6 yaklaşık 0.566, v7
yaklaşık 0.652 ve temizlenmiş top seçeneği yaklaşık 0.696 public spellF1
raporlanmış. Bunlar da ensemble'ın 0.7724 nokta skorunun altında. V7 bu
çalışmada yeniden üretildi; v5/v6 ve temiz-top sayıları tarihsel rapordur,
bu karşılaştırma için yeniden çalıştırılmadı. Bu ayrım korunmalıdır.

## Sınırlamalar

1. **Örnek çok küçük.** Beş dakika, 17 gerçek possession. Ensemble'ın resmî
   bootstrap %95 aralığı [0.5510,0.9018]; v7'nin [0.4296,0.8108]. Bunlar
   eşli farkın aralıkları değildir ve çakışma tek başına anlamlılık testi olmaz.
   Tam maçta 0.77 alacağımızı veya belirli bir yarışma sırasını söyleyemeyiz.
2. **Possession yakalama eksik.** IoU 0.5'te 11 doğru eşleşme, 6 kaçırılan GT
   aralığı, 1 fazladan/uyuşmayan tahmin var. Precision 0.9167, recall 0.6471.
   17 yerine 12 possession üretilmesi ve süre ortalamasının GT 14.2 saniyeye
   karşı 19.8 saniye olması birleştirme/kaçırma eğilimiyle uyumlu. 0.9433
   macroF1 bu eksikleri görünmez kılmamalı.
3. **Ensemble tek başına kazanım göstermedi.** Ayrı seed skor ortalaması 0.7770,
   en yüksek tek seed 0.7862; ensemble 0.7724. Public'e bakarak en yüksek
   seed'i veya yeni ağırlıkları seçmek doğru olmaz. Eşit ağırlıklı ensemble'ın
   Metrica/SkillCorner değerlendirmesi ayrıca yapılmış değildir; önceki dış
   sonuçlar ayrı seed modellerine aittir.
4. **Etiket tanımı ve alan farkı önemli.** Üç-seed dış deneyinde BiGRU
   SkillCorner-pp'de 0.6308 iken dengeli LGB 0.5930; Metrica-kontrollüde ise
   BiGRU 0.8024, dengeli LGB 0.8173. PFF kontrollü etiketleri muhafazakâr
   olay vekilleridir; gizli yarışma GT'siyle eşdeğerlikleri kanıtlanmadı.
5. **Public tekrar kullanımı sınırına dikkat.** Şimdiye kadar görülen skorlar
   artık bilinmektedir. Yeni model/ağırlık/decoder seçimleri bu pencereye göre
   yapılmamalı; mevcut eşit ensemble isteği tek sabit formülle çalıştırıldı.
   Ağırlık araması yapılmadı. Yeni seçimler eğitim içi ayrılmış maçlarda kalmalı.
6. **Teslimin yeniden üretilebilirliği ayrıca tamamlanmalı.** CSV biçimsel
   olarak geçerli, fakat bu oturumun yeni kod/raporları henüz GitHub'a
   gönderilmedi; ağırlıklar yerel cache'te. Teslim paketinde model ağırlıkları,
   ortam sürümleri ve PFF erişim/etiket üretim adımları organizatörün aynı
   CSV'yi üretmesine yetecek şekilde sunulmalı. Ham yarışma verisi yayımlanmamalı.

Benim değerlendirmem: **baseline'ı rahat geçen, v7'den daha güçlü ve üzerinde
çalışmaya değer bir aday**. Henüz “yarışmada üst sıralardayız”, “tam maç skoru
0.77” veya “genelleme sorunu çözüldü” demek için yeterli veri yok.

## Kaynaklar ve yeniden çalıştırma

- [Resmî baseline tablosu](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/README.md)
- [Metrikler ve public/tam-maç farkı](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)
- [Kurallar ve yeniden üretilebilirlik](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
- [Projenin tarihsel deney günlüğü](https://github.com/ahmedberatAI/mtah2027-possession/blob/681a731517bcb0ee1399a899afaba52189835eed/docs/DENEMELER.md)
- [12 CSV'nin tam metrik tablosu](METRICS.md), [resmî çıktı](official.log),
  [girdi manifesti](input_manifest.json), [ensemble hata sayımları](ensemble_diagnostics.json).

```powershell
.venv/Scripts/python.exe external/more-than-a-hack-2027/evaluation/run_evaluation.py --gt public --results cache/github_public_comparison/inputs --out cache/github_public_comparison/official --bootstrap 1000 --no-detail
.venv/Scripts/python.exe src/report_public_comparison.py
```

Kare düzeyi girdiler ve ayrıntılı resmî raporlar cache altında; bu klasör yalnız
toplu metrikler içerir. Yeni eğitim, submission değişikliği veya gönderim yok.
