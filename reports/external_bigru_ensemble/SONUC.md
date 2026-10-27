# Eşit ağırlıklı BiGRU ensemble — dış veri testi, 25 Ekim 2026

SkillCorner'da kazanç belirgin ve maçlar arasında tutarlı. Metrica'da sonuç
etiket tanımına bağlı: kontrollü possession'da v7 biraz daha iyi; olay
etiketlerinde ensemble daha iyi. Her koşulda üstünlük veya BAR-ATH gizli tam
maç skoru hakkında sonuç çıkarılamaz.

| Test ve etiket | Maç | v7 spellF1 | Ensemble spellF1 | Fark | Kazanılan maç |
|---|---:|---:|---:|---:|---:|
| Metrica kontrollü possession | 3 | 0.8147 | 0.8071 | −0.0076 | 1/3 |
| Metrica olay etiketleri | 3 | 0.6155 | 0.6452 | +0.0297 | 2/3 |
| SkillCorner oyuncu possession etiketleri (pp) | 20 | 0.5829 | 0.6336 | +0.0507 | 20/20 |
| SkillCorner faz etiketleri | 20 | 0.5816 | 0.6221 | +0.0406 | 20/20 |

Ana tablo her iki model için aynı sabit öğrenilmiş ölü-top kanalını ve B2
dekoderini kullanır. SkillCorner-pp farkının eşli maç bootstrap %95 aralığı
**[+0.0436, +0.0586]**, faz etiketlerinin **[+0.0329, +0.0489]**.
Metrica'da yalnız üç maç vardır; kontrollü fark aralığı [−0.0210, +0.0068],
olay fark aralığı [−0.0057, +0.0499]. İkisi de sıfırı kapsar.

## Aynı koşullar nasıl korundu?

- BAR-ATH ensemble'ındaki aynı üç checkpoint (20261025/26/27), aynı özellikler
  ve aynı kare bazlı P(home) aritmetik ortalaması kullanıldı; ağırlıklar 1/3.
- Yeniden eğitim, ağırlık seçimi veya eşik/ceza araması yapılmadı.
- Referans, BAR-ATH'ta 0.6519 veren gerçek v7 model dosyasıdır. Model hash'i
  doğrulandı, dış veri tahminleri bu dosyayla üretildi. Sonuçları önceki dış
  deneylerin v7 referansıyla sekiz hücrenin tamamında aynı çıktı.
- Aynı test kareleri, bozulma simülasyonu V3, etiketler ve B2 ayarları kullanıldı:
  pen_team=12, pen_dead=30, min_poss=8, min_dead=50, D0=175, c=48.
- Metrica'nın üç gürültü seed'i önce kendi maçları içinde ortalandı; dokuz
  bağımsız maç gibi sayılmadı. Sonra maçlara eşit ağırlık verildi.
- 40 değerlendirme hücresi hesaplandı. Tek seed'lerin önceki 24 hücresinin
  maç bazlı skorları birebir yeniden üretildi. Ensemble işleminin BAR-ATH
  işlemiyle eşitliğini, hizalamayı ve eşli bootstrap'ı kontrol eden üç test geçti.

## Kazanç birleştirmeden mi geliyor?

Birleştirmenin ek katkısı küçük. SkillCorner-pp'de ayrı seed skorlarının
ortalaması 0.6308, ensemble 0.6336: **+0.0028**. Fazda **+0.0027**.
Bu iki ek katkının maç-bootstrap aralıkları sıfırı içeriyor; ensemble'ın tek
seed'lere kesin üstünlüğü gösterilmiş değil. v7 üzerindeki büyük kazancın çoğu
birleştirmeden önce de BiGRU modellerinde vardı.

Metrica kontrollüde ensemble ayrı seed skor ortalamasından +0.0047 iyi, ancak
v7'nin altında. Metrica olay etiketlerinde ayrı seed ortalamasından −0.0011.

Gerçek ölü-top etiketleri verilerek yapılan tanısal kontrolde de SkillCorner
kazancı sürüyor: pp **0.7410 → 0.7881**, faz **0.7457 → 0.7815**, her ikisinde
20/20 maç. Bu, farkın yalnız ölü-top modelinin hatalarıyla açıklanmadığını
destekler. Gerçek ölü-top etiketleri kullanan bu sonuçlar ana performans değildir.

## Kısıtlar ve karar

SkillCorner'daki bulgu güçlü; fakat Metrica kontrollüde genel bir iyileşme yok.
Etiket tanımları farklıdır ve aynı kaynağın iki tanımı bağımsız iki test seti
sayılmaz. Dış testlerin ölü-top kanalı öğrenilmiş modeldir; BAR-ATH'taki
iki kamera verisine dayanan kurallar burada mevcut değildir. Aynılık, her
dış testte v7 ile ensemble'ın koşullarının aynı olması anlamına gelir.

Bu maçlar önceki deneylerde de incelendi; yeni ve hiç görülmemiş bir doğrulama
seti değildir. Bu turda ayar seçilmedi. Bootstrap, sabit modeller için maç
değişkenliğini ölçer; eğitim verisi veya model seçimi belirsizliğini kapsamaz.
Metrica'nın üç maçı özellikle sınırlıdır.

**Ensemble güçlü bir aday olarak korunmalı; bütün alanlarda v7'nin yerine
geçtiği iddia edilmemeli.** BAR-ATH açık bölümündeki 0.7724 hâlâ yalnız beş
dakikanın skorudur. Gizli tam maç skoru bilinmiyor; mevcut submission CSV'si
bu testte değiştirilmedi.

Tam sayılar: [RESULTS.md](RESULTS.md), [contrasts.json](contrasts.json).
Protokol/hash kayıtları: [design.json](design.json), [verification.json](verification.json).

```powershell
.venv/Scripts/python.exe src/exp_external_ensemble.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_external_ensemble.py
```
