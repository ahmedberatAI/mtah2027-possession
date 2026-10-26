# DFL kontrollü etiket deneyi

Amaç: yalnız PFF değil, DFL eğitiminde de kontrollü possession etiketleri
kullanıldığında BiGRU'nun Metrica'daki bölünmelerinin azalıp azalmadığını ölçmek.

## Etiket müdahalesi

Yeni eşik araması yapılmaz. Repodaki `bench.trk_ctrl` kuralı, temiz DFL takip
verisi üzerinde yeniden çalıştırılır ve önceki `trk` önbelleğiyle birebir
eşitlik doğrulanır. Bir possession doğrudan rakibin possession'ından sonra
geliyorsa, yeni takım yeterli kontrol kanıtı göstermediğinde önceki takıma
geri atanır. Ölü-top aralığı üzerinden sahiplik taşınmaz.

Kontrol vekili: topa 1.5 m içinde, top yüksekliği en fazla 2 m iken yeni
takımın en az iki oyuncusunun yakın sahip görünmesi veya yeni takım sahipliğinin
possession içinde toplam en az 25 kare sürmesi. Bu 25 kare zorunlu olarak
ardışık değildir; iki oyuncu görülmesi tamamlanmış pasın kesin kanıtı değildir.
Bu yüzden etiketler insan tarafından doğrulanmış resmî GT değil, takip verisinden
türetilen kontrollü possession yaklaşımıdır.

PFF kontrollü etiketleri, ortak-canlı eğitim maskesi, ham özellikler, gürültü
simülasyonu ve tüm dosya yolları korunur. DFL için yalnız yeni etiket dosyaları
ayrı cache klasörüne yazılır. Canlı, ölü ve bilinmeyen maskelerin değişmediği
doğrulanır. Bu, kaynak örneklemesini ve standardizasyonu da sabit tutar.

## Ayar seçimi

Önceki maç ayrımı korunur: 5 DFL + 52 PFF eğitim, 2 DFL + 12 PFF doğrulama.
DFL doğrulama etiketleri de `trk`, PFF etiketleri aynı kontrollü etiketlerdir.
Bu ayrım daha önce kullanılmıştır; yeni bir bağımsız holdout değildir.

İki önceden belirlenmiş BiGRU yapılandırması:

- Mevcut ayar: genişlik 64, dropout 0.15, öğrenme oranı 0.001.
- Önceki aramadaki alternatif: genişlik 96, dropout 0.25, öğrenme oranı 0.0005.

Her ayar üç seed ile eğitilir: 20261025, 20261026, 20261027. Epoch sayısı önceki
protokolden 11 olarak sabittir; yeni epoch araması veya erken durdurma yoktur.
192 adım/epoch, batch 16, pencere örneklemesi ve kayıp kaynak başına 50:50.
Scaler yalnız eğitim maçlarından hesaplanır, seedler arasında sabittir.

Her ayarın üç seed olasılığı eşit ortalanır. Seçim ölçütü: sabit B2 ve gerçek
ölü-top maskesi ile doğrulama maçlarının spellF1'i, önce maçlara sonra iki
kaynağa eşit ağırlık verilerek ortalanır. Tam eşitlikte listedeki ilk ayar.
Kontrollü DFL etiketleri altında eski native modellerin iç skoru da aynı
doğrulama setinde referans olarak ölçülür; aday seçimine katılmaz.

Seçim `selection.json` ile dış değerlendirmeden önce dondurulur. Seçilen
ayar tüm 7 DFL + 64 PFF üzerinde üç seed ile yeniden eğitilir. Alternatif
ayar seçilirse, mevcut 64 ayarıyla kontrollü etiketli üç model de ayrıca
eğitilir. Böylece sadece etiket değişimi ile ayar değişiminin etkisi ayrılır.

## Dış test ve hata analizi

Metrica ve SkillCorner'a yalnız dondurulmuş modeller uygulanır. Native mevcut
ensemble, kontrollü sabit ayar ve kontrollü seçilen ayar karşılaştırılır.
Ölü-top kanalı ve B2 aynıdır. Ana sonuç öğrenilmiş D; gerçek D yalnız tanısal.
BAR-ATH verisi veya açık GT bu turda kullanılmaz.

Ana metrik beş IoU eşikli resmî spellF1. Hata analizi IoU=.5: kaçan GT,
eşleşmeyen tahmin, <4 sn possession kaçırma ve örtüşme tabanlı bölme/birleştirme
göstergeleri. Bölme göstergesi bir GT possession'ın aynı takımın en az iki
tahminiyle pozitif zaman örtüşmesidir; kesin ve birbirini dışlayan hata nedeni
değildir. Birleştirme bunun ters yöndeki karşılığıdır.

Metrica'nın üç gürültü gerçekleşimi maç içinde ortalanır. Üç maç dokuz maç
gibi sayılmaz. SkillCorner 20 maçtır. Aynı maçların iki etiket tanımı bağımsız
setler değildir. Skor farkları ve hata oranı farkları için 10.000 eşli maç
bootstrap örneklemesi kullanılır. Üç maçlık Metrica aralıkları kırılgandır;
aralıklar eğitim/model seçimi belirsizliğini kapsamaz.

Dış setler önceki deneylerde kullanıldı. Bu turda dış eşik, ağırlık veya
model ayarı seçilmez; sonuç görüldükten sonra dondurulmuş seçim değiştirilmez.
Mevcut BAR-ATH submission dosyası korunur.

```powershell
.venv/Scripts/python.exe src/exp_controlled_dfl.py prepare
.venv/Scripts/python.exe src/exp_controlled_dfl.py train_internal
.venv/Scripts/python.exe src/exp_controlled_dfl.py select
.venv/Scripts/python.exe src/exp_controlled_dfl.py refit
.venv/Scripts/python.exe src/exp_controlled_dfl.py evaluate
.venv/Scripts/python.exe src/exp_controlled_dfl.py report
.venv/Scripts/python.exe src/audit_controlled_dfl.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_controlled_dfl.py
```
