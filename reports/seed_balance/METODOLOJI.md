# Üç seed ve kaynak dengesi kontrolü

Bu deney önceki iç doğrulamada seçilmiş ayarları dondurur. Yeni ayar araması,
en iyi seed seçimi veya dış skora göre model/dekoder değişikliği yapılmaz.
Üç yeni seed **20261025, 20261026, 20261027**; önceki 20261024 seed'i yeni
ortalamalara dahil değildir. Deney tanımı ve kod hash'leri [design.json](design.json).

## Eğitilen modeller

Her seed için TCN, BiGRU, normal ağırlıklı LightGBM ve kaynakları dengeli
LightGBM: toplam **12 yeni model**. Hepsi 7 DFL + 64 PFF ile eğitilir. Önceki
v7 ve tek-seed PFF LightGBM skorları ayrıca sabit referans olarak saklanır.

TCN/BiGRU ayarları: 64 kanal, dropout .15, AdamW lr .001, 11 epoch, epoch
başına 192 minibatch, minibatch 16. Her minibatch 8 DFL + 8 PFF penceresi;
kayıp kaynak başına eşit ağırlık alır. Seed; başlangıç ağırlıklarını, dropout'u
ve pencere örneklemesini değiştirir. Scaler bütün seed/mimarilerde aynı eğitim
verisinden aynı seed 20261024 ile hesaplanır. Aynı yeni seed'de TCN ve BiGRU
aynı pencere çekilişlerini kullanır. Yeni erken durdurma veya kontrol noktası
seçimi yapılmaz: 11. epoch çıktısı kullanılır.

LightGBM'nin her iki kontrolü de **aynı 5.892.500 satır**, 172 özellik, etiket,
stride 2, 400 ağaç ve aynı eski hiperparametreleri kullanır. Seed alanları
`seed`, `data_random_seed`, `feature_fraction_seed`, `bagging_seed` olarak
açıkça sabitlenir. Aynı seed'in iki LGB modeli arasında yalnız örnek ağırlıkları
değişir; böylece doğrudan kaynak dengesi karşılaştırması yapılabilir.

Normal LGB'de her satırın ağırlığı 1. Dengeli LGB'de kaynağın satır sayısı
`N_s`, bütün satırlar `N` olmak üzere ağırlık **N/(2*N_s)**. DFL ve PFF'nin
toplam ağırlıkları eşittir; ortalama ağırlık 1 kalır. Bu son koşul, toplam
kayıp ölçeği ve L2 regularizasyon ilişkisini gereksiz yere değiştirmemek içindir.
Kaynakların matris içindeki sırası önceki satır üreticisinin blok/etiket
sayılarıyla yeniden doğrulanır. Sayılar [weights.json](weights.json) içindedir.

Bu LGB kontrolü kaynaklar içinde kareleri eşit ağırlıklandırır. Sinir ağları
ise kaynak içinde maç/devre/canlı merkez seçer ve pencere canlı kaybını
normalize eder. Dolayısıyla dengeli LGB → sinir ağı farkı yalnız zaman
mimarisinin izole etkisi değildir: pencere örnekleme, normalizasyon, 12,5 Hz
ağ çıkarımı / 25 Hz'e taşıma ve öğrenici ailesi de değişir.

## Dış değerlendirme

Metrica (3 maç × 3 bozulma tohumu) ve SkillCorner (20 maç) önceki deneyle aynıdır.
Metrica tohumları önce maç içinde ortalanır; bağımsız maç sayısı üçtür. Aynı
öğrenilmiş ölü-top modeli ve sabit B2 kullanılır; oracle-D ayrıca tanıdır.
İki konvansiyonun D maskelerinin eşitliği denetlenerek dekodlar güvenle ortak
kullanılır. Yeni 12 model × 2 kaynak × 2 konvansiyon × 2 D = **96 hücre**;
önceki v7/PFF LightGBM'den 16 sabit referans hücresi eklenir.

Raporlanan ortalama, her seed'in maç ortalaması spellF1 skorlarının ortalamasıdır.
Standart sapma üç seed arasındaki örnek standart sapmadır (ddof=1). Tahminler
birleştirilmez; bu bir ensemble skoru değildir. Her seed sonucu ve min–maks
ayrıca verilir; en iyi seed üzerinden karar verilmez.

Eşli farklar için seed × maç matrisi oluşturulur. 10.000 bootstrap tekrarında
seed ekseni ve maç ekseni tekrar örneklenir; aynı çekilen maçlar bütün seedler
ve karşılaştırılan modeller için paylaşılır. Aynı maçı üç seed altında üç
bağımsız maç saymak yasaktır. **Yalnız üç seed** bulunduğundan bu %95 aralıklar
keşifsel belirsizlik özetleridir, kesin genellenebilirlik garantisi değildir.
Çoklu karşılaştırma düzeltmesi uygulanmaz; ana önceden tanımlı karşılaştırma
kaynak ağırlığı çifti ve her sıra modelinin dengeli LightGBM'ye farkıdır.

## Nasıl yorumlanmalı?

1. **Dengeli LGB − normal LGB:** kaynak ağırlıklandırmasının doğrudan eşli etkisi.
2. **TCN/BiGRU − dengeli LGB:** kaynaklar 50:50 iken sıra modeli düzeninin ek etkisi.
3. **TCN/BiGRU − normal LGB:** toplam fark; ilk iki farkın toplamına eşittir.

İkinci fark pozitif kalırsa kaynak dengesi tek başına bütün kazancı açıklamaz.
Fakat saf zaman mimarisi katkısını ölçmek için aynı sinir ağı eğitim protokolünde
zamansız bir ağ veya bağlam ablasyonu ayrıca gerekir. Bu deney böyle bir
ablasyon yapmış gibi yorumlanmaz. Önceki dış testler zaten görülmüş olduğundan
bu çalışma yeni, hiç görülmemiş bir test seti üzerinde genelleme kanıtı değildir.
PFF kontrollü etiketleri vekildir; BAR-ATH üzerinde skor veya submission üretilmez.

## Çalıştırma ve kayıtlar

```powershell
.venv/Scripts/python.exe src/exp_seed_balance.py prepare
.venv/Scripts/python.exe src/exp_seed_balance.py train_nn
.venv/Scripts/python.exe src/exp_seed_balance.py train_lgb
.venv/Scripts/python.exe src/exp_seed_balance.py evaluate --family nn
.venv/Scripts/python.exe src/exp_seed_balance.py evaluate --family lgb
.venv/Scripts/python.exe src/exp_seed_balance.py report
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Önceki PFF ve sequence cache'leri gerekir. GPU sinir ağı eğitimi ve CPU LGB
eğitimi ayrı süreçlerde birlikte çalışabilir. Değerlendirme dosyaları aile
bazında ayrıdır. Kontrol noktaları ve olasılıklar `cache/seed_balance_v1/`,
toplu sonuçlar `reports/seed_balance/` altındadır. Bütün büyük dosyalar
`D:\barca` içinde kalır. Frozen kaynak kodu veya seçim dosyası değişirse sürücü
devam etmeyi reddeder; yeni deney sürümü gerekir.
