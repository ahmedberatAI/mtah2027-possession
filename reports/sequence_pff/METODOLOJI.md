# Dengeli DFL/PFF TCN ve BiGRU protokolü

Deney 24 Ekim 2026'da, dış sonuçlara bakmadan tanımlandı. Sabit protokol ve maç
kimlikleri [design.json](design.json) içindedir. Ham PFF dönüştürücüsü önceki
`raw_control_v2` sürümüdür: 64 maç, 134 devre, kontrollü possession vekilleri,
25 Hz hizalama, yalnız ham tracking. Smoothed top yine kullanılmaz.

## Ayrım ve seçim

Seed 20261024 ile maç bazında **5 DFL + 52 PFF eğitim**, **2 DFL + 12 PFF iç
doğrulama** ayrılır. Aynı maçın hiçbir devresi iki tarafa da girmez. Metrica ve
SkillCorner ayar seçiminde yüklenmez. Bu iki dış kaynak önceki deneylerde
görülmüştür; burada yeni ayarlar bunların skorlarına göre değiştirilmez.

Her mimaride iki aday vardır: (64 kanal, dropout .15, lr .001) ve (96 kanal,
dropout .25, lr .0005). AdamW, weight decay .0001, gradient clip 1.0, 12 epoch,
epoch başına 192 minibatch, minibatch 16. Epoch burada rastgele pencerelerle
sabit güncelleme bütçesidir; verinin eksiksiz tek geçişi anlamına gelmez.

Her adayın kontrol noktası, sabit iç doğrulama pencerelerindeki en düşük BCE
ile seçilir. Her devreden sabit seed ile sekiz canlı merkez seçilir; her iki
takım görünümü değerlendirilir. Pencere kayıpları maç içinde, sonra kaynak
içinde ortalanır; DFL ve PFF son skorda eşit ağırlıktadır.

Ardından her mimarinin iki aday kontrol noktası iç doğrulama maçlarının tamamı
üzerinde çalıştırılır. Seçim ölçütü kaynak başına maç ortalaması **oracle-D +
sabit B2 spellF1**'ın 50:50 ortalamasıdır. Oracle-D burada yalnız takım modelini
seçmek için kullanılır; D etiketleri ağa girdi olmaz. Dekoder ayarlanmaz.

Seçilen kanal/lr/dropout ve epoch sayısı dondurulur. Son modeller **7 DFL + 64
PFF** üzerinde baştan eğitilir; scaler da bu eğitim verisinden yeniden
hesaplanır. Sonra dış değerlendirme yapılır. Seçim dosyası hash'i ve son model
hash'leri [final_manifest.json](final_manifest.json) ile bağlanır.

## Kaynak ve özellik dengesi

Her minibatch tam olarak 8 DFL ve 8 PFF penceresidir. Kaynak içinde maç,
devre ve geçerli canlı merkez rastgele seçilir. Kayıp önce her pencerenin
geçerli canlı karelerinde ortalanır, sonra kaynak başına ortalanır ve 50:50
birleştirilir. Böylece PFF'nin daha uzun süresi veya daha çok canlı karesi
tek başına eğitim ağırlığını artırmaz. Bu seçim DFL maçlarının daha sık
görülmesine neden olur; iç doğrulama bu aşırı uyum riskini izler.

Her takım görünümü önceki v7'deki aynı **172 özellikten** oluşur. Sinir ağı
girdisi standartlaştırılmış 172 sayı + 172 eksiklik göstergesi + 1 gerçek
zaman/padding göstergesidir (345 kanal). Etiketler girdi olarak kullanılmaz.
Eksiklik göstergesi geometri/özellik eksikliğidir; possession/D/X etiketi değildir.

Scaler sadece ilgili eğitim maçlarından, maç başına dengeli örneklerle ve iki
takım görünümüyle hesaplanır; kaynakların momentlerine eşit ağırlık verilir.
Değerler ±6 standard sapmada kırpılır; NaN için normalize değer sıfır ve
eksiklik göstergesi bir kullanılır. İç doğrulama scaler'a girmez.

DFL sim.V3, PFF ise mevcut ham yayın tracking özellikleriyle kullanılır. PFF
canlı hedef maskesi önceki LightGBM deneyindeki native/controlled ortak canlı
karelerdir. D ve X kayba girmez; geometri zaman bağlamında kalabilir. Pencereler
devre sınırını aşmaz; eksik baş/son bağlam sıfır padding ile işaretlenir.

## Zaman modelleri

Kaynak ve sonuç 25 Hz'dir. Ağ, stride 2 ile **12,5 Hz** çalışır; çıktının
logitleri 25 Hz'e doğrusal taşınır. Bu tercih önceki BiGRU'nun 5 Hz'inden daha
sık zaman örneklemesidir; LightGBM'nin tam-kare çıkarımıyla birebir aynı değildir.

Her ağ penceresi 512 adımdır (40,96 saniye). Yalnız ortadaki 256 adımın kaybı
ve çıktısı kullanılır (20,48 saniye); iki yanda 128 adım bağlam vardır.
TCN altı residual blok içerir, her blokta iki kernel-3 konvolüsyon bulunur;
dilation 1,2,4,8,16,32. Toplam receptive field 253 adım (~20,24 saniye),
merkezin her iki yanında 126 adımdır; 128 adımlık halo bunu kapsar.
BiGRU iki katmanlı ve çift yönlüdür; her pencereye bağımsız gizli durumla başlar.

Her iki model de gelecekteki geometriyi kullanabilen çevrimdışı modellerdir.
İki takım görünümünün logit farkının yarısı ev sahibi logiti olur; takım
sırasının değiştirilmesi P(H)'yi 1−P(H)'ye çevirir. Çıkarımda da aynı pencere
ve halo kullanılır; çok uzun devreyi GRU'ya tek seferde vererek eğitim/çıkarım
bağlamını değiştirmeyiz.

## Dış karşılaştırma ve sınırlar

Karşılaştırmalar mevcut v7 (DFL7) ve önceki DFL+PFF-controlled LightGBM
sonuçlarıyla yapılır. Öğrenilmiş D kanalı, B2, maçlar, bozulma tohumları ve
alternatif etiket konvansiyonları sabittir. Ayrıca oracle-D tanısı raporlanır.
Metrica'nın üç bozulma tohumu maç içinde ortalanır: örnek sayısı üç maçtır.
SkillCorner 20 maçtır. Eşli %95 aralıklar 10.000 maç bootstrap örneğidir.

Tek eğitim seed'i ve iki aday/mimari ile sınırlı bir aramadır. Aralıklar eğitim
seed'i belirsizliğini kapsamaz. Önceki LightGBM kaynakları kare sayısına göre
birleştiriyordu; burada kaynak dengesi, ağ özellik işlemesi ve zaman modeli
birlikte değişir. Sonuç farkı yalnız mimariye atfedilemez. PFF etiketleri
muhafazakâr vekildir; yarışmanın özel GT üretim kuralı hâlâ doğrulanmış değildir.
BAR-ATH üzerinde skor veya submission üretilmez.

## Çalıştırma

PyTorch 2.9.0 CUDA 12.8 paketi [resmi kurulum kaynağından](https://pytorch.org/get-started/previous-versions/)
kuruldu; GPU NVIDIA RTX 5070 Ti Laptop, 12 GB. Paket önbelleği ve büyük dosyalar
`D:\barca` altında. CUDA/cuDNN deterministik seçenekleri ve seed sabittir;
farklı donanım/yazılım sürümlerinde bit düzeyi eşitlik iddia edilmez.

```powershell
$env:UV_CACHE_DIR='D:\barca\.uv-cache'
uv pip install --python .venv/Scripts/python.exe torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe src/exp_seq_pff.py prepare
.venv/Scripts/python.exe src/exp_seq_pff.py search
.venv/Scripts/python.exe src/exp_seq_pff.py refit
.venv/Scripts/python.exe src/exp_seq_pff.py evaluate
.venv/Scripts/python.exe src/exp_seq_pff.py report
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Önceden hazırlanmış DFL/PFF/Metrica/SkillCorner özellikleri ve PFF LightGBM
raporu gerekir. Bunların komutları `reports/pff/RESULTS.md` içindedir.
Modeller ve kare çıktıları `cache/sequence_pff_v1/`, toplu raporlar bu klasördedir.
Kaynak kodu değiştiğinde eski cache ile karıştırmadan yeni deney sürümü kullanın.
