# Hata analizi ve BiGRU / dengeli LightGBM birleşimi

Protokol, iç doğrulama ve yeni dış değerlendirme başlamadan `design.json` ile
donduruldu. Ağırlık, dış hata analizine göre değiştirilmez.

## Ayrım ve eğitim

Önceki iç ayrım aynen korunur: 5 DFL + 52 PFF eğitim maçı, 2 DFL + 12 PFF
doğrulama maçı. Ayrılmış DFL maçları J03WPY ve J03WR9; PFF listesi
`design.json` içindedir. Hiçbir doğrulama maçı bu turdaki altı iç modelin
eğitimine girmez. Aynı doğrulama seti daha önce mimari seçimine hizmet
ettiğinden bağımsız, yeni bir son test olduğu iddia edilmez.

BiGRU: önceki seçilmiş 64 genişlik, 0.15 dropout, 0.001 öğrenme oranı;
11 epoch × 192 adım, batch 16. Her batch 8 DFL + 8 PFF pencere içerir.
Standardizasyon yalnız eğitim maçlarından hesaplanır, üç seed arasında sabittir.
Bu turda epoch seçimi veya erken durdurma yoktur.

LightGBM: aynı 172 özellik, 400 ağaç, stride 2; yalnız bilinen canlı kareler.
Eğitim alt kümesinde her kaynak toplam ağırlığın yarısını alır:
`N / (2*N_source)`. Kaynak ağırlığı bütün 71 maçtan değil, yalnız eğitim
alt kümesinden yeniden hesaplanır. İki takım görünümü de kullanılır.

Her aile için seedler 20261025, 20261026, 20261027. İç seçim için altı ayrı
model eğitilir; tüm 71 maçla eğitilmiş eski modeller iç doğrulamaya sokulmaz.

## Ağırlık seçimi

Önce her ailenin üç seed olasılığı eşit ortalanır. Ardından

`P_home = alpha * mean(P_BiGRU) + (1-alpha) * mean(P_LGB)`.

Sabit adaylar `alpha = 0, 0.25, 0.5, 0.75, 1`. Argmax etiket oyu veya logit
ortalaması yapılmaz. Her aday aynı B2 dekoderinden geçirilir.

Ölçüt önceki iç mimari seçimiyle aynıdır: gerçek ölü-top maskesiyle tam
doğrulama yarılarının spellF1'i, önce maçlar içinde, ardından kaynaklara
eşit ağırlık vererek ortalanır. DFL native, PFF kontrollü etiketler kullanılır.
PFF eğitim maskesi native ve kontrollü etiketlerin ortak bilinen-canlı
kümesidir. Bilinmeyen etiketler skor dışındadır.

En yüksek skor seçilir; tam eşitlikte daha yüksek BiGRU ağırlığı seçilir.
Uç nokta seçilirse bu da geçerli sonuçtur; karışım kazandırmış gibi sunulmaz.
`selection.json` dış hibrit değerlendirmesinden önce yazılır ve hash ile bağlanır.

DFL native ile kontrollü possession tanımı arasındaki fark seçim ölçütünün
bir kısıtıdır. Oracle ölü-top ile seçilen takım ağırlığının öğrenilmiş ölü-top
ile aynı davranması garanti değildir; dış test bunu ölçer.

## Dış değerlendirme

Seçilen tek ağırlık, önceki tüm 7 DFL + 64 PFF maçla eğitilmiş üç BiGRU ve
üç dengeli LightGBM modelinin olasılık ortalamalarına uygulanır. Yeniden
dış ağırlık araması yapılmaz. v7 gerçek BAR-ATH model dosyasıdır.

Her modele aynı dış girdiler, aynı ölü-top olasılıkları ve aynı B2 ayarları
verilir. Ana sonuç learned-D, gerçek ölü-toplu sonuç yalnız tanısaldır.
Önceki v7 ve BiGRU ensemble maç skorları birebir yeniden üretilir.

Metrica: 3 maç × 3 gürültü seed'i; seedler önce maç içinde ortalanır.
SkillCorner: 20 maç. Kontrollü/olay ve oyuncu/faz etiketleri aynı maçların
alternatif tanımlarıdır; bağımsız setler olarak toplanmaz.

Fark aralıkları 10.000 eşli maç bootstrap örneklemesiyle hesaplanır. Bunlar
sabit modeller için maç örneklemesi belirsizliğidir; eğitim/ayar seçimi
belirsizliğini kapsamaz. Üç maçlık Metrica aralıkları kırılgandır. Dış setler
önceki çalışmalarda kullanılmış olduğundan yeni, hiç görülmemiş test değildir.

## Hata analizi

Ana spellF1, resmî beş IoU eşiğinin ortalamasıdır. Tanısal hata sayımları
IoU=0.5 ile yapılır: süreye göre kaçan GT possession, eşleşmeyen tahmin,
örtüşme tabanlı birleştirme/bölme göstergeleri, geçiş eşleşmesi ve koşullu
zamanlama hatası. İki modelin ayrı ayrı kurtardığı GT possession da sayılır.
Bu analiz ayar seçimine girmez; model davranışını açıklar.

Hiçbir BAR-ATH etiketi bu deneyde okunmaz. Mevcut submission CSV'si
değiştirilmez. Ham girdiler, özellikler ve kare tahminleri yalnız gitignored
`cache/` altında kalır; raporlarda toplu istatistik vardır.

```powershell
.venv/Scripts/python.exe src/exp_hybrid.py prepare
.venv/Scripts/python.exe src/exp_hybrid.py diagnose
.venv/Scripts/python.exe src/exp_hybrid.py train_nn
.venv/Scripts/python.exe src/exp_hybrid.py train_lgb
.venv/Scripts/python.exe src/exp_hybrid.py select
.venv/Scripts/python.exe src/exp_hybrid.py evaluate
.venv/Scripts/python.exe src/exp_hybrid.py report
.venv/Scripts/python.exe src/report_hybrid_errors.py
.venv/Scripts/python.exe src/audit_hybrid.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_hybrid.py
```
