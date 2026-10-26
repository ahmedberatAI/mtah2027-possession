# PFF 25 fps deney protokolü — 24 Ekim 2026

Bu deney, önceki kapı testindeki v7 takım modeline 64 Dünya Kupası maçı ekler.
Ana aday `dfl_pff_controlled`; `pff_controlled` ve `dfl_pff_native` tanı modelleridir.
DFL etiketleri önceki deneydeki gibi bırakılmıştır. PFF kontrollü etiketleri olay
şemasına dayanan muhafazakâr vekillerdir; yarışmanın özel GT'siyle eşdeğerlikleri
doğrulanmış değildir.

## Zaman ve konum

- Kaynak `videoTimeMs` kullanılarak gerçek video zamanı üzerinde 0,04 saniyelik
  ızgara kurulur. Satır numarası veya nominal 29,97/30 fps üzerinden ölçeklenmez.
- Aynı periyot/frameNum tekrarları tekilleştirilir. 42.950 tekrarın konumları aynı;
  çelişkili geometri bulunmadı.
- Olay ve tracking ortak possession-event ID'leriyle periyot bazında hizalanır.
  En az 20 eşleşme, en fazla 120 saniye sabit ofset ve p95 artık hata ≤0,5 saniye
  kapısı kullanılır. Gerçekte 134 devrenin tamamı geçti; en kötü p95 32,86 ms.
  Bu, paylaşılan olay zamanları kontrolüdür; bağımsız video elle doğrulaması değildir.
- Yalnız ham `homePlayers`, `awayPlayers`, `balls` kullanılır. Sağlayıcı
  `ballsSmoothed` alanını olaylar ve top taşıyıcıdan oluşturduğu için bu alan
  etiket sızıntısı yaratabilir; hiçbir Smoothed alanı özelliklere girmez.
- En fazla 0,12 saniye komşu örnek aralığında doğrusal enterpolasyon yapılır.
  Kamera kesmeleri üzerinden doldurulmaz; boşluk çevresindeki 2 saniye ve
  takım başına altıdan az oyunculu geometri eğitim dışı bırakılır.
- 10510 ve 10511 maçlarında olay verisinde olup tracking'de bulunmayan 3./4.
  periyotlar dışarıda kalır. Diğer mevcut uzatmalar dahil toplam 134 devre vardır.

## Etiket kuralları

`H` ev sahibi, `A` deplasman, `D` oyun dışı, `X` belirsiz/eğitim dışıdır.
IT ilk kontrol; PA/CR/SH/CL/BC bilinçli aksiyonlar; TC'nin kontrollü ağır dokunuş
ve takeover türleri kontrol kanıtıdır. Top taşıyıcının kontrolünü içeren,
serbest top düellosu olmayan CH grupları da kullanılır.

Sekme RE, başarısız kontrol ve tek başına düello önceki sahipliği değiştirmez.
Kazanılmış hava topu da sonraki kontrol kanıtını bekler. Bir düellodan sonraki
pasın kontrolü düellonun başına geri taşınmaz. OUT/END sonrasında D, sonraki
kontrol/yeniden başlama anına kadar sürer; duran topta topun yerleştirilmesi
değil gerçek vuruş zamanı esas alınır. Geçersiz/nonEvent, videoMissing ve
belirsiz takım kayıtları X üretir. D üzerinden eski sahiplik taşınmaz; kontrol
kanıtı, grubun bitişi/çapanın en geç olanından 15 saniye sonra geçerliliğini yitirir.

`native` adı PFF'nin hazır yoğun GT'sini ifade etmez: her OTB grubunun aktör
takımını kullanan karşılaştırma vekilidir. Kontrollü etiketler bu vekile göre
ortak canlı karelerin %2,18'ini değiştirir; bilinen canlı H↔A geçişleri
15.249'dan 12.473'e iner. Bu geçiş sayısı bütün possession bölümlerinin sayısı değildir.

## Eğitim ve dış değerlendirme

PFF etiket karşılaştırması aynı `(native < 2) & (controlled < 2)` karelerinde
yapılır. Ortak canlı veri 5.339.329 kare, yaklaşık 59,33 saattir. Stride 2 ve
iki takım perspektifiyle DFL+PFF toplam 5.892.500 eğitim satırı verir;
örnek sayısı eski DFL eğitimine göre 10,65 kattır. Kaynaklar kare başına eşit
ağırlık alır; böylece PFF verisi baskındır. Kaynak dengelemesi bu deneyde denenmedi.

Aynı 172 özellik, 400 LightGBM ağacı, sabit seed ve sekiz iş parçacığı kullanılır.
DFL sim.V3 ile bozulur; PFF zaten yayın tracking'i olduğu için tekrar simüle
edilmez. B2 pen_team=12, D0=175, c=48 sabittir. Öğrenilmiş D modeli de önceki
yedi DFL modeli olarak sabit tutulur. Bu, v7 takım modelinin ortak dış test
düzeneğindeki karşılaştırmasıdır; BAR-ATH'a özgü v7 kural tabanlı D kanalı bu
dış maçlarda çalıştırılmış değildir.

Üç Metrica maçındaki üçer bozulma tohumu önce maç içinde ortalanır. SkillCorner
20 maçtır. Metrica ctrl/events ve SkillCorner pp/phase alternatif etiketleri
ayrı raporlanır. Resmi spellF1 maç ortalaması ve eşli farklar kullanılır.
%95 aralıklar 10.000 maç düzeyi bootstrap örneğidir; eğitim seed'i/örnek seçimi
belirsizliğini kapsamaz. Metrica'nın yalnız üç maç olması çıkarımı sınırlar.

Oracle-D sadece tanı içindir: gerçek D'den yüksek güvenli olasılık verilir;
dekoderin son D sınırları sert sabitlenmez. Her veri kaynağındaki iki etiket
konvansiyonunun D maskelerinin aynı olduğu ayrıca doğrulandı. Böylece oracle
dekodunun iki konvansiyonda ortak kullanılması geçerlidir.

Dış test skorlarına göre özellik, hiperparametre, eşik veya kaynak ağırlığı
seçilmedi. BAR-ATH üzerinde yeni değerlendirme veya submission üretilmedi.

## Yerel çıktılar

- `cache/pff25/raw_control_v2/*.npz`: 25 Hz tracking, etiketler ve geometri maskesi.
- `cache/pff25/raw_control_v2/labels/*.csv.gz`: 134 devrenin açık etiket tabloları.
  `half_frame25` devre ızgarasındaki indeks, `video_time_s` mutlak kaynak video
  zamanı, `team_training_eligible` gerçek ortak canlı eğitim maskesidir.
  NPZ içindeki `training_mask` ise yalnız geometri kabul maskesidir.
- `cache/pff_experiment/raw_control_v2/*_controlled.txt` ve `dfl_pff_native.txt`:
  eğitilmiş LightGBM modelleri; özellik adları eğitim JSON'larında sıralıdır.
- `design.json`, `*_training.json`, `data_summary.json`, `data_audit.json`:
  protokol, sayımlar, hizalama ve kaynak kod hash'leri.
- `comparison.json`, `summary.json`, `RESULTS.md`: ham toplu skorlar ve eşli sonuçlar.

Ham veri ve kare düzeyi çıktılar gitignored klasörlerde, `D:\barca` altında
tutulur. Etiketlerin üretimi [gradient.py](../../src/gradient.py), eğitim ve
değerlendirme [exp_pff.py](../../src/exp_pff.py), raporlama
[report_pff.py](../../src/report_pff.py) içindedir. Komutlar RESULTS.md'de bulunur.
