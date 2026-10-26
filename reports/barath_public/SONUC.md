# BAR-ATH açık bölüm sonucu — 25 Ekim 2026

**BiGRU, açık etiketli bölümde v7'yi üç seed'in tamamında geçti.** Ortalama
spellF1 **0.7770**, v7 **0.6519**: fark **+0.1252**. Dengeli LightGBM'nin
skoru **0.6667**, farkı **+0.0148**.

| Model | Açık bölüm spellF1 | v7'ye fark | Seed aralığı |
|---|---:|---:|---|
| v7 | 0.6519 | — | Tek referans |
| Dengeli DFL+PFF LightGBM | 0.6667 ± 0.0000 | +0.0148 | 0.6667–0.6667 |
| DFL+PFF BiGRU | **0.7770 ± 0.0080** | **+0.1252** | 0.7724–0.7862 |

± değerleri üç eğitim seed'i arasındaki standart sapmadır; üç ayrı maçın
belirsizliği veya ensemble skoru değildir. Seedler 20261025, 20261026, 20261027.
BiGRU skorları sırasıyla **0.7724, 0.7724, 0.7862**. LightGBM'nin üç seed'i
aynı spellF1'i verdi; bu, tahmin dosyalarının aynı olduğu anlamına gelmez.

## Neyin karşılaştırıldığı

Yalnız takım modeli değiştirildi. Bütün modellerde v7'nin **kural tabanlı
ölü-top kanalı**, yayın/taktik hizalaması ve **B2 dekoderi** aynı. Önceki dış
Metrica/SkillCorner deneyindeki öğrenilmiş D kanalı burada kullanılmadı.

V7 referansı `model_v7.py` içindeki tam tarifle, yedi DFL maçının doğal
sıralaması ve sekiz CPU iş parçacığıyla yeniden eğitildi. Elde edilen 0.6519,
reponun tarihsel yaklaşık 0.652 v7 sonucunu yeniden üretiyor. BiGRU ve dengeli
LightGBM, önceki üç-seed deneyinin kayıtlı ağırlıklarıdır; yeniden ayarlanmadı.

İki devrenin tamamında tahmin yapıldı. **Yedi adet 199.319 satırlık CSV**,
etiketler skorlamaya alınmadan önce üretildi, doğrulandı ve hash ile donduruldu.
Ardından reponun sarmalayıcısı ve ayrıca resmî `run_evaluation.py` çalıştırıldı;
skorları uyuşuyor. Açık bölümde hiperparametre/seed seçimi, elle etiketleme veya
yarışmaya gönderim yapılmadı.

## Sonucun sınırı

Bu, **yaklaşık beş dakikalık ve 17 GT possession içeren açık bölüm** skorudur;
tam maçın gizli skoru değildir. V7 ve dengeli LightGBM açık bölümde 10,
BiGRU 12 possession üretti. Üç seed aynı kısa bölümü gördüğünden bağımsız
üç maç gibi yorumlanamaz.

Resmî aracın 1.000 bootstrap tekrarıyla bireysel %95 spellF1 aralıkları geniş:
v7 **[0.4296,0.8108]**, dengeli LightGBM **[0.4444,0.8276]**, BiGRU
**[0.5510,0.9018]** (ilk iki seed) ve **[0.5517,0.9169]** (üçüncü seed).
Bunlar eşli farkın aralıkları değildir; çakışmalarından tek başına bir fark
anlamlılık testi çıkarılamaz. Nokta skorundaki yükseliş güçlü bir aday sinyali,
tam maç üstünlüğü garantisi değildir.

Önceki dış test bulgusuyla birlikte okunduğunda BiGRU hem SkillCorner'da hem
bu açık bölümde umut verici; Metrica-kontrollüdeki düşüş hâlâ geçerli. En yüksek
public seed'i seçerek varsayılan model değiştirilmedi. Tüm üç seed çıktısı
ayrı saklandı.

## Dosyalar ve doğrulama

- Ham veri: `data/tactical/`, `data/broadcast/`; videolar indirilmedi.
- İki indirme arşivi Downloads'tan `cache/barath_download/` altına taşındı.
  Arşiv CRC'leri ve beklenen yedi dosya adı doğrulandı.
- Tahmin CSV'leri: `submissions/barath_public_v1/`.
- Olasılıklar, v7 model dosyası ve resmî ayrıntılar: `cache/barath_public_v1/`.
- Toplu sonuçlar: [RESULTS.md](RESULTS.md), [results.json](results.json),
  [official.log](official.log). Kare düzeyi resmî çıktılar yalnız cache içindedir.
- **32 test geçti.** Kare başlangıcı/uzunluğu, geçerli olasılıklar, CSV
  biçimi, sabit model hash'leri ve iki skor yolunun uyumu denetlendi.

[Deney protokolü](design.json) · [Dondurulmuş tahminler](prediction_manifest.json) ·
[Doğrulama kaydı](verification.json) · [Çalıştırma komutları](RESULTS.md)
