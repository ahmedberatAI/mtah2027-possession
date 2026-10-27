# Dengeli PFF/DFL TCN ve BiGRU sonuçları — 24 Ekim 2026

> **Üç-seed güncellemesi:** [Yeni kontrol deneyi](../seed_balance/SONUC.md)
> tamamlandı. SkillCorner kazancı dengeli LightGBM üzerinde de sürüyor;
> Metrica-kontrollüde aşağıdaki tek seed daha iyimser kalmış. Kararlılık için
> yeni üç-seed sonuçlarını esas alın.

**İki sıra modeli de SkillCorner'da hem v7'yi hem PFF'li LightGBM'yi geçti.
Metrica kontrollü possession'da belirgin üstünlük yok.** TCN bu konvansiyonda
biraz daha dengeli; BiGRU SkillCorner oyuncu-possession ve Metrica olay
konvansiyonunda daha yüksek skor verdi. İki model ayrı aday olarak saklandı;
mevcut v7/submission değiştirilmedi.

## Ana karşılaştırma

Sabit öğrenilmiş ölü-top kanalı ve sabit B2 ile maç ortalaması **spellF1**:

| Dış test / etiket | v7 — DFL7 | DFL+PFF LightGBM | TCN | BiGRU |
|---|---:|---:|---:|---:|
| Metrica kontrollü | 0.8147 | 0.8144 | **0.8168** | 0.8120 |
| Metrica olay | 0.6155 | 0.6100 | 0.6253 | **0.6435** |
| SkillCorner oyuncu possession | 0.5829 | 0.5913 | 0.6265 | **0.6284** |
| SkillCorner faz | 0.5816 | 0.5897 | **0.6208** | 0.6201 |

SkillCorner oyuncu-possession'da:

- **TCN:** v7'ye +0.0436; PFF LightGBM'ye **+0.0352**, eşli %95 aralık
  **[+0.0278,+0.0430]**. Her iki referansa karşı **20/20 maç** iyileşti.
- **BiGRU:** v7'ye +0.0455; PFF LightGBM'ye **+0.0370**, eşli %95 aralık
  **[+0.0300,+0.0444]**. Her iki referansa karşı **20/20 maç** iyileşti.
- Faz konvansiyonunda PFF LightGBM'ye TCN +0.0312 (20/20), BiGRU +0.0304
  (19/20). Oracle-D ile de kazanımlar korunuyor: oyuncu-possession'da
  PFF LightGBM'ye TCN +0.0333, BiGRU +0.0395.

Metrica kontrollü etiketlerde v7'ye göre TCN +0.0021 [%95: −0.0012,+0.0041],
BiGRU −0.0027 [−0.0081,+0.0063]. Bunlar belirgin üstünlük veya kayıp kanıtı
değil. Metrica yalnız üç bağımsız maç; üç bozulma tohumu maç içinde ortalandı.
BiGRU olay etiketlerinde PFF LightGBM'yi üç maçta da geçti (+0.0336), ancak
bu alternatif tanımı yarışmanın özel GT'siyle eşitlemiyoruz.

Kare takım doğruluğu SkillCorner-pp'de v7 %90.41, PFF LightGBM %91.56,
TCN %92.46, BiGRU %92.57. Dolayısıyla possession bölümü kazanımına kare
doğruluğu artışı da eşlik ediyor.

## Ayarlar nasıl seçildi?

**5 DFL + 52 PFF eğitim; 2 DFL + 12 PFF iç doğrulama.** Ayrım maç bazında
ve sabit seed ile yapıldı. Normalizasyon yalnız eğitim kısmından hesaplandı.
Metrica/SkillCorner bu seçime girmedi. Her minibatch 8 DFL + 8 PFF penceresi;
kayıp ve iç seçim skoru kaynaklara 50:50 ağırlık veriyor.

Her mimaride iki boyut/lr/dropout seçeneği 12 epoch eğitildi. Kontrol noktası
sabit iç pencerelerin BCE'siyle seçildi; iki aday arasında karar iç maçların
tamamında kaynakları eşit ağırlıklandıran oracle-D/B2 spellF1 ile verildi:

| Aday | Seçilen epoch | İç DFL spellF1 | İç PFF spellF1 | Dengeli skor |
|---|---:|---:|---:|---:|
| **TCN-64** | 11 | 0.6586 | 0.7339 | **0.6962** |
| TCN-96 | 11 | 0.6539 | 0.7160 | 0.6849 |
| **BiGRU-64** | 11 | 0.6643 | 0.7418 | **0.7031** |
| BiGRU-96 | 11 | 0.6501 | 0.7303 | 0.6902 |

İki mimaride de **64 kanal, lr=.001, dropout=.15, 11 epoch** donduruldu.
Ardından tüm **7 DFL + 64 PFF** ile sıfırdan eğitim yapıldı. Her son model
16.896 DFL + 16.896 PFF penceresi gördü; tüm 71 maç örneklendi. Canlı kayıp
pozisyonları DFL'de 3.668.524, PFF'de 3.700.739; bunlar tekrarları içeren
pozisyon sayılarıdır, benzersiz kare sayısı değildir.

TCN 172.097, BiGRU 146.817 parametredir. Her takım için aynı 172 v7 özelliği,
eksiklik ve padding göstergeleriyle kullanılır. Ağ 12,5 Hz'de çalışır;
logitler 25 Hz'e taşınır. 40,96 saniyelik pencerenin ortadaki 20,48 saniyesi
çıktı/kayıp için kullanılır. TCN altı dilated residual blok; BiGRU iki çift
yönlü katmandır. Her ikisi çevrimdışı gelecek bağlamını kullanır.

## Yorum ve sınırlar

Bu dengeli düzende önceki BiGRU deneyindeki büyük kayıp tekrarlanmadı.
Ancak değişen tek unsur veri miktarı değil: kaynak dengesi, zaman örneklemesi,
özellik işleme ve eğitim protokolü de farklı. Kazancı tek başına 64 maça veya
tek başına mimariye atfetmek için ek ablasyon gerekir. Mevcut LightGBM
kaynakları kare sayısına göre birleştiriyordu; dengeli LightGBM kontrolü bu
deneyde eğitilmedi.

İki aday/mimari, tek eğitim seed'i ile sınırlı bir aramadır. %95 aralıklar
maç bootstrap'ıdır; eğitim seed'i belirsizliğini içermez. PFF kontrollü
etiketleri muhafazakâr vekillerdir. Dış maçlar geçmiş deneylerde de görüldüğü
için bunlar hiç görülmemiş yeni bir test seti değildir; bu çalışmada sonuçlara
bakarak ayar değiştirilmedi. BAR-ATH özel skoru veya submission üretilmedi.

**Pratik karar:** TCN ve BiGRU ileri doğrulama için güçlü adaylar. SkillCorner
kazancı iki etiket konvansiyonunda ve oracle-D tanısında destekleniyor.
Metrica-kontrollü için henüz model yükseltme kanıtı yok. Sonraki çalışma için
birden fazla eğitim seed'i ve dengeli LightGBM kontrolü, mimari katkısını
ayırt etmeye yardımcı olur; mevcut dış testlerde ek ayar araması yapılmamalı.

## Dosyalar ve doğrulama

- TCN: `cache/sequence_pff_v1/tcn_final.pt`.
- BiGRU: `cache/sequence_pff_v1/bigru_final.pt`.
- Her checkpoint ağırlıklar, scaler, mimari ayarları, epoch, stride ve pencere
  tanımını içerir. Yükleme/çıkarım `src/exp_seq_pff.py::load_model` ve
  `src/seq_pff.py::predict_raw` üzerinden yapılır.
- **23 test geçti.** Kaynak kaybı dengesi, normalizasyon, takım simetrisi,
  devre sınırı, son pencere, TCN bağlam sürekliliği ve geri yayılım denetlendi.
- Dört iç aday, iki tam veri eğitimi, 32 dış değerlendirme hücresi tamamlandı.
  Dış hücrelerin 16'sı aynı önceki v7/LightGBM referans skorlarıdır.
- Ham veri, GPU kurulum önbelleği ve büyük ara dosyalar `D:\barca` altında.

[Tüm eşli karşılaştırmalar](RESULTS.md) · [Yöntem ve komutlar](METODOLOJI.md) ·
[Dondurulmuş seçim](selection.json) · [Model hash'leri](final_manifest.json) ·
[Örnekleme denetimi](sampling_audit.json) · [Test günlüğü](tests.log)
