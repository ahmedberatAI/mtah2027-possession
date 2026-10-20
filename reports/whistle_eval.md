# Hakem düdüğü tespiti ve D sınırı zamanlaması — değerlendirme

*20 Ekim 2026 · tüm sayılar toplu istatistik (veri politikası: ham satır / görüntü / spektrogram yok; kare bazlı
çıktılar yalnızca `cache/` içinde). Ayrıntılı çıktı: `reports/whistle_eval_output.txt`.*

## Karar

1. **Hazır AudioSet modelleri (PANNs CNN14, "Whistle" sınıfı) hakem düdüğünü ayırt etmiyor.** Lokal CPU'da üç
   varyant (SED 32 kHz, clip 16 kHz, clip 32 kHz) denendi. v5 D başlangıçlarına yakınlıkta lift ≈ 1.0–1.9 (anlamsız ya da
   zayıf). En iyi PANNs varyantı (clip16 + 30 sn medyan çıkarma) top-200'de lift 2.5'e çıkıyor, ama iki santra
   düdüğünü de kaçırıyor (persentil 89 / 93). Yani düdüğü değil, muhtemelen tribün tepkisini yakalıyor.
2. **Santra düdüğünden örnekle arama (query-by-example) açıkça ayırt edici.** `src/whistle_ref.py`'nin dayandığı
   fikir şu: hakem tüm maç aynı düdüğü çalar, ses parmak izi (ana ton ~3.8–4.3 kHz ve harmonikleri) sabittir; tribün
   ıslıkları ise rastgele perdelerde. Detektör, 1. yarı santrasının şablonuyla 2. yarı santrasındaki düdüğü **~3000
   tepe arasında 5. sırada** buluyor (−1.24 sn). Tersi de geçerli: 2. yarı şablonu 1. yarınınkini **9. sırada**
   buluyor (−1.08 sn). v5 D başlangıçlarının ±1 sn içinde **12/118 isabet**, rastgele %3 (**lift 3.2, p = 4e-4**).
   En güçlü 100 tepenin **%74'ü tahmini D içinde** (D payı %50).
3. **`model_v6` (v5 + düdük) D başlangıç zamanlamasını biraz iyileştiriyor, başka hiçbir şeyi değiştirmiyor.**
   Tam maçta **9 D başlangıcı öne çekiliyor** (medyan 0.56 sn). Değişen kareler 135 (%0.08), hepsi canlı→D yönünde.
   Yeni ya da silinen segment yok, takım değişimi yok.
   - Public pencere: macroF1 **0.901 → 0.908**, alive/dead F1 0.974 → 0.977, geçiş-F1 (±1 sn) **0.389 → 0.444**.
     **spellF1 değişmedi: 0.566.** Tam maç tanıları ve tüm vekil metrikler birebir aynı.
   - Kazanç küçük ve resmi gürültü bandının içinde (macroF1 farkı < 0.01, 4 D başlangıcından 2'si etkilendi).
     Ama yön tutarlı ve parametrelere karşı kararlı. **Düşük riskli bir iyileştirme; sıralama metriği (spellF1)
     etkilenmiyor.**

## 1. Detektör karşılaştırması

v5 olayları = `submissions/baseline_v5.csv`'deki 118 canlı→D ve 117 D→canlı geçişi (değerlendirilen kareler).
İsabet = skorun en güçlü 100 yerel tepesinden (±1 sn bastırma) biri pencereye düşüyor mu? Rastgele = aynı pencere tüm
değerlendirilen karelere ya da tüm tahmini-canlı karelere konunca isabet oranı (büyüğü). p = tek yönlü binom testi.

| Detektör | Santra 1: persentil / sıra / ofset | Santra 2: persentil / sıra / ofset | D başl. ±1 sn: isabet (rastgele) lift, p | D başl. [−3,+1] sn: lift | D bitişi [−4,+1] sn: lift | Top-100 tepe D içinde |
|---|---|---|---|---|---|---|
| Eski bant tonalitesi (`whistle.py`) | 99.91 / 56 / −1.08 sn | 99.61 / 184 / −1.24 sn | 4/118 (0.03) 1.1, p=0.52 | 0.8 | 1.6 (p=0.05) | 0.56 |
| PANNs SED "Whistle" | 97.30 / 196 / −1.80 sn | 98.06 / 147 / −1.60 sn | 5/118 (0.03) 1.3, p=0.33 | 1.0 | 0.9 | 0.61 |
| PANNs clip16 "Whistle" −medyan | 88.76 / 589 / −3.00 sn | 92.86 / 431 / +0.20 sn | 8/118 (0.03) 2.1, p=0.04 | 1.7 | 1.0 | 0.63 |
| PANNs clip32 "Whistle" | 99.64 / 35 / −1.24 sn | 99.68 / 30 / −0.88 sn | 7/118 (0.03) 1.9, p=0.09 | 1.5 | 1.1 | 0.59 |
| **Santra şablonu (`ref:s`)** | **sıra 9, −1.08 sn** (dışarıda bırakılmış) | **sıra 5, −1.24 sn** (dışarıda bırakılmış) | **12/118 (0.03) 3.2, p=4e-4** | **2.3 (p=2e-3)** | **2.2 (p=9e-4)** | **0.74** |

- Şablon satırındaki santra sıraları **leave-one-out** (bir santranın şablonu → öbür santra). İki şablonun ortalaması
  (`ref:s`, v6'da kullanılan) kendi santralarında 5. / 8. sırada.
- Kendi kendini eğitme (en güçlü 30 tespitten şablon, santralar dışarıda): ±1 sn lift 3.8 (14/118). Fark gürültü
  düzeyinde, bu yüzden v6 daha basit iki-santra şablonunu kullanıyor.
- **Public GT çevresinde olay-tetiklemeli ortalama (n = 4 D başlangıcı, sağlam z-skoru):** [−1,0] / [0,+1] sn
  aralığında şablon +4.3 / +3.4. Diğerleri: clip32 +2.4 / +1.6, SED +1.6 / +0.9, eski detektör +1.3 / +0.1.
  4 GT D başlangıcından 2'sinde top-100 düdük var ve sınıra çok yakın: **−0.28 sn ve +0.08 sn**.
  4 GT D bitişinin (yeniden başlama) hiçbirinde düdük tepesi yok.
- Düdüğün yeri: top-100 tepenin 26'sı v5-canlı karelerde. Bunların 12'sinde v5'in D başlangıcı sonraki 3 sn
  içinde, yani v5 geç kalmış. 74'ü v5-D içinde, 50'si duraklamanın başından >10 sn sonra (serbest vuruş
  işaretleri, kartlar vb.).

## 2. model_v6

**Tasarım.** v5 skorları + Viterbi'de **D'ye giriş bonusu** (`wh_bonus = 30`), düdük tepesinin ±0.32 sn'sinde.
Bonus yalnızca *v5 geç kaldığında* uygulanıyor: düdük anında v5 topu canlı sayıyor ve v5'in kendi canlı→D geçişi
sonraki 3 sn içinde geliyor. Bu kısıt şart. Bonus her düdüğe verilince Viterbi D başlangıçlarını düdüğe kadar
*geciktiriyor* (11 hareketin 6'sı geriye) ve duraklama ortasındaki düdükleri tekrar toplamak için duraklamaları
bölüyor. `wh_bonus=80, top-200` ile D segmenti 119 → 160 oluyor, public spellF1 0.566 → 0.533 düşüyor.

**compare.py** (`FEAT=features_merged.pkl`):

```
model                         macro  spell  np | Npos  mean   Nd   dsh |  tagA  spOK   rst  bout
model_v5                      0.901  0.566  12 |  261  12.2  119 0.501 | 0.866 0.815 0.932 0.891
model_v6                      0.908  0.566  12 |  261  12.1  119 0.502 | 0.866 0.815 0.932 0.891
```

**Resmi skorlayıcı, public, 1000 bootstrap:**

| | macroF1 [%5–95] | spellF1 [%5–95] | alive/dead F1 | geçiş-F1 ±0.5 sn | geçiş-F1 ±1 sn | eşleşen geçiş |
|---|---|---|---|---|---|---|
| v5 | 0.9010 [0.771–0.937] | 0.5655 [0.359–0.760] | 0.9737 | 0.167 | 0.389 | 7/20 |
| v6 | 0.9081 [0.777–0.944] | 0.5655 [0.359–0.760] | 0.9769 | 0.222 | 0.444 | 8/20 |

Public D-başlangıç ofsetleri (tahmin − GT, kare): v5 `+7 +44 −116 −1` → v6 `+1 +10 −116 −1`. D bitişleri
değişmiyor (`+4 +18 +16 +49`).

**Kararlılık.** `wh_bonus` 15–100 ve öne çekme sınırı 2–5 sn aralığında sonuç birebir aynı (aynı 9 hareket). Top-50
düdükle 4 hareket (macroF1 0.907). Top-200 ile 11 hareket (0.905); buradaki fazladan bir hareket, zaten 4.6 sn erken
olan public sınırı daha da erkene çekiyor. Taşınan 9 sınırın hiçbirinde v5'in başlangıcında top saha dışında
değildi. Yani taç/aut/korner sınırlarına dokunulmadı; düdük–faul sınırlarıyla tutarlı.

**Belirleyicilik.** `make_submission.py --model model_v6` iki kez çalıştırıldı: aynı MD5, `validate_submission.py`:
Valid. v6 sadece numpy/scipy kullanıyor; torch gerekmiyor.

## 3. Sınırlar ve sonraki adımlar

- Public pencerede yalnızca 4 D başlangıcı var. Tam maçtaki 9 hareket etiketsiz doğrulanamaz. Elle etiketlenmiş
  doğrulama seti (`tools/labeler.html`) çıkınca düdüğe çekilen sınırlar ilk kontrol edilecek şey.
- Yeniden başlama düdükleri: v5'in public D bitişlerinin hepsi geç (+0.16…+2 sn), ama GT D bitişlerinde düdük
  görülmedi (0/4). Bu yüzden D bitişine düdük kanıtı eklenmedi.
- Kaçırılmış kısa duraklamalar: canlı oyunda, yakınında v5 D'si olmayan güçlü düdükler var (top-100'de ~10).
  Ama sadece düdükle segment yaratmak spellF1'i düşürdü (yukarıda).

## Yeniden üretim

```bash
.venv/Scripts/python.exe src/whistle_ref.py                 # şablon detektörü -> cache/whistle_ref.npz (~30 sn CPU)
.venv/Scripts/python.exe src/whistle_eval.py ref:s ref:s1 ref:s2 naive:tonal_ex
FEAT=features_merged.pkl .venv/Scripts/python.exe src/compare.py model_v5 model_v6
.venv/Scripts/python.exe src/make_submission.py --model model_v6 --out submissions/baseline_v6.csv
# yalnızca PANNs karşılaştırması (torch CPU + panns_inference, varyant başına ~6 dk):
.venv/Scripts/python.exe src/whistle_panns.py sed clip16 clip32
.venv/Scripts/python.exe src/whistle_eval.py sed:Whistle clip16:Whistle@exc clip32:Whistle
```

- Yeni dosyalar: `src/whistle_ref.py`, `src/whistle_panns.py`, `src/whistle_eval.py`, `src/model_v6.py`.
  Mevcut hiçbir dosya değiştirilmedi.
- PANNs için `.venv`'e uv ile kuruldu: `torch 2.14.1+cpu`, `panns_inference 0.1.1` (+ `librosa`, `torchlibrosa`).
  Ağırlıklar Zenodo 3987831'den `~/panns_data`'ya indirildi ve `torch.load(weights_only=True)` + yalnızca numpy
  dizi kurucuları allowlist ile yüklendi. Ses yalnızca lokal işlendi; hiçbir bulut servisine gönderilmedi.
- `cache/` altındaki kare bazlı çıktılar: `tactical_audio_32k.wav`, `whistle_panns_{sed,clip16,clip32}.npz`,
  `whistle_ref.npz`, `whistle_ref_P.npy`.
