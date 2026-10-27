# Paylaşılan üç BiGRU checkpoint'i

Bunlar mevcut BAR-ATH ensemble'ını üreten **native DFL + kontrollü PFF**
modelleridir. Sonradan denenen kontrollü DFL modelleri değildir.

| Dosya | Seed | Boyut |
|---|---:|---:|
| `bigru_s20261025.pt` | 20261025 | 594553 bayt |
| `bigru_s20261026.pt` | 20261026 | 594553 bayt |
| `bigru_s20261027.pt` | 20261027 | 594553 bayt |

Toplam yaklaşık 1.78 MB; normal Git dosyalarıdır, LFS gerekmez.
SHA256, yapılandırma ve sıralı özellik adları: [manifest.json](manifest.json).
Dosyalar `cache/seed_balance_v1/` altındaki orijinallerin birebir kopyasıdır.

## Eğitim kaynağı

7 açık DFL maçı ve erişim formuyla edinilmiş 64 Gradient/PFF Dünya Kupası maçı.
DFL native, PFF `raw_control_v2` kontrollü etiketler; kaynak örneklemesi 50:50.
BAR-ATH eğitim veya model ayarı seçimine girmedi. Metrica/SkillCorner eğitimde
kullanılmadı. İç ayar seçimi ayrılmış DFL/PFF maçlarında yapıldı, son modeller
tüm 7+64 maçta 11 epoch eğitildi.

"Yalnız açık lisanslı veri" ifadesi bu notla garanti edilmiyor: PFF verisi
sağlayıcının erişim koşullarıyla alındı. Burada ham DFL/PFF/BAR-ATH verisi veya
etiket dosyaları yayımlanmıyor; yalnız model state_dict'i, 172 özellik için
scaler ortalama/ölçeği ve ağ/pencere ayarları var.

## BAR-ATH dosyasını üretmek

1. Reponun bu klasörü ve `src/reproduce_shared_bigru.py` dosyasını içeren
   sürümünü çekin. Eski deney sürücülerinin yerel Windows yol/manifests
   kontrollerini kullanmak zorunda değilsiniz.
2. Python 3.12 ortamını kurun. Test edilen ortam `requirements-gates.lock.txt`
   içinde; Torch sürümü **2.9.0+cu128**, CUDA 12.8. CUDA Torch paketi için:

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gates.lock.txt
```

3. Resmî scorer deposu `external/more-than-a-hack-2027` altında bulunmalı.
   Önceki deneylerde kullanılan sürüm:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
git -C external/more-than-a-hack-2027 checkout f8f2629ff1e0985f81d7e59d8413678514ee1530
```

4. Katılımcı olarak edinilmiş kendi BAR-ATH paketinizi `data/` altına koyun
   veya `MTAH_DATA_DIR` ortam değişkenini paket köküne yönlendirin. Gerekli
   dosyalar: tactical ve broadcast ATD `.txt`/`.xml`, homography `.csv` ve
   tactical smart-tagging `.xml`. Videolar ve DFL/PFF ham eğitim verisi gerekmez.
   Aynı kod sürümü ve paketle oluşturulmuş özellik önbellekleri kullanılabilir;
   farklı özellik denemelerinden kalan bir cache ile karıştırmayın.

```bash
python src/reproduce_shared_bigru.py --device cuda --score
```

Çıktı: `submissions/shared_bigru/bigru_ensemble_equal.csv`.
Komut üç checkpoint'in hash'ini kontrol eder, ağları yeniden çalıştırır,
P(home) olasılıklarını eşit aritmetik ortalar ve değişmemiş v7 ölü-top/B2
dekoderiyle tüm maçı üretir. Eski tahmin cache'lerini veya seed CSV'lerini
okumaz. Özellik cache'lerini kullanabilir; yoksa BAR-ATH girdilerinden üretir.
`--score` yalnız tahmin tamamlandıktan sonra açık bölüm kontrolünü ekler.

Beklenen sonuç:

- 199319 veri satırı, `frame,label` başlığı.
- Açık bölüm spellF1 **0.7724137931034483**, macroF1 **0.9432893901972056**.
- CSV SHA256: `f4a4b86501fab3e94e5ef9e288b064a431818c35908982678af39c0ee1cfd225`.

27 Ekim 2026'da paylaşılan kopyalardan, mevcut özellik cache'iyle ağ çıkarımı
yeniden çalıştırıldı; bu hash birebir üretildi. Doğrulanan donanım RTX 5070 Ti
Laptop / CUDA. CPU seçeneği vardır (`--device cpu`), ancak CPU ve farklı
Torch/GPU sürümlerinde aynı CSV ayrıca doğrulanmalıdır. Hash farklıysa komut
hata verir; çıktı sessizce eşdeğer sayılmaz. Kontrol kaydı
`cache/shared_bigru/verification.json` altındadır.

## v10 veya başka bir dekoderle birleştirme

BiGRU yalnız takım olasılığını verir; bağımsız bir ölü-top modeli değildir.
Bu komutun oluşturduğu `cache/shared_bigru/barath_equal_probabilities.pkl`,
yarı başlangıç karesi -> `(None, P_home_array)` biçimindedir. Kare olasılıkları
25 Hz'dir; her yarı ayrı çıkarılır. Dosya BAR-ATH türevi olduğundan yerelde
kalmalıdır.

Dış maçlarda aynı v7 özellik sırasıyla kullanım:

```python
import torch
import seq_pff as seq  # src/ import yolunda olmalı

# Önce manifestteki SHA256'yı doğrulayın. Yalnız bu güvenilen checkpoint'i yükleyin.
ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
c = ck["config"]
net = seq.Net(345, c["kind"], c["width"], c["drop"]).to(device)
net.load_state_dict(ck["state"])
net.eval()
scaler = seq.Scaler(ck["mean"], ck["scale"])
p_home = seq.predict_raw(net, scaler, seq.views(features), len(features["XF"]))
```

Üç seed'in **olasılıklarını** float64 olarak eşit ortalayın; sert etiket oyu
veya seed skorlarının ortalaması aynı ensemble değildir. Dış testte iki
yöntemin ölü-top kanalı, dekoderi, girdileri ve etiket tanımı aynı tutulmalı.
v10 birleşim ağırlığı dış maçlarda veya BAR-ATH açık pencerede seçilmemeli;
eğitimden ayrılmış maçlarda dondurulmalıdır. Bu paylaşımda v10 birleşimi
denenmedi ve yarışmaya gönderim yapılmadı.
