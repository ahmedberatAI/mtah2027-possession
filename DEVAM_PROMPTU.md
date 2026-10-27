# Güncel devam noktası — 27 Ekim 2026, kullanıcı ara verdi

Önce `AGENTS.md` ve `docs/RAPOR.md` §4 **Tur 13** kaydını oku. Aşağıdaki eski devam notundaki
112 değişiklik / GitHub birleştirmesi bilgileri artık tarihsel: tur 9–12 `541b79e` ile korundu,
`origin/main` (`39fd4de`) `de5d862` ile birleştirildi. Dal: `codex/rounds9-13-integration`.
Push yapılmadı; push için onay gerekiyor. Kullanıcı ara verdiği için hiçbir deney kendi kendine yeniden başlatılmaz.

- v10 ve paylaşılan BiGRU referans CSV'leri birebir üretildi; 52 test geçti, ağırlık hash'leri doğrulandı.
- `src/exp_shared_blend.py` tamamlandı: 216 dış skor hücresi, üç eşli tohum. %75 BiGRU olasılık karışımı
  SC fazda ≈ +0.022, pp'de ≈ +0.029; Metrica ≈ 0. DFL LOMO eksik; varsayılan **v10 değişmedi**.
- Tam sonuçlar `reports/shared_blend/RESULTS.md`; ayrıntılı gerekçe ve güven aralıkları RAPOR Tur 13.
- Dokunuş v2 tohum 1 bitti, DFL-evctrl kazancı tekrarlanmadı. Tohum 2 nihai dosyayı yazmadan durduruldu;
  eğitimi baştan çalışacak. Ardından yeni `src/exp_touch_verify.py` üç tohum özeti çıkarır (henüz çalıştırılmadı).
- Kamera robust işi durduruldu; core/norm × 3 tohum önbellekleri mevcut, nihai skor tablosu bekliyor.
- İki kendi CPU işi Ctrl-C ile durduruldu; Colab açılmadı, paket/veri/ağırlık indirilmedi.
- PFF 64 maç yerelde yok, bağlı Drive aramasında bulunamadı. Colab yeniden eğitimi için erişim bağlantısı gerekiyor.

---

# Önceki devam notu (tarihsel; yukarıdaki güncellemeler önceliklidir)

# Devam promptu — MTAH 2027 possession modeli (27 Ekim 2026)

Sen bu projede modeli geliştirmeye kaldığımız yerden devam edecek bir kodlama ajanısın. Kullanıcıyla Türkçe konuş; kısa
ve dürüst raporla. Amacın **ciddi kaldıraçlar** (küçük artışlar değil) bulup dış veride doğrulamak ve başarılı olanı
kalıcı yapmak. GPU gereken her işi **Google Colab'da** yap.

## 1. Proje ve hedef

- Yarışma: More than a Hack 2027, Faz 1. BAR-ATH maçının (FC Barcelona – Athletic Club) 199,319 karesinin her birine
  H / A / D etiketi (X: cevap yok). Değerlendirilen kare sayısı 158,857.
- Sıralama: tam maçın gizli GT'sinde **spellF1** (possession spell'leri, IoU 0.3–0.7). Eşitlikte macroF1.
- Son teslim: **17 Kasım 2026**. Teslim paketi:
  - CSV,
  - dosyayı birebir yeniden üreten herkese açık repo (verisiz),
  - kısa teknik doküman.
- GT, Metrica'nın düzeltilmiş takibindeki "topa sahip takım" alanı. Değişim için rakibin **kontrollü** sahipliği
  gerekir; serbest topta son sahip korunur.
- Yerel repo: `C:\Users\omen\Desktop\barcahackathon` (Windows 11; Git Bash + PowerShell; Python `.venv`, 3.12).
  - Komut kalıbı: `.venv/Scripts/python.exe -I src/<betik>.py`.
  - Betikler `sys.path`'e `src/`'yi kendileri ekliyor.
  - Stdin okuyabilen komutlara `< /dev/null` ekle.
- GitHub: `ahmedberatAI/mtah2027-possession` (özel). Ekip arkadaşı **Alperen Aydın** da bu repoya push ediyor.

## 2. Mutlak kurallar (ihlal = diskalifiye riski)

**1. Veri hiçbir barındırılan LLM'e gösterilmez; sen de buna dahilsin.**
- Yarışma şartları: videoların, veri setlerinin ya da bunlardan herhangi bir parçanın bulut / hosted LLM'e verilmesi
  yasak.
- Bu yüzden şunları **asla açma, okuma ya da yazdırma:**
  - `data/` altındaki ham takip satırları, homografi satırları, Smart Tagging satırları,
  - video kareleri, ekran görüntüleri,
  - ses ve transkriptler: `cache/asr/` asla okunmaz,
  - kare bazlı tahminler, `cache/` içeriği.
- Yalnız yerelde betik çalıştır; betikler **yalnız toplu istatistik** (skor, oran, sayı) bassın.
- Veriden türetilmiş rapor ya da grafik hiçbir yerde yayımlanmaz.
- Açık veriler (DFL / IDSSE, Metrica sample, SkillCorner) kısıtsız.

**2. Güncel resmî kurallar** (resmî repo commit `f8f2629`, 23 Ekim):
- Public 5 dakika yalnız hat kontrolü içindir: **eğitim verisi değil, ayar seti değil.** Hiçbir model, parametre ya da
  sürüm seçimi public skora bakılarak yapılmaz.
- BAR-ATH'ta elle etiket hiçbir miktarda ve hiçbir amaçla yok.
- Maç hakkında insan üretimi veri (olay verisi, düzeltilmiş takip) kullanılmaz.
- Başka maçların verisi serbest. Takibi elle düzeltmek de elle etikettir.
- Yerel resmî klonu güncellemek için:
  `git -C external/more-than-a-hack-2027 merge --ff-only origin/main`
  Klon güncel değilse önce `git -C external/more-than-a-hack-2027 fetch origin`.

**3. İndirme:** kullanıcı kendi (kotalı) internetinde. Kullanıcının bağlantısından hiçbir şey (veri seti, ağırlık,
paket) **sormadan indirme**; sorarken boyutu söyle. Büyük şeyleri Colab VM'ine indir.

**4. Commit ve push:** yerel commit ve dal açmak serbest. **Push etmeden önce kullanıcıya sor.** Veri, `cache/` ve
`submissions/` asla commit edilmez.

**5. Ortak süreçler:** başkasının işlerini, oturumlarını ve süreçlerini (Alperen'in Colab oturumları, başka projeler)
durdurma. Google OAuth ve token işlemlerini kullanıcı kendisi yapar.

## 3. İlk iş: repo durumunu birleştir (veri kaybetmeden)

- Yerel çalışma ağacında **commit edilmemiş ~112 değişiklik** var: tur 9–12'nin tamamı.
  - v8–v10, `docs/RAPOR.md`,
  - DFL olay senkronu,
  - kamera deneyleri,
  - teslim sağlamlaştırma.
- GitHub'daki son ortak nokta `681a731` (v7). Alperen onun üzerine iki commit push etti:
  - `a3b3070`: PFF + BiGRU / TCN deneyleri ve raporları.
  - `39fd4de`: 3 BiGRU ağırlığı, `models/bigru_dfl_pff_v1/*.pt` + `manifest.json`, ve
    `src/reproduce_shared_bigru.py` (BAR-ATH için tek başına çıkarım).

Adımlar:
1. `git status` ile neyin ekleneceğine bak. Veri, `cache/`, `submissions/` ve büyük dosyalar `.gitignore`'da olmalı.
   `models/v10_team.txt` (2.85 MB) ve `models/bigru_dfl_pff_v1/*.pt` girecek.
2. Yerel işi yeni bir yerel dalda commit et: örneğin `git switch -c rounds9-12`, sonra `git add -A` ve commit.
3. `git merge origin/main`. Çakışmaları çöz:
   - `README.md`: iki içeriği birleştir.
   - `docs/DENEMELER.md`: biz silip `docs/RAPOR.md`'ye taşıdık. Alperen'in eklediği satırları RAPOR.md §3'e taşı;
     DENEMELER.md silinmiş kalsın.
   - `src/decode2.py`: onun `nogil` satırını al.
   - `.gitignore`: onun `.pt` istisnasını al.
4. Kontrol:
   - `.venv/Scripts/python.exe -I src/make_submission.py` hâlâ `submissions/candidate_v10.csv` üretmeli, MD5
     `ca20c5b657a395d2f7c0e310cdaabd64`.
   - Alperen'in testleri: `python -m unittest discover -s tests`.
5. Push için kullanıcıya sor. Push edersek Alperen de v10'un üstüne kurabilir.
6. Bu klasöre erişimin yoksa (yalnız GitHub görüyorsan): tur 9–12 işi GitHub'da yok. Kullanıcıdan önce bunun push
   edilmesini iste.

## 4. Belgeler ve ölçüt (önce bunları oku)

- **`docs/RAPOR.md`: tek fikir / deneme / sonuç belgesi. Yeni fikir ararken ilk bakılacak yer.**
  - §1: güncel özet, işe yarayanlar ve yaramayanlar, dersler.
  - §3: her deneme bir satır.
  - §4: tur notları; Tur 11–12 en güncel.
  - **Her yeni deneme** §3 tablosunun sonuna bir satır olarak eklenir. Uzun anlatım §4'te yeni "Tur N"'e yazılır.
    Model ya da skor değişirse §1 güncellenir.
  - Başarısız denemelere yeni bir gerekçe olmadan dönme.
- Ortak ölçüt `src/bench.py`. Dış bağlamlar `src/exp_a2.py` içindeki `SETS`:
  - DFL-trk ve DFL-evctrl: 7 maç, maç dışarıda bırakılarak. evctrl, DFL olaylarından kontrollü possession;
    `src/dfl_events.py`.
  - Metrica-ctrl: 3 maç × 3 bozulma, oyun dışarıda bırakılarak.
  - SkillCorner faz ve pp: 20 maç, eğitimde hiç kullanılmadı.
  - Her biri öğrenilmiş ve gerçek ölü topla ölçülür.
- **Karar kuralı:**
  - Dış bağlamlarda en kötü durum pişmanlığı.
  - Fark < 0.005 ise mevcut ya da basit olan kalır.
  - Her aday **en az 3 tohumda** ölçülür; eşli maç bootstrap'ı kullanılır.
  - BAR-ATH yalnız etiketsiz tanı içindir (spell sayısı, ölü top payı); public skor yalnız bilgi.
- Simülatör: `sim.V3` (pürüzsüz konum gürültüsü, BAR-ATH'ın toplu istatistiklerine göre top kayıpları).

## 5. Şu anki durum

- **Gönderim:** `v10` (`src/model_v10.py`), düdük yok. Takım modeli `models/v10_team.txt`; yalnız açık veriyle
  eğitilmiş, repoda.
  - `make_submission.py` argümansız çalışınca dosyayı ~1 dakikada üretir.
  - `MTAH_RETRAIN=1` sıfırdan yeniden eğitir; bayt bayt aynı sonucu verir.
- **v10'un bileşenleri:**
  - Ölü top: v3 kuralları (tempo, top dışarıda / görünmüyor / duruyor, yayın homografisi geçersiz → tekrar / yakın
    çekim).
  - Takım: LightGBM, DFL native + 3 Metrica maçı (kontrollü etiket, maç başına ağırlık 2), sim.V3 ile bozulmuş;
    v7 özellikleri + topsuz etkileşim özellikleri (`src/bfree.py`); ±25 kare logit yumuşatma.
  - Decoder: B2 (ceza 12, ölü top 30, en kısa 8 / 50, rampa 175 / 48).
- **Skorlar** (3 tohum ortalaması, öğrenilmiş / gerçek D):

  | Bağlam | v10 | Alperen'in BiGRU topluluğu |
  |---|---|---|
  | DFL-trk | 0.769 / 0.798 | — (bütün DFL eğitimde) |
  | DFL-evctrl | 0.782 / 0.808 | — |
  | Metrica-ctrl | 0.823 / 0.882 | 0.807 / 0.864 |
  | SC faz | 0.597 / 0.759 | 0.622 / 0.782 |
  | SC pp | 0.592 / 0.748 | 0.634 / 0.788 |

  Public (yalnız bilgi): v10 0.667, BiGRU 0.772.
- **Teknik doküman taslağı:** `docs/TECHNICAL_DOCUMENT.md`. Herkese açık repo içeriği `tools/export_public.py` ile
  çıkarılır.
- **Uyum riski:** v2/v3 ölü top sabitleri ilk gün public pencereye bakılarak kuruldu. Hem v10 hem BiGRU gönderimi
  bunları kullanıyor. Dış veride (DFL BallStatus, Metrica, SkillCorner yayın-geçerlilik bayrağı) yeniden türet.

## 6. Önemli dersler (tekrar etme)

- Darboğaz: 4 sn'den kısa oyun içi top kayıpları (ikili mücadele, hava topu) ve aynası olan yanlış değişimler.
  Kusursuz konum bile bu açıdan ≈ +0.005 kazandırıyor; eksik bilgi "topa kim dokundu".
- Model **sık dönen etiketle** eğitilmeli, kontrollü sahiplik kuralını decoder taşımalı. DFL'yi trk ya da evctrl
  etiketiyle eğitmek her yerde kaybettirdi.
- Yeni bir ipucunun modelin emin olduğu karelerdeki uyumu, değerinin ölçüsü değil.
  - Yayın kamerası kadrajı v10'un emin karelerinde AUC 0.95 verdi, ama etiketsiz eklenince SkillCorner'da her ayarda
    kaybettirdi.
  - Değer, belirsiz anlarda ve gerçek etiketle ölçülür.
- Kaybedenler (ayrıntı RAPOR §3):
  - decoder hileleri,
  - büyük / başka öğreniciler (eski BiGRU denemesi, istifleme),
  - rol, kontrolcü, mücadele özellikleri,
  - SkillCorner'ı LightGBM eğitimine katmak (Metrica'yı düşürdü),
  - kendi kendine eğitim,
  - topluluklar,
  - top yüksekliği,
  - spiker adları (Colab'da ASR, seyrek ve ev sahibine yanlı).

## 7. Açık işler ve ciddi kaldıraç adayları (öncelik sırası)

**1. BiGRU'yu doğrula ve v10 ile birleştir.** En somut büyük kaldıraç.
- `src/reproduce_shared_bigru.py` ile Alperen'in BAR-ATH dosyasını yeniden üret; hash'i
  `models/bigru_dfl_pff_v1/manifest.json` ile karşılaştır.
- BiGRU olasılıklarını Metrica ve SkillCorner bloklarında hesapla; onun raporundaki sayıları doğrula.
- **v10 + BiGRU birleşimi:** logit ya da olasılık ortalaması, ağırlık 0.25 / 0.5 / 0.75; ardından ±25 yumuşatma ve
  B2. Dış bağlamlarda 3 tohumla ölç.
- DFL'de BiGRU ölçülemiyor; bunun için Colab'da maç dışarıda bırakılarak yeniden eğitim gerekir.

**2. BiGRU'yu Colab GPU'sunda DFL + PFF + Metrica ile yeniden eğit.**
- Metrica, LightGBM'de en büyük kaldıracımızdı; BiGRU'da yok.
- Metrica'yı oyun dışarıda bırakarak, DFL'yi maç dışarıda bırakarak değerlendir.
- Girdilere v10'un topsuz özelliklerini ekle; çıkışa ±1 sn logit yumuşatmayı uygula.
- PFF verisine Alperen'in hattıyla eriş (`src/exp_pff.py`, `src/seq_pff.py`, `src/exp_seq_pff.py`). PFF erişimi
  form ister; Alperen'de var. Veriyi Colab VM'ine çek, kullanıcının bağlantısına değil.

**3. Kameranın marjinal katkısı** (tur 12, umut verici).
- SkillCorner'da gerçek etiketle öğrenilen ikinci aşama (kameralı eksi kamerasız eş, pp ile eğitim, α 1), v10'un
  logitine eklenince SkillCorner'ın 4 bağlamında da artı: faz +0.014 / +0.015, pp +0.006 / +0.006.
- Sıradaki adımlar:
  - `src/camera_sc.py robust`: zoom ve pan olmadan; kadraj farkı ölçekli.
  - `src/camera.py resid`: BAR-ATH'a uygulama; SkillCorner tanımlarıyla aynı özellikler (`sc_style_features`),
    yalnız etiketsiz tanı.
- Dikkat: BAR-ATH kadrajı SkillCorner'dakinden ~2.5 kat geniş; alan kayması var.

**4. Ölü top sabitlerini dış veride yeniden türet.** Hem uyum hem olası kazanç için.

**5. Dokunuş v2'yi doğrula.**
- Tohum 0'da DFL-evctrl +0.007 / +0.005, ama ~35 aday taramasından seçildi; şans etkisi olabilir.
- `TOUCH_V=2 LGB_SEED=1` ve `LGB_SEED=2` ile `src/exp_touch.py`.

**6. Diğer fikirler** (RAPOR Tur 12):
- R12-4: öğrenilmiş dokunuş-takım dedektörü (Metrica ve senkronize DFL olaylarından).
- R12-5: serbest top durumlu sahiplik HMM'i.
- Yeni eğitim verisi: SoccerTrack v2 (Hugging Face onayı gerekiyor; topu olaylardan kurulmuş, yalnız topsuz
  özelliklerle).

## 8. Colab iş akışı (GPU işleri burada)

- CLI: `google-colab-cli`, WSL Ubuntu-24.04 içinde; Windows'tan şöyle çağrılır:
  `wsl.exe -d Ubuntu-24.04 -- bash -lc "~/.local/bin/colab <komut>"`
  - Komutlarda tam yol kullan; `$C` gibi değişkenler WSL'e geçmiyor.
  - Token `~/.config/colab-cli/` içinde; OAuth'u kullanıcı yaptı, sen dokunma.
- Komutlar:
  - `colab new -s <ad> --gpu T4` (gerekirse `--high-mem`)
  - `colab upload -s <ad> <yerel> <uzak>`
  - `colab exec -s <ad> -f <dosya.py>`
  - `colab download -s <ad> <uzak> <yerel>`
  - `colab stop -s <ad>`
  - `colab usage` (bakiye ~95 compute unit)
  - Yalnız kendi açtığın oturumları durdur; iş bitince mutlaka durdur.
- Kalıp:
  - `src/` ve `colab/` zip'lenip `/content/mtah_code.zip` olarak yüklenir.
  - `colab/launch.py` içindeki `JOB` adı değiştirilip `colab exec -f` ile çalıştırılır: `colab/<JOB>.sh`'i arka planda
    başlatır, log `/content/mtah/logs/` altında.
  - `colab/poll.py` log kuyruğunu gösterir.
  - Örnek iş: `colab/run_asr.sh`. CUDA 12 kütüphane notu orada.
  - VM'de `pip install` ve açık veri indirme (`colab/fetch_open_data.py`) serbesttir; kullanıcının bağlantısını
    kullanmaz.
- Yarışma verisi Colab'a iki yolla gider:
  - **Drive kısayolu + `colab drivemount -s <ad>`:** kullanıcı her VM'de OAuth URL'sini açıp onaylar.
    Yol: `/content/drive/.shortcut-targets-by-id/<id>/{tactical,broadcast}/`.
  - **Yerelde hazırlanmış küçük girdi dosyaları:** özellik ya da ses gibi, `colab upload` ile.
  - Colab yalnız hesaplama içindir: Colab'ın Gemini özellikleri veriye asla yöneltilmez. Çıktılar yine yalnız
    toplu sayı basar.

## 9. Çalışma biçimi

- Her fikir için önce RAPOR'da denenmiş mi bak. Sonra dış veride, 3 tohumla, karar kuralıyla ölç.
- Uzun işleri arka planda çalıştır; bekleme döngüsü kurma.
- Başarılı fikri kalıcı yap:
  - yeni model dosyası (örneğin `src/model_v13.py`),
  - `make_submission.py` varsayılanı,
  - temiz kopyada sıfırdan yeniden üretim ve MD5 kontrolü,
  - RAPOR.md §1 güncellemesi.
  Ardından kullanıcıya kısaca raporla.
- Kullanıcı büyük kaldıraç istiyor. Sonuçları abartmadan, sayılarla ve "ne kadar emin olduğumuzla" birlikte söyle.
