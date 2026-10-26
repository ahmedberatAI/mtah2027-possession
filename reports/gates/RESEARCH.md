# Kaynak ve yöntem incelemesi — 24 Ekim 2026

> **Sonraki güncelleme:** Form kullanıcı talimatıyla gönderildi; 64 maç indirildi.
> Önceden erişim bekleyen alanların denetimi [GRADIENT.md](GRADIENT.md) dosyasında.

Çalışma deposu: `681a731517bcb0ee1399a899afaba52189835eed`.
Resmi yarışma deposu: `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
Yeni sonuçlar `learning_curve.json`, `contacts.json` ve `label_sensitivity.json`
dosyalarındadır. Aşağıdaki eski sonuçlar bu oturumda yeniden ölçülmüş değildir.

## Önceki yaklaşımın özeti

- v1–v3: geometrik takım kanıtı, canlı/ölü ayrımı, yayın/taktik kamera hizalaması,
  minimum süreli Viterbi. v3'te ölü-top ayrımı güçlü; takım değişimi darboğaz.
- v5: v3 canlı/ölü kuralları + açık DFL'den öğrenilen LightGBM H/A olasılıkları.
- v7: sim V3 (zamanda pürüzsüz gürültü), `own_half` NaN düzeltmesi, baskı altında
  kontrol, formasyon/şekil/hareket özellikleri; B2 süre cezası ve takım geçiş cezası 12.
- Repo günlüğündeki public spellF1 v5 0.566, v7 0.652; bunlar sadece 17 possession
  içeren örnek pencereden. Tam maç sonucu veya yeni deney sonucu değildir.
- SkillCorner dış testinde eski v7 kazanımı yaklaşık +0.017. DFL-native / Metrica-events
  ile kontrollü etiketler farklı modelleri/dekoderleri ödüllendiriyor.
- Önceki A1 denemesi, DFL'den türetilmiş `trk` kontrollü etiketlerle eğitimi zaten
  sınamış: DFL-trk 0.756→0.749, Metrica-ctrl 0.808→0.802. Dolayısıyla yalnız
  mevcut DFL etiketlerini filtrelemek yeni ve kanıtlanmış bir kazanç yolu değil.
  Gradient bahsinin farkı daha fazla ve daha iyi açıklanmış olay denetimi olmalı.
- BiGRU −0.04…−0.10; veri azlığının sebep olduğu kanıtlanmış değil. Alan farkı,
  etiket farkı, mimari/eğitim düzeni de olası. 64 maçın otomatik olarak bunu
  düzelteceği sonucuna varılamaz.
- Oyuncu kontrolcüsü, rol özellikleri, zamansal yığınlama ve çoğu geçiş düzeltmesi
  tutarlı dış kazanım sağlamamış. Önceki video pilotu yayın videosunda pas/top sürme;
  pas çapası simülasyonu iyimser ve sınırlı kazanç vermiş.

## Bahis 1 — Gradient / PFF

[Gradient duyurusu](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset)
64 maçta eşlenmiş yayın takip ve olay verisini doğruluyor. Güncellenmiş paket maç bazında
olay/metadata/roster dosyaları içeriyor. 64/7 = 9.14; DFL'ye eklenirse 71/7 = 10.14.
Bu sayı ham maç miktarıdır; etkin etiketli örnek sayısı ve alan uygunluğu değildir.

Yeni duyuru sayfasında veri formu görüntülenmedi; çalışan eski form
[PFF'nin özgün duyurusunda](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)
bulundu. Kullanıcı yanıtından sonra Git profilindeki iletişim bilgileriyle gönderildi.
64 maç indirildi; challenge/rebound/touch alanları doğrulandı. Ayrı bir açık lisans
dosyası bulunmadı; PDF'ler proprietary ibareli. “Ücretsiz erişim” açık yeniden
dağıtım lisansı demek değildir. Güncel denetim: [GRADIENT.md](GRADIENT.md).

Veri geldiğinde ilk kontrol: olay sözlüğü, kontrollü/kontrolsüz dokunuş ayrımı,
duel sonucu ile sonraki kontrol arasındaki bağ, ölü-top/restart zamanları,
takım kimlikleri, tracking-event senkronizasyonu, örneklem hızı ve saha koordinatları.
Belirsiz olaylarda etiketi kesinleştirmek yerine eğitim maskesi kullanılmalı.
Sırf mücadele sınıfının bulunması yarışmanın GT'sini birebir üretmeye yetmez.

## Bahis 2 — temas testi ve SoccerTrack

14 vs 38 dokunuş/canlı dakika, önceki eşikli tracking dedektörünün çıktısıdır;
gerçek temas sayımı değildir. Alanlar arasında gürültü, smoothing ve topun
gözlenme biçimi değiştiğinden bu oran tek başına kaç temasın kaçırıldığını
belirlemez. Yeni deney, ek olay bilgisinin değerini ayrıca sınar.

`exp_anchor.py` pasların yanında pas alımı, recovery/interception, şut, taşıma,
uygun yeniden başlama ve kazanılmış/faulsüz mücadele vekillerini çıkarabiliyor.
Kaybedilmiş mücadele/pressure/on-ball engagement otomatik temas sayılmıyor.
SkillCorner'ın player-possession başlangıç ve uygun bitişleri kullanılıyor.
Bu, olay dosyalarının kapsadığı temasların testidir; her dribbling dokunuşu veya
kontrolsüz sekme her iki kaynakta da tam etiketli değildir.

Ana senaryo: olay precision 0.70, recall 0.70, doğru olaylarda takım doğruluğu 0.95,
zaman hatası std 3 kare (120 ms). Ayrı team70 senaryosu precision=1.0, recall=.70,
takım doğruluğu=.70. Kusursuz olay tavanı ayrıca ölçülür. β=4 önceki deneyden sabit;
β=2 duyarlılık. Pencere eski pas deneyindeki [−12,+3] karedir ve iki akışta aynıdır.
Alım/araya girme için temas öncesi pencere ideal olmayabilir; sonuç bu füzyonu sınar.

İlk Metrica sonuçlarında bu sakınca görüldükten sonra **keşifsel** `contacts_typed`
duyarlılığı eklendi: release için [−12,+3], reception/recovery için [−3,+12],
duel için [−3,+3] kare. Bu varyant önceden belirlenen ana test değildir; olay
türünün doğru bilindiğini varsaydığından ayrıca iyimserdir. Sonradan en yüksek
skorlu varyant seçilip doğrulanmış sonuç diye sunulmaz.

Etiketler ve sentetik olaylar aynı olay kaynağından geldiği için deney **iyimser bir
bilgi değeri testi**, bağımsız video detektörü doğrulaması değildir. Yalnız pozitif
sonuca dayanarak taktik videoda aynı kazanç beklenemez. <+0.03 sonucu bu test edilen
yaklaşım için durdurma işaretidir; mümkün her video yaklaşımının üst sınırı değildir.

[SoccerTrack veri kartı](https://huggingface.co/datasets/atomscott/soccertrack-v2):
10 maç, tüm sahayı kapsayan panoramik video, oyuncu konumları ve 12 olay sınıfı.
Veri CC BY 4.0; [kod MIT](https://github.com/AtomScott/SoccerTrack-v2).
Header, Ball Player Block ve Player Successful Tackle sınıfları mevcut.
Ancak Out/Goal temas sınıfı değildir; 12 sınıf “bütün fiziksel temaslar” demek değildir.
v1.2 kartı 21,428 olay bildirir (yerel sayım değil).

v1.2 revision: `325bd4c57ee8e135ab652f3921d274526ada496b`.
API `gated=auto` döndürüyor; olay dosyası anonim indirmesi 401 verdi. Veri kartı ve
dosya listesi okunabildi, olay içeriği denetlenemedi. Yetkili erişim sonrası
`src/audit_soccertrack.py` küçük anotasyonları indirip sayım yapar. Büyük videolar
henüz indirilmedi. **Auxiliary ball** konumları olaylara bağlanıp arada doğrusal
enterpole ediliyor; aynı olayların bağımsız takip kanıtı olarak kullanılmamalı.
Yeni v1.2 kartında zaman ilk video karesine göre; eski GitHub format dokümanındaki
kickoff ifadesiyle fark var. Kullanılacak release'e göre hizalama doğrulanmalı.

## Bahis 3 — tanım biliniyor, üretim mekanizması hâlâ bilinmiyor

[Güncel CHALLENGE.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md)
kontrolsüz dokunuş, kontrolsüz hava topu ve pas sapmasının possession değiştirmediğini
belirtiyor. Fakat bu tanımın GT üretiminin teknik spesifikasyonu olmadığını da açıkça
söylüyor. GT, düzeltilmiş tracking'in dead/owning alanlarından mekanik olarak
çıkarılıyor; sonrasında smoothing/interpolation/elle düzeltme yok.

Bu, düzeltilmiş tracking'in **owning alanını upstream'de kimin/nasıl ürettiğini**
açıklamıyor. Operatör mü, yakınlık modeli mi, hibrit mi: açık kaynaklarda doğrulanamadı.
Metrica örnek maçları da bu alanı sunmuyor; repo kendi olaydan türetilmiş etiketlerini
kullanıyor. `ctrl` vekili yarışmanın özel GT'siyle özdeş kabul edilemez.

Organizatöre gönderilmek üzere soru taslağı (gönderilmedi):

> The challenge explains the mechanical export from corrected tracking. Could you
> clarify how the upstream team-owning and ball-dead fields are produced: human
> annotation, an automated rule/model, or a hybrid? Are ownership changes anchored
> to first contact or the first controlled action, and do uncontrolled deflections
> and aerial-duel wins without subsequent control retain the previous team? We are
> asking about the annotation convention, not any private-match labels/statistics.

## Güncel kural farkı

[RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
sağlanan maçta **eğitim ve doğrulama dahil hiçbir amaçla elle etiketlemeye** izin
vermiyor. Eski repo notları güncelliğini yitirmiş. README ve deney günlüğüne düzeltme
eklendi. Yeni deneyler yalnız açık dış maçlarda; BAR-ATH public pencereyle ayar yok.
