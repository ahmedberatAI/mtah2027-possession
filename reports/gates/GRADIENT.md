# Gradient/PFF erişimi ve veri denetimi — 24 Ekim 2026

Kullanıcının formu gönderme talimatı üzerine, Git yapılandırmasındaki ad/e-posta
ve kurum alanında `Independent research` kullanılarak başvuru gönderildi.
Form doğrudan veri bağlantısını verdi; e-posta beklemek gerekmedi.

## Yerel veri

`D:\barca\mtah2027-possession\external_data\gradient` altında:

| İçerik | Maç/dosya | Boyut |
|---|---:|---:|
| Olay JSON | 64 | 1,149,120,881 bayt |
| Sıkıştırılmış takip JSONL | 64 | 2,776,070,163 bayt |
| Metadata JSON | 64 | 63,141 bayt |
| Kadro JSON | 64 | 483,962 bayt |
| Olay v2.5 ve takip v2.2 şema PDF | 2 | 1,010,466 bayt |

Toplam 258 kaynak dosyası, **3,926,748,613 bayt (~3.93 GB)**. Bütün ham dosyaların
boyutu sağlayıcı metadata'sıyla karşılaştırıldı. SHA256 manifesti:
[gradient_manifest.json](gradient_manifest.json). Veri ve özel indirme referansları
gitignore kapsamındadır; uzak repoya yüklenmedi.

## İlk varsayım artık doğrulanmış durumda

64 maçın tamamındaki 144,541 olay satırında:

| Olay kodu | Anlam | Satır sayısı |
|---|---|---:|
| PA | Pas | 66,715 |
| IT | İlk dokunuş | 47,097 |
| CH | Mücadele | 11,555 |
| RE | Sekme / kontrolsüz temas | 2,544 |
| TC | Dokunuş alt türleri | 1,290 |

Bu sayılar dosya satırlarıdır; ayrı fiziksel dokunuşların eksiksiz sayımı değildir.
Bir ilk dokunuş başka bir olayla çakışırsa kod IT yerine o olayın kodu olabilir.

Olay şeması (v2.5, s. 23–25) RE'yi kontrol veya dağıtma niyeti olmayan temas olarak
tanımlar. TC içinde başarısız trap/catch, ağır dokunuş ve takım arkadaşına bırakma
ayrıdır. CH sonuçları içinde rakibin temasına rağmen topu tutma (`K`), kazananın
kontrolü sürdürmesi (`R`) ve topun başka oyuncuya gitmesi (`P`) gibi ayrımlar vardır.
Dolayısıyla mücadele kazananını her durumda yeni possessor saymak gerekmiyor.

Bu, metindeki **etiket kalitesi bahsini destekler**. Ancak alanların bulunması,
yarışmanın özel GT'siyle eşdeğer etiket üretildiği anlamına gelmez. Bu oturumda
bu veriyle yeni bir takım/sıra modeli eğitilmedi.

## Doğrudan eğitime eklemeyi engelleyen teknik farklar

10511 numaralı tek takip dosyasının akış denetimi:

- 196,098 satır, 195,468 benzersiz kare: **630 yinelenen kayıt**.
- Kare başına yaklaşık **33.3667 ms (~29.97 fps)**; mevcut düzenek 25 fps.
- Bir kare aralığı 2,002 kare atlıyor; dosya satır sırası zaman ekseni sayılamaz.
- 143,305 satırda `home_ball` boolean değil/eksik. Bu alan yoğun bir kare GT'si
  gibi kopyalanamaz; eksik değer de doğrudan ölü top demek değildir.
- Aynı karede çelişen `home_ball` bulunmadı; bu sadece denetlenen örnek dosya için.
- Takip dosyası 1 ve 2. periyotları içeriyor; aynı maçın güncel olay dosyasında
  3 ve 4. periyotlar da var. Takipsiz uzatma olayları eğitimde kullanılmamalı.

Takip ve olay dosyalarının farklı sürümleri/naming kuralları var: eski tracking
`home_ball`/`game_event`, yeni event JSON `homeTeam`/`gameEvents`/`sequence` kullanıyor.
Eşleme event ID, periyot ve gerçek video zamanıyla doğrulanmalı.

Tüm olaylarda 1,082 satır `initialNonEvent`, 1,081 satır `nonEvent`, 4,616 satır
`videoMissing` işaretli. Bunlar üst üste gelebilir. Etiket maskesi için göz önünde
bulundurulmalı; geçersiz veya eksik kayıtlar sessizce kesin etikete çevrilmemeli.

## Etiket dönüştürücüsü için sonraki adım

**Güncelleme:** Aşağıdaki adımlar tamamlandı; 64 maç / 134 devre 25 Hz'e
hizalandı, etiketler üretildi ve üç model eğitildi. [Sonuç](../pff/SONUC.md),
[uygulanan yöntem](../pff/METODOLOJI.md). Aşağıdaki liste ilk denetimin planıdır.

1. `(maç, periyot, video zamanı)` ekseninde tekrarları çözmek; büyük zaman
   boşluklarını koruyarak 25 fps'e yeniden örneklemek.
2. Oyun dışı ve yeniden başlama aralıklarını OTB oyuncu kontrol aralıklarından
   ayrı üretmek; seyrek sahiplik alanını doğrudan ileri doldurup D kaybetmemek.
3. RE ve başarısız kontrolü tek başına takım değişimi saymamak. Mücadelede sonraki
   kontrol olayını kullanmak; anlaşmazlıklarda eğitim maskesi bırakmak.
4. Native ve muhafazakâr kontrollü etiketleri ayrı tutup aynı dış testlerle
   karşılaştırmak. Yarışmanın upstream GT konvansiyonu sorusu hâlâ açık.

## Kaynak ve kullanım durumu

- [Erişim formu](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)
- [Sağlayıcının veri klasörü](https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa)
- [Olay şeması v2.5](https://drive.google.com/file/d/1Q5Wg-q2lOdFreqyZqfmWKBXLPE8HJopt/view)
- [Takip şeması v2.2](https://drive.google.com/file/d/17kl9XOlA932IU_K6969VYDYntYJt3Pan/view)
- [Değişiklik günlüğü](https://docs.google.com/document/d/1GfeBwyUHV0lZAgoDJqQ0rMwfhv7wASMYSpxNh4xNf1I/edit)

Erişim ücretsiz form üzerinden sağlandı. Klasörün kökünde ayrı açık lisans dosyası
bulunmadı; PDF'ler proprietary ibareli. Bu veri için CC BY lisansı veya yeniden
dağıtım hakkı varsayılmadı. Kaynak dosyaları yalnız yerelde tutuluyor.

Makine tarafından üretilmiş toplu denetim: [gradient_audit.json](gradient_audit.json).
Yeniden çalıştırma: `.venv/Scripts/python.exe src/audit_gradient.py`.
