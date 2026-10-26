# Başlangıç sonuçları — 24 Ekim 2026

> **Erişim güncellemesi:** Kullanıcının yanıtından sonra PFF formu gönderildi;
> 64 maçın 256 ham dosyası ve iki şema PDF'i (~3.93 GB) yerelde. CH/RE/TC/IT alanları
> gerçek dosyalarda doğrulandı. Sonraki aşamada 64 maç 25 Hz'e hizalandı ve üç
> model eğitildi: [PFF eğitim sonucu](../pff/SONUC.md). Aşağıdaki kapı testleri
> bu eğitimden önceki sonuçlardır. Veri denetimi: [GRADIENT.md](GRADIENT.md).

Repo `D:\barca\mtah2027-possession` altına klonlandı. Önceki deneyler ve güncel
yarışma kuralları incelendi; 7 DFL, 3 Metrica, 20 SkillCorner maçından CPU test
düzeneği yeniden kuruldu. 15 takım modeli eğitildi. 120 öğrenme-eğrisi hücresi,
240 temas hücresi ve 8 etiket/dekoder karşılaştırması tamamlandı. Tekrarlı
tohumlar bağımsız maç olarak sayılmadı. Yedi kod testi geçti.

**Ana sonuç:** En öncelikli belirsizlik etiket konvansiyonu. Daha çok veri küçük
kazanç sağlıyor. Temas bilgisinin yararı ise kullanılan possession tanımına göre
işaret değiştiriyor. Bu sonuçlardan henüz BAR-ATH özel skor artışı çıkarılamaz.

## 1. Daha çok eğitim verisi: zayıf ama olumlu sinyal

Ölü-top modeli ve B2 sabit; yalnız takım modelinin eğitim maçları değiştirildi.
2 ve 4 maç için yedi dengeli, iç içe alt küme kullanıldı.

| Dış test / etiket | 2 DFL | 4 DFL | 7 DFL | 4→7 farkı |
|---|---:|---:|---:|---:|
| Metrica kontrollü | 0.8022 | 0.8095 | 0.8147 | +0.0052 |
| SkillCorner oyuncu possession | 0.5792 | 0.5813 | 0.5829 | +0.0016 |
| SkillCorner faz | 0.5768 | 0.5811 | 0.5816 | +0.0005 |

Tablo learned-D ile spellF1'dir. SkillCorner oyuncu-possession 4→7 farkının eşli
maç bootstrap %95 aralığı **[−0.0012,+0.0043]**: güçlü artış kanıtı değil.
Oracle D kanıtı altında aynı fark **+0.0044 [0.0019,0.0068]**; takım öğrenmesinin
küçük yararı ölü-top hataları tarafından kısmen örtülüyor.

**Karar:** Bahsi tamamen bırakmıyorum; fakat mevcut LightGBM'de artan veriyle
büyük sıçrama göstergesi yok. Gradient'ten önce küçük bir örneğin olay sözlüğü,
kalitesi ve senkronizasyonu doğrulanmalı. 64 maç/Transformer/TCN eğitiminin faydası
bu deneyle kanıtlanmış değil. Eski BiGRU kaybının yalnız veri azlığından kaynaklandığı
da kanıtlanmış değil.

Gradient'in 64 maçlık paketi indirildi ve şema denetiminden geçti. Kontrolsüz sekme,
mücadele ve başarısız dokunuş ayrımları doğrulandı. Yoğun ve yarışmayla uyumlu
kare etiketlerine dönüşüm henüz yapılmadı; tracking örneği 29.97 fps ve seyrek
sahiplik alanları içeriyor. Ayrıntılar [GRADIENT.md](GRADIENT.md).

## 2. Temas bilgisi: genel füzyon kapıyı geçmedi, tür ayrımı umut veriyor

Ana senaryo: gerçek olayları bulma recall=.70, olay precision=.70, bulunan gerçek
olayın takımını doğru bilme=.95, zaman hatası std=.12 saniye; üç gürültü tohumu.
Bu tanım “takım doğruluğu %70” ile karıştırılmadı; o da ayrıca test edildi.
β=4 sabit, aynı yedi maçlık v7 referansına göre:

| Dış etiket | Yalnız pas | Tüm mevcut temas vekilleri | Türüne göre pencere* |
|---|---:|---:|---:|
| Metrica kontrollü | +0.0017 | **−0.0254** | **−0.0204** |
| Metrica olay | +0.0075 | **+0.0327** | **+0.0469** |
| SkillCorner oyuncu possession | +0.0170 | **+0.0269** | **+0.0430** |
| SkillCorner faz | +0.0127 | **+0.0214** | **+0.0307** |

*Türüne göre pencere, ilk Metrica sonuçlarından sonra eklenen **keşifsel** kontrol:
pas/şut öncesi, recovery/interception sonrası kanıt. Olay türünün doğru bilindiğini
varsayar. Önceden belirlenen ana kapı testiyle aynı kanıt gücüne sahip değildir.

Tür ayrımlı SkillCorner oyuncu-possession kazancı **20/20 maçta**, eşli %95 aralık
**[+0.0384,+0.0474]**. SkillCorner faz etiketiyle de 20/20 maçta pozitif.
Metrica kontrollü etiketlerde ise üç maçın tamamında negatif.

**Karar:** Eski pas füzyonunu tüm temaslara genişletmek için +0.03 kapısı geçilmedi.
Tür ayrımlı olay yaklaşımı SkillCorner'da eşiği geçiyor ve araştırmaya değer; henüz
genellenmiş bir video kazancı değil. Önce etiket konvansiyonu netleştirilmeli;
sonra bağımsız taktik video üzerinde küçük bir pilot düşünülmeli.

Temaslar gerçek video modelinden gelmiyor: dış olaylardan alınan ve yapay hatayla
bozulan **bilgi değeri simülasyonu**. Etiketler aynı olay kaynaklarından üretildiği
için iyimser. Olay dosyaları her fiziksel dribbling dokunuşunu veya sekmeyi kapsamaz.

SoccerTrack v2'nin veri lisansı **CC BY 4.0**, kodu MIT. Ancak Hugging Face erişimi
auto-gated; anonim olay indirmesi 401 verdi. Lisans ve dosya listesi doğrulandı,
olay içerikleri henüz denetlenmedi. Olaylardan enterpole edilmiş yardımcı top izi
bağımsız tracking GT'si olarak kullanılamaz.

## 3. Etiket tanımı: risk ölçüldü, kaynak mekanizması açıklığa kavuşmadı

Aynı model, aynı pen_team=12; yalnız B2 açılıp kapatıldığında:

| Dış etiket | B2'nin spellF1 etkisi (learned D) |
|---|---:|
| Metrica kontrollü | +0.0424 |
| Metrica olay | −0.0345 |
| SkillCorner oyuncu possession | +0.0097 |
| SkillCorner faz | +0.0250 |

Bu karşılaştırma iki dekoderi ayrı ayrı optimize etmez; konvansiyon duyarlılığını
gösterir. Özel GT'nin hangi tanımı fiilen uyguladığını teşhis etmez.

Güncel `CHALLENGE.md`, kontrolsüz dokunuş ve kontrolle sonuçlanmayan mücadelelerin
possession değiştirmediğini söylüyor. GT export'u düzeltilmiş tracking'in dead/owning
alanlarından mekanik. Fakat owning alanının **upstream** üretiminin operatör,
yakınlık kuralı veya model olup olmadığı açıklanmıyor. Bu nokta hâlâ açık.
Organizatöre sorulabilecek net metin hazırlandı, dışarı mesaj gönderilmedi.

## Güncel kural düzeltmesi ve devam sırası

Resmi kurallar artık BAR-ATH'ta **eğitim/doğrulama dahil hiçbir amaçla elle etiketlemeye
izin vermiyor**. Eski repo notlarının üstüne düzeltme eklendi. Yeni testler yalnız
dış maçlarda; public beş dakikayla ayar yapılmadı.

1. Organizatörden owning alanının üretim ve kontrol/sekme konvansiyonunu netleştirmek.
2. İndirilen PFF verisinden zaman hizalaması ve kontrollü etiket dönüştürücüsü üretip doğrulamak.
3. Olay konvansiyonu desteklerse, tür ayrımlı temas yaklaşımını bağımsız taktik videoda doğrulamak.
4. Bu aşamalardan sonra daha büyük eğitim ve sıra modeli deneyi için karar vermek.

Kaynaklar, erişim ayrıntıları ve gönderilmemiş soru taslağı: [RESEARCH.md](RESEARCH.md).
Tam sayısal tablolar, güven aralıkları ve yeniden çalıştırma komutları: [SUMMARY.md](SUMMARY.md).
112 açık veri dosyasının boyut ve SHA256 bilgisi: [provenance.json](provenance.json).
Tüm büyük girdiler, ortam ve cache `D:\barca` altında tutuldu. Uzak repoya push yapılmadı.
