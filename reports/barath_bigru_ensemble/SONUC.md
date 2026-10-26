# Eşit ağırlıklı BiGRU ensemble ve submission — 25 Ekim 2026

**Ensemble açık bölüm spellF1: 0.7724.** V7'ye göre **+0.1206**;
üç ayrı seed skorunun ortalamasına göre **−0.0046**. Eşit ağırlıklı birleşim
bu açık örnekte ayrı seed skor ortalamasını yükseltmedi.

| Karşılaştırma | spellF1 |
|---|---:|
| v7 | 0.6519 |
| Ayrı BiGRU seed'leri | 0.7724 / 0.7724 / 0.7862 |
| Ayrı seed skorlarının ortalaması | 0.7770 |
| **Eşit ağırlıklı olasılık ensemble** | **0.7724** |

20261025, 20261026 ve 20261027 seed'lerinin her karedeki ev sahibi possession
olasılıkları **1/3 + 1/3 + 1/3** ağırlıkla aritmetik olarak ortalandı. Sonuç,
aynı v7 kural tabanlı ölü-top kanalı ve B2 dekoderinden geçirildi. Etiketlere
çoğunluk oyu uygulanmadı; farklı ağırlıklar veya dekoder ayarları denenmedi.

Ensemble macroF1 **0.9433**. Açık bölümde 17 GT possession'a karşı 12 tahmin
possession var. Resmî 1.000-tekrarlı bootstrap spellF1 aralığı yaklaşık
**[0.5510,0.9018]**; yalnız 7.500 kare / beş dakikalık açık bölüm ölçülüyor.
Tam maçın gizli skoru bilinmiyor. Önceki public seed sonuçları zaten görülmüştü;
bu deney kullanıcının istediği sabit eşit ağırlıklı birleşimdir.

## Hazır teslim dosyası

`submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv`

- Başlık: `frame,label`; **199.319 veri satırı**, frame **0–199318**.
- Tam maç için H/A/D/X çıktısı; değerlendirme aralıklarının dışında X.
- Resmî `validate_submission.py`: **OK**.
- Resmî `run_evaluation.py` ile sarmalayıcı skorları uyuştu.
- SHA256: `f4a4b86501fab3e94e5ef9e288b064a431818c35908982678af39c0ee1cfd225`.
- Dosya yerelde, gitignored klasörde hazırdır; yarışma platformuna yüklenmedi.

Kaynak olasılıklar önce eski üç seed CSV'sini aynen yeniden üreterek denetlendi.
Birleşim ve CSV, yeni skorlamadan önce hash ile donduruldu. Eşit ortalama,
geçersiz olasılık ve devre/kare uyumsuzluğu için üç test geçti. Olasılıkların
tam aritmetik ortalaması, girdi hash'leri, kare hizalaması ve resmî skor uyumu
ayrıca doğrulandı.

[Ayrıntılar ve komutlar](RESULTS.md) · [Doğrulama](verification.json) ·
[Submission kaydı](prediction_manifest.json)
