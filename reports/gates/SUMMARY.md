# Üç bahis: CPU kapı testlerinin sonuçları

24 Ekim 2026. Skorlar resmi **spellF1**; dış maçların test amaçlı etiketleriyle ölçülmüştür.
BAR-ATH özel GT sonucu değildir. Eğitimde 7 DFL maçı; testte 3 Metrica ve 20 SkillCorner maçı.

## 1. Eğitim verisini 2 → 4 → 7 maça büyütmek

Aynı v7 özellikleri, sim V3, 400 ağaç, native etiketler ve B2. Ölü-top modeli 7 maçla sabit.
2 ve 4 maç için yedi dengeli/iç içe alt küme; 7 maç için tek deterministik model.

| Veri / etiket | D | 2 maç | 4 maç | 7 maç | 7−4 | Eşli %95 GA |
|---|---|---:|---:|---:|---:|---|
| metrica@v3 / ctrl | learned | 0.8022 | 0.8095 | 0.8147 | +0.0052 | [+0.0016, +0.0073] |
| metrica@v3 / ctrl | oracle | 0.8609 | 0.8684 | 0.8715 | +0.0031 | [-0.0013, +0.0055] |
| metrica@v3 / events | learned | 0.6126 | 0.6150 | 0.6155 | +0.0005 | [-0.0027, +0.0064] |
| metrica@v3 / events | oracle | 0.6586 | 0.6614 | 0.6595 | -0.0019 | [-0.0039, +0.0020] |
| sc@v3 / pp | learned | 0.5792 | 0.5813 | 0.5829 | +0.0016 | [-0.0012, +0.0043] |
| sc@v3 / pp | oracle | 0.7330 | 0.7366 | 0.7410 | +0.0044 | [+0.0019, +0.0068] |
| sc@v3 / phase | learned | 0.5768 | 0.5811 | 0.5816 | +0.0005 | [-0.0022, +0.0032] |
| sc@v3 / phase | oracle | 0.7362 | 0.7423 | 0.7457 | +0.0033 | [-0.0002, +0.0069] |

GA: test maçları üzerinde eşli bootstrap; Metrica üç bozulma tohumu önce maç içinde ortalanır.
Yedi alt küme bağımsız yedi veri kümesi değildir. GA, eğitim-maçı evreninin belirsizliğini kapsamaz.
Üç Metrica maçıyla GA özellikle zayıftır. DFL içinde büyüme, Gradient alanına taşınmayı garanti etmez.

## 2. Paslardan tüm mevcut temas vekillerine

Ana test: event precision=.70, recall=.70, team accuracy=.95, jitter std=120 ms.
team70: event precision=1, recall=.70, team accuracy=.70. β=4 önceden sabit; β=2 duyarlılık.
Δ aynı 7 maçlık v7 modeline göre. Kontakların GT ile ortak kaynağı nedeniyle iyimser simülasyondur.

| Veri / etiket | D | Çapa | Senaryo | β | ΔspellF1 | %95 GA | Artan maç |
|---|---|---|---|---:|---:|---|---|
| metrica@v3 / ctrl | learned | contacts | perfect | 4 | -0.0115 | [-0.0157, -0.0084] | 0/3 |
| metrica@v3 / ctrl | learned | contacts | precision70 | 2 | -0.0000 | [-0.0026, +0.0047] | 1/3 |
| metrica@v3 / ctrl | learned | contacts | precision70 | 4 | -0.0254 | [-0.0267, -0.0244] | 0/3 |
| metrica@v3 / ctrl | learned | contacts | team70 | 4 | -0.0307 | [-0.0353, -0.0279] | 0/3 |
| metrica@v3 / ctrl | learned | contacts_typed | perfect | 4 | -0.0110 | [-0.0185, -0.0061] | 0/3 |
| metrica@v3 / ctrl | learned | contacts_typed | precision70 | 2 | +0.0021 | [-0.0028, +0.0098] | 1/3 |
| metrica@v3 / ctrl | learned | contacts_typed | precision70 | 4 | -0.0204 | [-0.0219, -0.0185] | 0/3 |
| metrica@v3 / ctrl | learned | contacts_typed | team70 | 4 | -0.0259 | [-0.0281, -0.0236] | 0/3 |
| metrica@v3 / ctrl | learned | passes | perfect | 4 | +0.0162 | [+0.0092, +0.0228] | 3/3 |
| metrica@v3 / ctrl | learned | passes | precision70 | 2 | +0.0023 | [+0.0008, +0.0036] | 3/3 |
| metrica@v3 / ctrl | learned | passes | precision70 | 4 | +0.0017 | [-0.0047, +0.0071] | 2/3 |
| metrica@v3 / ctrl | learned | passes | team70 | 4 | +0.0018 | [-0.0036, +0.0065] | 2/3 |
| metrica@v3 / ctrl | oracle | contacts | perfect | 4 | -0.0068 | [-0.0115, -0.0043] | 0/3 |
| metrica@v3 / ctrl | oracle | contacts | precision70 | 2 | +0.0045 | [-0.0012, +0.0094] | 2/3 |
| metrica@v3 / ctrl | oracle | contacts | precision70 | 4 | -0.0218 | [-0.0255, -0.0165] | 0/3 |
| metrica@v3 / ctrl | oracle | contacts | team70 | 4 | -0.0297 | [-0.0369, -0.0237] | 0/3 |
| metrica@v3 / ctrl | oracle | contacts_typed | perfect | 4 | -0.0043 | [-0.0130, +0.0044] | 1/3 |
| metrica@v3 / ctrl | oracle | contacts_typed | precision70 | 2 | +0.0061 | [+0.0026, +0.0120] | 3/3 |
| metrica@v3 / ctrl | oracle | contacts_typed | precision70 | 4 | -0.0170 | [-0.0197, -0.0133] | 0/3 |
| metrica@v3 / ctrl | oracle | contacts_typed | team70 | 4 | -0.0273 | [-0.0296, -0.0237] | 0/3 |
| metrica@v3 / ctrl | oracle | passes | perfect | 4 | +0.0204 | [+0.0133, +0.0248] | 3/3 |
| metrica@v3 / ctrl | oracle | passes | precision70 | 2 | +0.0040 | [+0.0029, +0.0050] | 3/3 |
| metrica@v3 / ctrl | oracle | passes | precision70 | 4 | +0.0038 | [-0.0038, +0.0116] | 2/3 |
| metrica@v3 / ctrl | oracle | passes | team70 | 4 | +0.0025 | [-0.0022, +0.0082] | 2/3 |
| metrica@v3 / events | learned | contacts | perfect | 4 | +0.0500 | [+0.0289, +0.0740] | 3/3 |
| metrica@v3 / events | learned | contacts | precision70 | 2 | +0.0157 | [+0.0038, +0.0314] | 3/3 |
| metrica@v3 / events | learned | contacts | precision70 | 4 | +0.0327 | [+0.0135, +0.0552] | 3/3 |
| metrica@v3 / events | learned | contacts | team70 | 4 | +0.0210 | [+0.0032, +0.0394] | 3/3 |
| metrica@v3 / events | learned | contacts_typed | perfect | 4 | +0.0766 | [+0.0540, +0.1044] | 3/3 |
| metrica@v3 / events | learned | contacts_typed | precision70 | 2 | +0.0235 | [+0.0111, +0.0375] | 3/3 |
| metrica@v3 / events | learned | contacts_typed | precision70 | 4 | +0.0469 | [+0.0272, +0.0695] | 3/3 |
| metrica@v3 / events | learned | contacts_typed | team70 | 4 | +0.0273 | [+0.0107, +0.0471] | 3/3 |
| metrica@v3 / events | learned | passes | perfect | 4 | +0.0078 | [-0.0022, +0.0129] | 2/3 |
| metrica@v3 / events | learned | passes | precision70 | 2 | +0.0044 | [+0.0006, +0.0073] | 3/3 |
| metrica@v3 / events | learned | passes | precision70 | 4 | +0.0075 | [+0.0008, +0.0152] | 3/3 |
| metrica@v3 / events | learned | passes | team70 | 4 | +0.0043 | [-0.0019, +0.0091] | 2/3 |
| metrica@v3 / events | oracle | contacts | perfect | 4 | +0.0603 | [+0.0435, +0.0875] | 3/3 |
| metrica@v3 / events | oracle | contacts | precision70 | 2 | +0.0186 | [+0.0103, +0.0336] | 3/3 |
| metrica@v3 / events | oracle | contacts | precision70 | 4 | +0.0403 | [+0.0256, +0.0626] | 3/3 |
| metrica@v3 / events | oracle | contacts | team70 | 4 | +0.0249 | [+0.0121, +0.0439] | 3/3 |
| metrica@v3 / events | oracle | contacts_typed | perfect | 4 | +0.0864 | [+0.0686, +0.1160] | 3/3 |
| metrica@v3 / events | oracle | contacts_typed | precision70 | 2 | +0.0279 | [+0.0194, +0.0419] | 3/3 |
| metrica@v3 / events | oracle | contacts_typed | precision70 | 4 | +0.0546 | [+0.0398, +0.0780] | 3/3 |
| metrica@v3 / events | oracle | contacts_typed | team70 | 4 | +0.0314 | [+0.0207, +0.0512] | 3/3 |
| metrica@v3 / events | oracle | passes | perfect | 4 | +0.0116 | [+0.0026, +0.0188] | 3/3 |
| metrica@v3 / events | oracle | passes | precision70 | 2 | +0.0062 | [+0.0044, +0.0095] | 3/3 |
| metrica@v3 / events | oracle | passes | precision70 | 4 | +0.0107 | [+0.0060, +0.0188] | 3/3 |
| metrica@v3 / events | oracle | passes | team70 | 4 | +0.0049 | [-0.0002, +0.0099] | 2/3 |
| sc@v3 / phase | learned | contacts | perfect | 4 | +0.0444 | [+0.0382, +0.0507] | 20/20 |
| sc@v3 / phase | learned | contacts | precision70 | 2 | +0.0145 | [+0.0114, +0.0178] | 20/20 |
| sc@v3 / phase | learned | contacts | precision70 | 4 | +0.0214 | [+0.0169, +0.0259] | 20/20 |
| sc@v3 / phase | learned | contacts | team70 | 4 | +0.0045 | [+0.0005, +0.0088] | 12/20 |
| sc@v3 / phase | learned | contacts_typed | perfect | 4 | +0.0586 | [+0.0521, +0.0654] | 20/20 |
| sc@v3 / phase | learned | contacts_typed | precision70 | 2 | +0.0204 | [+0.0168, +0.0242] | 20/20 |
| sc@v3 / phase | learned | contacts_typed | precision70 | 4 | +0.0307 | [+0.0255, +0.0363] | 20/20 |
| sc@v3 / phase | learned | contacts_typed | team70 | 4 | +0.0091 | [+0.0044, +0.0141] | 15/20 |
| sc@v3 / phase | learned | passes | perfect | 4 | +0.0275 | [+0.0219, +0.0333] | 20/20 |
| sc@v3 / phase | learned | passes | precision70 | 2 | +0.0093 | [+0.0061, +0.0125] | 18/20 |
| sc@v3 / phase | learned | passes | precision70 | 4 | +0.0127 | [+0.0077, +0.0178] | 17/20 |
| sc@v3 / phase | learned | passes | team70 | 4 | +0.0026 | [-0.0022, +0.0074] | 13/20 |
| sc@v3 / phase | oracle | contacts | perfect | 4 | +0.0528 | [+0.0464, +0.0593] | 20/20 |
| sc@v3 / phase | oracle | contacts | precision70 | 2 | +0.0140 | [+0.0110, +0.0174] | 20/20 |
| sc@v3 / phase | oracle | contacts | precision70 | 4 | +0.0185 | [+0.0141, +0.0231] | 19/20 |
| sc@v3 / phase | oracle | contacts | team70 | 4 | +0.0024 | [-0.0025, +0.0077] | 10/20 |
| sc@v3 / phase | oracle | contacts_typed | perfect | 4 | +0.0637 | [+0.0569, +0.0705] | 20/20 |
| sc@v3 / phase | oracle | contacts_typed | precision70 | 2 | +0.0187 | [+0.0149, +0.0228] | 20/20 |
| sc@v3 / phase | oracle | contacts_typed | precision70 | 4 | +0.0267 | [+0.0207, +0.0329] | 20/20 |
| sc@v3 / phase | oracle | contacts_typed | team70 | 4 | +0.0069 | [+0.0010, +0.0128] | 13/20 |
| sc@v3 / phase | oracle | passes | perfect | 4 | +0.0313 | [+0.0256, +0.0374] | 20/20 |
| sc@v3 / phase | oracle | passes | precision70 | 2 | +0.0085 | [+0.0050, +0.0121] | 16/20 |
| sc@v3 / phase | oracle | passes | precision70 | 4 | +0.0103 | [+0.0045, +0.0161] | 15/20 |
| sc@v3 / phase | oracle | passes | team70 | 4 | +0.0022 | [-0.0029, +0.0073] | 12/20 |
| sc@v3 / pp | learned | contacts | perfect | 4 | +0.0507 | [+0.0440, +0.0571] | 20/20 |
| sc@v3 / pp | learned | contacts | precision70 | 2 | +0.0161 | [+0.0125, +0.0198] | 20/20 |
| sc@v3 / pp | learned | contacts | precision70 | 4 | +0.0269 | [+0.0227, +0.0308] | 20/20 |
| sc@v3 / pp | learned | contacts | team70 | 4 | +0.0140 | [+0.0107, +0.0174] | 20/20 |
| sc@v3 / pp | learned | contacts_typed | perfect | 4 | +0.0747 | [+0.0678, +0.0814] | 20/20 |
| sc@v3 / pp | learned | contacts_typed | precision70 | 2 | +0.0254 | [+0.0215, +0.0291] | 20/20 |
| sc@v3 / pp | learned | contacts_typed | precision70 | 4 | +0.0430 | [+0.0384, +0.0474] | 20/20 |
| sc@v3 / pp | learned | contacts_typed | team70 | 4 | +0.0211 | [+0.0174, +0.0248] | 20/20 |
| sc@v3 / pp | learned | passes | perfect | 4 | +0.0339 | [+0.0283, +0.0396] | 20/20 |
| sc@v3 / pp | learned | passes | precision70 | 2 | +0.0100 | [+0.0066, +0.0134] | 17/20 |
| sc@v3 / pp | learned | passes | precision70 | 4 | +0.0170 | [+0.0124, +0.0218] | 19/20 |
| sc@v3 / pp | learned | passes | team70 | 4 | +0.0071 | [+0.0030, +0.0114] | 17/20 |
| sc@v3 / pp | oracle | contacts | perfect | 4 | +0.0615 | [+0.0557, +0.0676] | 20/20 |
| sc@v3 / pp | oracle | contacts | precision70 | 2 | +0.0177 | [+0.0139, +0.0217] | 20/20 |
| sc@v3 / pp | oracle | contacts | precision70 | 4 | +0.0276 | [+0.0237, +0.0315] | 20/20 |
| sc@v3 / pp | oracle | contacts | team70 | 4 | +0.0139 | [+0.0093, +0.0188] | 18/20 |
| sc@v3 / pp | oracle | contacts_typed | perfect | 4 | +0.0833 | [+0.0773, +0.0896] | 20/20 |
| sc@v3 / pp | oracle | contacts_typed | precision70 | 2 | +0.0253 | [+0.0207, +0.0299] | 20/20 |
| sc@v3 / pp | oracle | contacts_typed | precision70 | 4 | +0.0423 | [+0.0373, +0.0471] | 20/20 |
| sc@v3 / pp | oracle | contacts_typed | team70 | 4 | +0.0203 | [+0.0155, +0.0251] | 20/20 |
| sc@v3 / pp | oracle | passes | perfect | 4 | +0.0374 | [+0.0322, +0.0432] | 20/20 |
| sc@v3 / pp | oracle | passes | precision70 | 2 | +0.0099 | [+0.0064, +0.0134] | 17/20 |
| sc@v3 / pp | oracle | passes | precision70 | 4 | +0.0156 | [+0.0106, +0.0209] | 17/20 |
| sc@v3 / pp | oracle | passes | team70 | 4 | +0.0072 | [+0.0034, +0.0113] | 17/20 |

Bu test her fiziksel dokunuşu kapsamaz. Kontrolsüz dokunuşu possession diye etiketlemez;
temas takımını yalnız yerel olasılık kanıtı olarak ekler. Füzyon penceresi iki akışta da [−12,+3] kare.
`contacts_typed` sonradan eklenen keşifsel duyarlılıktır: alım/top kazanımında [−3,+12],
mücadelede [−3,+3], pas/şutta [−12,+3]. Olay türünü doğru bildiğini varsayar; ana test değildir.
Gürültü tohumları önce maç içinde ortalanır. Eşik +0.03 test edilen yaklaşım için uygulanır.

## 3. B2 ve etiket konvansiyonu

| Veri / etiket | D | B2 kapalı | B2 açık | Fark |
|---|---|---:|---:|---:|
| metrica@v3 / ctrl | learned | 0.7723 | 0.8147 | +0.0424 |
| metrica@v3 / events | learned | 0.6500 | 0.6155 | -0.0345 |
| metrica@v3 / ctrl | oracle | 0.8347 | 0.8715 | +0.0368 |
| metrica@v3 / events | oracle | 0.7006 | 0.6595 | -0.0411 |
| sc@v3 / pp | learned | 0.5732 | 0.5829 | +0.0097 |
| sc@v3 / phase | learned | 0.5566 | 0.5816 | +0.0250 |
| sc@v3 / pp | oracle | 0.7289 | 0.7410 | +0.0121 |
| sc@v3 / phase | oracle | 0.7117 | 0.7457 | +0.0340 |

B2 karşılaştırması sabit pen_team=12 ile yapılır; iki dekoder ayrı ayrı ayarlanmadı.
Bu ölçüm özel GT üretim yöntemini teşhis etmez. Güncel resmi doküman owning alanının upstream
üretimini açıklamıyor. Kaynak incelemesi ve gönderilmemiş soru taslağı: [RESEARCH.md](RESEARCH.md).

## Yeniden üretim

PowerShell, repo kökünde:

```powershell
$env:BENCH_PROCS='3'; $env:BENCH_THREADS='8'
.venv/Scripts/python.exe src/download_idsse.py
.venv/Scripts/python.exe src/metrica_sample.py download
.venv/Scripts/python.exe src/fetch_gate_data.py
.venv/Scripts/python.exe src/exp_data_gate.py prepare
.venv/Scripts/python.exe src/exp_data_gate.py train
.venv/Scripts/python.exe src/exp_data_gate.py evaluate
.venv/Scripts/python.exe src/exp_contact_gate.py
.venv/Scripts/python.exe src/exp_label_gate.py
.venv/Scripts/python.exe src/summarize_gates.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Sürümler `requirements-gates.lock.txt`; deney tasarımı `design.json`. Frame düzeyindeki
çıktılar/model dosyaları gitignore kapsamındaki `cache/gates/` altında. Yalnız toplu skorlar raporda.
