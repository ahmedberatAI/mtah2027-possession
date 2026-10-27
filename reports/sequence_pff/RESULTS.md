# TCN / BiGRU — dış değerlendirme

Ayar seçimi yalnız ayrılmış DFL/PFF maçlarında; son eğitim tüm 7+64 maçta.
DFL/PFF 50:50; aynı v7 özellik ailesi ve sabit B2. Learned-D modeli sabit.

| Model | Test | D | spellF1 | Δ v7 | %95 GA | Δ PFF LightGBM | %95 GA |
|---|---|---|---:|---:|---|---:|---|
| tcn | metrica@v3/ctrl | learned | 0.8168 | +0.0021 | [-0.0012,+0.0041] | +0.0024 | [-0.0101,+0.0118] |
| tcn | metrica@v3/ctrl | oracle | 0.8717 | +0.0002 | [-0.0054,+0.0038] | +0.0008 | [-0.0128,+0.0104] |
| tcn | metrica@v3/events | learned | 0.6253 | +0.0098 | [-0.0178,+0.0305] | +0.0153 | [-0.0069,+0.0316] |
| tcn | metrica@v3/events | oracle | 0.6711 | +0.0115 | [-0.0134,+0.0305] | +0.0163 | [-0.0043,+0.0325] |
| bigru | metrica@v3/ctrl | learned | 0.8120 | -0.0027 | [-0.0081,+0.0063] | -0.0024 | [-0.0151,+0.0083] |
| bigru | metrica@v3/ctrl | oracle | 0.8686 | -0.0029 | [-0.0093,+0.0068] | -0.0023 | [-0.0135,+0.0078] |
| bigru | metrica@v3/events | learned | 0.6435 | +0.0280 | [-0.0074,+0.0552] | +0.0336 | [+0.0034,+0.0512] |
| bigru | metrica@v3/events | oracle | 0.6915 | +0.0320 | [-0.0023,+0.0599] | +0.0367 | [+0.0068,+0.0533] |
| tcn | sc@v3/pp | learned | 0.6265 | +0.0436 | [+0.0355,+0.0520] | +0.0352 | [+0.0278,+0.0430] |
| tcn | sc@v3/pp | oracle | 0.7804 | +0.0394 | [+0.0322,+0.0475] | +0.0333 | [+0.0256,+0.0418] |
| tcn | sc@v3/phase | learned | 0.6208 | +0.0392 | [+0.0314,+0.0476] | +0.0312 | [+0.0234,+0.0394] |
| tcn | sc@v3/phase | oracle | 0.7802 | +0.0345 | [+0.0262,+0.0444] | +0.0276 | [+0.0187,+0.0372] |
| bigru | sc@v3/pp | learned | 0.6284 | +0.0455 | [+0.0370,+0.0541] | +0.0370 | [+0.0300,+0.0444] |
| bigru | sc@v3/pp | oracle | 0.7866 | +0.0456 | [+0.0376,+0.0540] | +0.0395 | [+0.0319,+0.0474] |
| bigru | sc@v3/phase | learned | 0.6201 | +0.0385 | [+0.0299,+0.0475] | +0.0304 | [+0.0223,+0.0387] |
| bigru | sc@v3/phase | oracle | 0.7824 | +0.0367 | [+0.0282,+0.0461] | +0.0298 | [+0.0210,+0.0387] |

Maç düzeyinde eşli bootstrap; Metrica tohumları maç içinde ortalanır (3 maç, 9 değil).
Aralıklar tek eğitim seedinin maç belirsizliğidir; eğitim belirsizliğini kapsamaz.
Özel BAR-ATH skoru ölçülmedi. PFF kontrollü etiketleri muhafazakâr vekillerdir.
