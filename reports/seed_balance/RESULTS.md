# Üç seed ve kaynak dengesi — tüm sonuçlar

Skorlar seed başına maç ortalaması spellF1; ± örnek standart sapma (ddof=1).
Yeni seedler: 20261025, 20261026, 20261027. Önceki tek seed bu ortalamaya girmez.

| Test | D | Model | Ortalama ± SS | Min–maks | Seed skorları |
|---|---|---|---|---|---|
| metrica@v3/ctrl | learned | lgb_unbalanced | 0.8130 ± 0.0011 | 0.8119–0.8140 | 0.8132, 0.8119, 0.8140 |
| metrica@v3/ctrl | learned | lgb_balanced | 0.8173 ± 0.0028 | 0.8148–0.8204 | 0.8204, 0.8169, 0.8148 |
| metrica@v3/ctrl | learned | tcn | 0.8046 ± 0.0076 | 0.7959–0.8091 | 0.7959, 0.8091, 0.8089 |
| metrica@v3/ctrl | learned | bigru | 0.8024 ± 0.0053 | 0.7990–0.8086 | 0.7997, 0.8086, 0.7990 |
| metrica@v3/ctrl | oracle | lgb_unbalanced | 0.8705 ± 0.0012 | 0.8695–0.8717 | 0.8695, 0.8702, 0.8717 |
| metrica@v3/ctrl | oracle | lgb_balanced | 0.8747 ± 0.0034 | 0.8710–0.8774 | 0.8774, 0.8758, 0.8710 |
| metrica@v3/ctrl | oracle | tcn | 0.8611 ± 0.0083 | 0.8515–0.8664 | 0.8515, 0.8654, 0.8664 |
| metrica@v3/ctrl | oracle | bigru | 0.8581 ± 0.0052 | 0.8537–0.8639 | 0.8567, 0.8639, 0.8537 |
| metrica@v3/events | learned | lgb_unbalanced | 0.6107 ± 0.0014 | 0.6091–0.6117 | 0.6117, 0.6114, 0.6091 |
| metrica@v3/events | learned | lgb_balanced | 0.6159 ± 0.0007 | 0.6153–0.6166 | 0.6153, 0.6166, 0.6157 |
| metrica@v3/events | learned | tcn | 0.6414 ± 0.0107 | 0.6333–0.6536 | 0.6536, 0.6373, 0.6333 |
| metrica@v3/events | learned | bigru | 0.6463 ± 0.0104 | 0.6371–0.6575 | 0.6443, 0.6371, 0.6575 |
| metrica@v3/events | oracle | lgb_unbalanced | 0.6558 ± 0.0009 | 0.6548–0.6564 | 0.6562, 0.6564, 0.6548 |
| metrica@v3/events | oracle | lgb_balanced | 0.6606 ± 0.0010 | 0.6599–0.6618 | 0.6603, 0.6618, 0.6599 |
| metrica@v3/events | oracle | tcn | 0.6857 ± 0.0100 | 0.6776–0.6968 | 0.6968, 0.6826, 0.6776 |
| metrica@v3/events | oracle | bigru | 0.6910 ± 0.0110 | 0.6819–0.7032 | 0.6881, 0.6819, 0.7032 |
| sc@v3/pp | learned | lgb_unbalanced | 0.5911 ± 0.0005 | 0.5906–0.5916 | 0.5916, 0.5910, 0.5906 |
| sc@v3/pp | learned | lgb_balanced | 0.5930 ± 0.0004 | 0.5927–0.5934 | 0.5927, 0.5930, 0.5934 |
| sc@v3/pp | learned | tcn | 0.6243 ± 0.0058 | 0.6194–0.6308 | 0.6194, 0.6308, 0.6227 |
| sc@v3/pp | learned | bigru | 0.6308 ± 0.0019 | 0.6286–0.6320 | 0.6286, 0.6320, 0.6318 |
| sc@v3/pp | oracle | lgb_unbalanced | 0.7470 ± 0.0006 | 0.7463–0.7475 | 0.7463, 0.7472, 0.7475 |
| sc@v3/pp | oracle | lgb_balanced | 0.7475 ± 0.0033 | 0.7441–0.7505 | 0.7480, 0.7441, 0.7505 |
| sc@v3/pp | oracle | tcn | 0.7765 ± 0.0044 | 0.7716–0.7801 | 0.7716, 0.7801, 0.7778 |
| sc@v3/pp | oracle | bigru | 0.7841 ± 0.0035 | 0.7812–0.7880 | 0.7812, 0.7832, 0.7880 |
| sc@v3/phase | learned | lgb_unbalanced | 0.5898 ± 0.0004 | 0.5893–0.5901 | 0.5901, 0.5900, 0.5893 |
| sc@v3/phase | learned | lgb_balanced | 0.5920 ± 0.0009 | 0.5909–0.5926 | 0.5909, 0.5924, 0.5926 |
| sc@v3/phase | learned | tcn | 0.6119 ± 0.0082 | 0.6027–0.6183 | 0.6027, 0.6183, 0.6148 |
| sc@v3/phase | learned | bigru | 0.6195 ± 0.0023 | 0.6169–0.6209 | 0.6169, 0.6209, 0.6206 |
| sc@v3/phase | oracle | lgb_unbalanced | 0.7527 ± 0.0006 | 0.7521–0.7533 | 0.7521, 0.7528, 0.7533 |
| sc@v3/phase | oracle | lgb_balanced | 0.7517 ± 0.0026 | 0.7492–0.7545 | 0.7516, 0.7492, 0.7545 |
| sc@v3/phase | oracle | tcn | 0.7681 ± 0.0097 | 0.7570–0.7745 | 0.7570, 0.7729, 0.7745 |
| sc@v3/phase | oracle | bigru | 0.7774 ± 0.0038 | 0.7740–0.7815 | 0.7740, 0.7767, 0.7815 |

## Eşli farklar

| Test | D | Fark | Ortalama Δ | Seed+maç %95 aralık | Artan maç (seed ort.) |
|---|---|---|---:|---|---:|
| metrica@v3/ctrl | learned | lgb_balanced − lgb_unbalanced | +0.0043 | [-0.0003,+0.0096] | 3/3 |
| metrica@v3/ctrl | learned | tcn − lgb_balanced | -0.0127 | [-0.0270,+0.0028] | 0/3 |
| metrica@v3/ctrl | learned | bigru − lgb_balanced | -0.0149 | [-0.0278,-0.0021] | 0/3 |
| metrica@v3/ctrl | learned | tcn − lgb_unbalanced | -0.0084 | [-0.0255,+0.0060] | 2/3 |
| metrica@v3/ctrl | learned | bigru − lgb_unbalanced | -0.0106 | [-0.0272,+0.0038] | 1/3 |
| metrica@v3/ctrl | learned | tcn − v7 | -0.0101 | [-0.0203,+0.0003] | 0/3 |
| metrica@v3/ctrl | learned | bigru − v7 | -0.0123 | [-0.0222,-0.0010] | 0/3 |
| metrica@v3/ctrl | learned | bigru − tcn | -0.0022 | [-0.0094,+0.0041] | 1/3 |
| metrica@v3/ctrl | oracle | lgb_balanced − lgb_unbalanced | +0.0043 | [-0.0018,+0.0109] | 2/3 |
| metrica@v3/ctrl | oracle | tcn − lgb_balanced | -0.0136 | [-0.0261,+0.0011] | 0/3 |
| metrica@v3/ctrl | oracle | bigru − lgb_balanced | -0.0166 | [-0.0303,-0.0053] | 0/3 |
| metrica@v3/ctrl | oracle | tcn − lgb_unbalanced | -0.0094 | [-0.0248,+0.0047] | 1/3 |
| metrica@v3/ctrl | oracle | bigru − lgb_unbalanced | -0.0123 | [-0.0311,+0.0032] | 0/3 |
| metrica@v3/ctrl | oracle | tcn − v7 | -0.0104 | [-0.0223,+0.0015] | 1/3 |
| metrica@v3/ctrl | oracle | bigru − v7 | -0.0134 | [-0.0255,-0.0015] | 0/3 |
| metrica@v3/ctrl | oracle | bigru − tcn | -0.0030 | [-0.0124,+0.0066] | 0/3 |
| metrica@v3/events | learned | lgb_balanced − lgb_unbalanced | +0.0051 | [-0.0008,+0.0100] | 2/3 |
| metrica@v3/events | learned | tcn − lgb_balanced | +0.0255 | [-0.0008,+0.0475] | 2/3 |
| metrica@v3/events | learned | bigru − lgb_balanced | +0.0304 | [+0.0041,+0.0521] | 3/3 |
| metrica@v3/events | learned | tcn − lgb_unbalanced | +0.0307 | [+0.0053,+0.0534] | 3/3 |
| metrica@v3/events | learned | bigru − lgb_unbalanced | +0.0356 | [+0.0102,+0.0583] | 3/3 |
| metrica@v3/events | learned | tcn − v7 | +0.0258 | [-0.0056,+0.0525] | 2/3 |
| metrica@v3/events | learned | bigru − v7 | +0.0308 | [-0.0007,+0.0541] | 2/3 |
| metrica@v3/events | learned | bigru − tcn | +0.0049 | [-0.0091,+0.0219] | 3/3 |
| metrica@v3/events | oracle | lgb_balanced − lgb_unbalanced | +0.0048 | [-0.0012,+0.0105] | 2/3 |
| metrica@v3/events | oracle | tcn − lgb_balanced | +0.0250 | [-0.0035,+0.0468] | 2/3 |
| metrica@v3/events | oracle | bigru − lgb_balanced | +0.0304 | [+0.0008,+0.0543] | 3/3 |
| metrica@v3/events | oracle | tcn − lgb_unbalanced | +0.0298 | [+0.0041,+0.0535] | 3/3 |
| metrica@v3/events | oracle | bigru − lgb_unbalanced | +0.0352 | [+0.0080,+0.0589] | 3/3 |
| metrica@v3/events | oracle | tcn − v7 | +0.0261 | [-0.0047,+0.0515] | 2/3 |
| metrica@v3/events | oracle | bigru − v7 | +0.0315 | [-0.0003,+0.0554] | 2/3 |
| metrica@v3/events | oracle | bigru − tcn | +0.0054 | [-0.0086,+0.0243] | 3/3 |
| sc@v3/pp | learned | lgb_balanced − lgb_unbalanced | +0.0020 | [-0.0029,+0.0070] | 13/20 |
| sc@v3/pp | learned | tcn − lgb_balanced | +0.0313 | [+0.0221,+0.0401] | 20/20 |
| sc@v3/pp | learned | bigru − lgb_balanced | +0.0378 | [+0.0311,+0.0443] | 20/20 |
| sc@v3/pp | learned | tcn − lgb_unbalanced | +0.0332 | [+0.0237,+0.0422] | 20/20 |
| sc@v3/pp | learned | bigru − lgb_unbalanced | +0.0397 | [+0.0324,+0.0467] | 20/20 |
| sc@v3/pp | learned | tcn − v7 | +0.0414 | [+0.0311,+0.0513] | 20/20 |
| sc@v3/pp | learned | bigru − v7 | +0.0479 | [+0.0406,+0.0551] | 20/20 |
| sc@v3/pp | learned | bigru − tcn | +0.0065 | [+0.0008,+0.0117] | 17/20 |
| sc@v3/pp | oracle | lgb_balanced − lgb_unbalanced | +0.0005 | [-0.0050,+0.0061] | 10/20 |
| sc@v3/pp | oracle | tcn − lgb_balanced | +0.0289 | [+0.0196,+0.0378] | 20/20 |
| sc@v3/pp | oracle | bigru − lgb_balanced | +0.0366 | [+0.0287,+0.0448] | 20/20 |
| sc@v3/pp | oracle | tcn − lgb_unbalanced | +0.0295 | [+0.0215,+0.0368] | 20/20 |
| sc@v3/pp | oracle | bigru − lgb_unbalanced | +0.0371 | [+0.0299,+0.0451] | 20/20 |
| sc@v3/pp | oracle | tcn − v7 | +0.0355 | [+0.0265,+0.0438] | 20/20 |
| sc@v3/pp | oracle | bigru − v7 | +0.0431 | [+0.0353,+0.0519] | 20/20 |
| sc@v3/pp | oracle | bigru − tcn | +0.0076 | [+0.0017,+0.0134] | 17/20 |
| sc@v3/phase | learned | lgb_balanced − lgb_unbalanced | +0.0022 | [-0.0026,+0.0071] | 12/20 |
| sc@v3/phase | learned | tcn − lgb_balanced | +0.0199 | [+0.0089,+0.0297] | 17/20 |
| sc@v3/phase | learned | bigru − lgb_balanced | +0.0275 | [+0.0197,+0.0352] | 19/20 |
| sc@v3/phase | learned | tcn − lgb_unbalanced | +0.0221 | [+0.0105,+0.0328] | 18/20 |
| sc@v3/phase | learned | bigru − lgb_unbalanced | +0.0297 | [+0.0212,+0.0380] | 19/20 |
| sc@v3/phase | learned | tcn − v7 | +0.0303 | [+0.0181,+0.0412] | 19/20 |
| sc@v3/phase | learned | bigru − v7 | +0.0379 | [+0.0299,+0.0460] | 20/20 |
| sc@v3/phase | learned | bigru − tcn | +0.0075 | [+0.0013,+0.0143] | 17/20 |
| sc@v3/phase | oracle | lgb_balanced − lgb_unbalanced | -0.0010 | [-0.0058,+0.0038] | 8/20 |
| sc@v3/phase | oracle | tcn − lgb_balanced | +0.0164 | [+0.0042,+0.0271] | 18/20 |
| sc@v3/phase | oracle | bigru − lgb_balanced | +0.0256 | [+0.0170,+0.0347] | 19/20 |
| sc@v3/phase | oracle | tcn − lgb_unbalanced | +0.0154 | [+0.0037,+0.0262] | 18/20 |
| sc@v3/phase | oracle | bigru − lgb_unbalanced | +0.0247 | [+0.0157,+0.0343] | 18/20 |
| sc@v3/phase | oracle | tcn − v7 | +0.0225 | [+0.0096,+0.0343] | 16/20 |
| sc@v3/phase | oracle | bigru − v7 | +0.0317 | [+0.0225,+0.0417] | 20/20 |
| sc@v3/phase | oracle | bigru − tcn | +0.0092 | [+0.0022,+0.0171] | 18/20 |

Aralıklar seed ve maç eksenlerini birlikte örnekleyen 10.000 bootstrap tekrarından.
Yalnız üç seed vardır; aralıklar keşifseldir. Metrica 3, SkillCorner 20 bağımsız maçtır.
Metrica bozulma tohumları önce maç içinde ortalanır. Seed skorlarının ortalaması ensemble değildir.
Dengeli/normal LGB çifti yalnız kaynak ağırlığını değiştirir. Sinir ağı karşılaştırması
normalizasyon, pencere örnekleme, örnekleme hızı ve mimari farklarını birlikte içerir.
