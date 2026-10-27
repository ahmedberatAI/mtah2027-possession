# PFF 25 fps: kontrollü etiketler ve v7 karşılaştırması

Sabit deney: aynı 172 özellik, 400 LightGBM ağacı, stride 2, B2 / pen_team=12.
Ana model DFL+PFF-controlled; PFF-only ve native etiketler tanı amaçlı. Ölü-top modeli sabit.
PFF native/controlled eğitiminde aynı bilinen canlı kareler kullanılır. Dış testte ayar seçimi yok.

| Model | Veri / konvansiyon | D | v7 spellF1 | Yeni | Δ | Eşli %95 GA | Artan maç |
|---|---|---|---:|---:|---:|---|---|
| dfl_pff_controlled | metrica@v3 / ctrl | learned | 0.8147 | 0.8144 | -0.0003 | [-0.0077, +0.0088] | 1/3 |
| dfl_pff_controlled | metrica@v3 / ctrl | oracle | 0.8715 | 0.8709 | -0.0006 | [-0.0081, +0.0074] | 1/3 |
| dfl_pff_controlled | metrica@v3 / events | learned | 0.6155 | 0.6100 | -0.0056 | [-0.0150, +0.0092] | 1/3 |
| dfl_pff_controlled | metrica@v3 / events | oracle | 0.6595 | 0.6548 | -0.0047 | [-0.0150, +0.0099] | 1/3 |
| dfl_pff_controlled | sc@v3 / pp | learned | 0.5829 | 0.5913 | +0.0084 | [+0.0023, +0.0144] | 14/20 |
| dfl_pff_controlled | sc@v3 / pp | oracle | 0.7410 | 0.7471 | +0.0061 | [+0.0012, +0.0115] | 11/20 |
| dfl_pff_controlled | sc@v3 / phase | learned | 0.5816 | 0.5897 | +0.0081 | [+0.0018, +0.0142] | 14/20 |
| dfl_pff_controlled | sc@v3 / phase | oracle | 0.7457 | 0.7526 | +0.0069 | [+0.0003, +0.0143] | 11/20 |
| pff_controlled | metrica@v3 / ctrl | learned | 0.8147 | 0.8123 | -0.0024 | [-0.0114, +0.0083] | 1/3 |
| pff_controlled | metrica@v3 / ctrl | oracle | 0.8715 | 0.8711 | -0.0004 | [-0.0107, +0.0127] | 1/3 |
| pff_controlled | metrica@v3 / events | learned | 0.6155 | 0.6022 | -0.0133 | [-0.0203, -0.0026] | 0/3 |
| pff_controlled | metrica@v3 / events | oracle | 0.6595 | 0.6486 | -0.0109 | [-0.0186, +0.0002] | 1/3 |
| dfl_pff_native | metrica@v3 / ctrl | learned | 0.8147 | 0.8043 | -0.0104 | [-0.0190, +0.0006] | 1/3 |
| dfl_pff_native | metrica@v3 / ctrl | oracle | 0.8715 | 0.8650 | -0.0065 | [-0.0118, +0.0005] | 1/3 |
| dfl_pff_native | metrica@v3 / events | learned | 0.6155 | 0.6152 | -0.0003 | [-0.0117, +0.0172] | 1/3 |
| dfl_pff_native | metrica@v3 / events | oracle | 0.6595 | 0.6607 | +0.0012 | [-0.0088, +0.0130] | 1/3 |
| pff_controlled | sc@v3 / pp | learned | 0.5829 | 0.5891 | +0.0062 | [+0.0000, +0.0126] | 12/20 |
| pff_controlled | sc@v3 / pp | oracle | 0.7410 | 0.7434 | +0.0024 | [-0.0034, +0.0084] | 12/20 |
| pff_controlled | sc@v3 / phase | learned | 0.5816 | 0.5856 | +0.0040 | [-0.0022, +0.0101] | 11/20 |
| pff_controlled | sc@v3 / phase | oracle | 0.7457 | 0.7463 | +0.0006 | [-0.0071, +0.0089] | 9/20 |
| dfl_pff_native | sc@v3 / pp | learned | 0.5829 | 0.5888 | +0.0059 | [+0.0012, +0.0105] | 14/20 |
| dfl_pff_native | sc@v3 / pp | oracle | 0.7410 | 0.7449 | +0.0039 | [-0.0012, +0.0090] | 12/20 |
| dfl_pff_native | sc@v3 / phase | learned | 0.5816 | 0.5853 | +0.0037 | [-0.0015, +0.0086] | 14/20 |
| dfl_pff_native | sc@v3 / phase | oracle | 0.7457 | 0.7488 | +0.0032 | [-0.0031, +0.0094] | 11/20 |

GA maç düzeyinde eşli bootstrap. Metrica bozulma tohumları önce maç içinde ortalanır;
3 Metrica maçı bağımsız 9 maç gibi sayılmaz. Eğitim veri seçimi belirsizliği bu GA içinde değildir.
Oracle D, gerçek D durumundan üretilen kuvvetli olasılık kanıtıdır; son çıktı D sınırlarını sertçe sabitlemez.
Skorlar dış veri vekil etiketlerinedir; özel BAR-ATH skoru ölçülmedi.

## Veri ve etiket çıktıları

Sayım/hizalama özeti: [data_summary.json](data_summary.json); maç bazında: [data_audit.json](data_audit.json).
Etiket CSV.GZ: `cache/pff25/raw_control_v2/labels/`. Her satır bir 25 fps karesi;
`video_time_s` kaynak video zamanı, `half_frame25` yeniden örneklenen devrenin 0 tabanlı indeksi.
H/A/D/X etiketlerinde X eğitim dışı/belirsizdir. X ölü top sayılmaz.
Ham konumlar kullanıldı; PFF olaylardan oluşturulan yumuşatılmış top konumu özelliklere girmedi.

## Yeniden çalıştırma

```powershell
.venv/Scripts/python.exe src/exp_pff.py prepare --workers 6
.venv/Scripts/python.exe src/report_pff.py data
.venv/Scripts/python.exe src/exp_pff.py train
.venv/Scripts/python.exe src/exp_pff.py evaluate
.venv/Scripts/python.exe src/report_pff.py results
.venv/Scripts/python.exe -m unittest discover -s tests -v
```
