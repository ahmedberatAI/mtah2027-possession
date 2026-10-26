"""Render aggregate diagnostics without exposing frames or coordinates."""
from pathlib import Path
import exp_hybrid as exp


def main():
    p=exp.REPORT
    rows=exp.read(p/'errors.json' if (p/'errors.json').exists() else p/'diagnostic_errors.json')
    comp=exp.read(p/'complementarity.json' if (p/'complementarity.json').exists() else p/'diagnostic_complementarity.json')
    lines=['# Possession hata analizi','',
        'Bu tablolar IoU=0.5 eşleşmesiyle tanısaldır. Ana spellF1 beş IoU eşiğinin ortalamasıdır.',
        'Metrica gürültü seedleri önce maç içinde ortalanır; ardından sayımlar maçlar üzerinde toplanır.',
        'Oranlar bu toplamların oranıdır; maçlara eşit ağırlıklı ana skorla aynı toplama işlemi değildir.',
        'Eşleşmeyen tahmin, bütünüyle uydurma possession demek değildir: yetersiz zaman örtüşmesi de olabilir.','',
        '| Test | Model | <4sn kaçırma | Tüm possession kaçırma | Eşleşmeyen tahmin oranı | Birleştirme göstergesi | Bölme göstergesi |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        c=r['counts'];q=r['rates'];short=c['0-2s_gt']+c['2-4s_gt']
        miss=(c['0-2s_missed']+c['2-4s_missed'])/short if short else 0
        values=(miss,q['missed_rate'],q['false_discovery_rate'],q['merge_proxy_rate'],q['split_proxy_rate'])
        lines.append(f'| {r["source"]}/{r["conv"]} | {r["model"]} | '+' | '.join(f'{100*x:.2f}%' for x in values)+' |')
    lines+=['','Birleştirme göstergesi: bir tahminin aynı takıma ait en az iki GT possession ile pozitif zaman örtüşmesi.',
        'Bölme göstergesi: bir GT possession ile aynı takıma ait en az iki tahminin pozitif zaman örtüşmesi.',
        'Bunlar tanısal göstergelerdir; birbirini dışlayan, kesin neden etiketleri değildir.','',
        '| Test | Model | Geçiş recall ±1sn | Geçiş precision ±1sn | Ortalama gecikme | Ortalama mutlak hata |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        q=r['rates']
        lines.append(f'| {r["source"]}/{r["conv"]} | {r["model"]} | {100*q["transition_recall"]:.2f}% | '
            f'{100*q["transition_precision"]:.2f}% | {q["mean_latency_seconds"]:+.3f} sn | {q["mean_abs_latency_seconds"]:.3f} sn |')
    lines+=['','Geçişler ölü-top geçişlerini de içerir; hedef etiketi aynı olan geçişler ±25 karede birebir eşleştirilir.',
        'Zamanlama yalnız eşleşen geçişler içindir. Kaçan geçişleri içermez; genel zamanlama başarısı diye okunmamalı.',
        'Pozitif gecikme tahminin geç geldiğini belirtir. Bilinmeyen GT aralıkları arasında eşleşmeye izin verilmez.','',
        '| Test | İkisi doğru | Yalnız LGB doğru | Yalnız BiGRU doğru | İkisi de kaçırdı |',
        '|---|---:|---:|---:|---:|']
    for r in comp:
        lines.append(f'| {r["source"]}/{r["conv"]} | '+' | '.join(f'{r[k]:.2f}' for k in
            ('both_match','only_lgb_matches','only_bigru_matches','neither_matches'))+' |')
    lines+=['','Doğru burada GT possession’ın IoU=0.5 eşiğinde eşleşmesi anlamına gelir.',
        'Metrica sayıları üç gürültü gerçekleşimi ortalandığından kesirli olabilir.',
        'Tamamlayıcılık tablosu ideal seçimle kurtarılabilecek GT örneklerini gösterir; olasılık harmanının bunları otomatik kurtaracağını göstermez.',
        'Kare bazlı veya possession başlangıç/bitiş verileri bu rapora yazılmamıştır.']
    (p/'HATA_ANALIZI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
