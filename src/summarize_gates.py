"""Aggregate gate results at MATCH level, retaining subset and noise caveats."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import exp_data_gate as gate


def ci(values):
    a = np.array(values)
    rng = np.random.default_rng(24)
    boot = a[rng.integers(0, len(a), size=(5000, len(a)))].mean(1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def read(name):
    p = gate.REPORT / name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else []


def avg_matches(rows, key):
    matches = sorted(rows[0][key])
    return {m: float(np.mean([r[key][m] for r in rows])) for m in matches}


def main():
    learning, contacts, sensitivity = read('learning_curve.json'), read('contacts.json'), read('label_sensitivity.json')
    if (len(learning), len(contacts), len(sensitivity)) != (120, 240, 8):
        raise RuntimeError(f'Incomplete run: learning={len(learning)}/120, contacts={len(contacts)}/240, labels={len(sensitivity)}/8')
    for rows, fields in ((learning, ('source','n','rep','conv','dead')),
                         (contacts, ('source','anchors','scenario','seed','beta','conv','dead'))):
        assert len({tuple(r[k] for k in fields) for r in rows}) == len(rows), 'Duplicate experiment cell'
    summary = dict(learning=[], contacts=[], label_sensitivity=sensitivity)
    lines = ['# Üç bahis: CPU kapı testlerinin sonuçları', '',
             '24 Ekim 2026. Skorlar resmi **spellF1**; dış maçların test amaçlı etiketleriyle ölçülmüştür.',
             'BAR-ATH özel GT sonucu değildir. Eğitimde 7 DFL maçı; testte 3 Metrica ve 20 SkillCorner maçı.', '',
             '## 1. Eğitim verisini 2 → 4 → 7 maça büyütmek', '',
             'Aynı v7 özellikleri, sim V3, 400 ağaç, native etiketler ve B2. Ölü-top modeli 7 maçla sabit.',
             '2 ve 4 maç için yedi dengeli/iç içe alt küme; 7 maç için tek deterministik model.', '',
             '| Veri / etiket | D | 2 maç | 4 maç | 7 maç | 7−4 | Eşli %95 GA |',
             '|---|---|---:|---:|---:|---:|---|']
    for src in gate.SOURCES:
        for conv in gate.CONVS[src]:
            for dead in ('learned', 'oracle'):
                rs = [r for r in learning if r['source'] == src and r['conv'] == conv and r['dead'] == dead]
                groups = {n: [r for r in rs if r['n'] == n] for n in (2, 4, 7)}
                if [len(groups[n]) for n in (2, 4, 7)] != [7, 7, 1]:
                    continue
                vals = {n: avg_matches(groups[n], 'spell_by_match') for n in groups}
                means = {n: float(np.mean(list(v.values()))) for n, v in vals.items()}
                diffs = {m: vals[7][m] - vals[4][m] for m in vals[7]}
                interval = ci(list(diffs.values()))
                delta = float(np.mean(list(diffs.values())))
                row = dict(source=src, conv=conv, dead=dead, means=means, delta_7_4=delta,
                           ci_7_4=interval, delta_by_match=diffs,
                           subset_mean_ranges={n: [min(r['mean'] for r in groups[n]), max(r['mean'] for r in groups[n])] for n in groups})
                summary['learning'].append(row)
                lines.append(f'| {src} / {conv} | {dead} | {means[2]:.4f} | {means[4]:.4f} | {means[7]:.4f} | {delta:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] |')
    lines += ['', 'GA: test maçları üzerinde eşli bootstrap; Metrica üç bozulma tohumu önce maç içinde ortalanır.',
              'Yedi alt küme bağımsız yedi veri kümesi değildir. GA, eğitim-maçı evreninin belirsizliğini kapsamaz.',
              'Üç Metrica maçıyla GA özellikle zayıftır. DFL içinde büyüme, Gradient alanına taşınmayı garanti etmez.', '',
              '## 2. Paslardan tüm mevcut temas vekillerine', '',
              'Ana test: event precision=.70, recall=.70, team accuracy=.95, jitter std=120 ms.',
              'team70: event precision=1, recall=.70, team accuracy=.70. β=4 önceden sabit; β=2 duyarlılık.',
              'Δ aynı 7 maçlık v7 modeline göre. Kontakların GT ile ortak kaynağı nedeniyle iyimser simülasyondur.', '',
              '| Veri / etiket | D | Çapa | Senaryo | β | ΔspellF1 | %95 GA | Artan maç |',
              '|---|---|---|---|---:|---:|---|---|']
    ids = sorted({(r['source'],r['conv'],r['dead'],r['anchors'],r['scenario'],r['beta']) for r in contacts})
    for src, conv, dead, mode, scenario, beta in ids:
        rs = [r for r in contacts if (r['source'],r['conv'],r['dead'],r['anchors'],r['scenario'],r['beta']) == (src,conv,dead,mode,scenario,beta)]
        expected = 1 if scenario == 'perfect' else 3
        if len(rs) != expected:
            continue
        vals = avg_matches(rs, 'delta_by_match')
        delta = float(np.mean(list(vals.values())))
        interval = ci(list(vals.values()))
        row = dict(source=src,conv=conv,dead=dead,anchors=mode,scenario=scenario,beta=beta,
                   delta=delta,ci=interval,positive_matches=sum(v>0 for v in vals.values()),matches=len(vals),delta_by_match=vals)
        summary['contacts'].append(row)
        lines.append(f'| {src} / {conv} | {dead} | {mode} | {scenario} | {beta:g} | {delta:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] | {row["positive_matches"]}/{len(vals)} |')
    lines += ['', 'Bu test her fiziksel dokunuşu kapsamaz. Kontrolsüz dokunuşu possession diye etiketlemez;',
              'temas takımını yalnız yerel olasılık kanıtı olarak ekler. Füzyon penceresi iki akışta da [−12,+3] kare.',
              '`contacts_typed` sonradan eklenen keşifsel duyarlılıktır: alım/top kazanımında [−3,+12],',
              'mücadelede [−3,+3], pas/şutta [−12,+3]. Olay türünü doğru bildiğini varsayar; ana test değildir.',
              'Gürültü tohumları önce maç içinde ortalanır. Eşik +0.03 test edilen yaklaşım için uygulanır.', '',
              '## 3. B2 ve etiket konvansiyonu', '',
              '| Veri / etiket | D | B2 kapalı | B2 açık | Fark |',
              '|---|---|---:|---:|---:|']
    for r in sensitivity:
        a,b = np.mean(list(r['plain'].values())), np.mean(list(r['b2'].values()))
        lines.append(f'| {r["source"]} / {r["conv"]} | {r["dead"]} | {a:.4f} | {b:.4f} | {r["delta"]:+.4f} |')
    lines += ['', 'B2 karşılaştırması sabit pen_team=12 ile yapılır; iki dekoder ayrı ayrı ayarlanmadı.',
              'Bu ölçüm özel GT üretim yöntemini teşhis etmez. Güncel resmi doküman owning alanının upstream',
              'üretimini açıklamıyor. Kaynak incelemesi ve gönderilmemiş soru taslağı: [RESEARCH.md](RESEARCH.md).', '',
              '## Yeniden üretim', '', 'PowerShell, repo kökünde:', '', '```powershell',
              "$env:BENCH_PROCS='3'; $env:BENCH_THREADS='8'",
              '.venv/Scripts/python.exe src/download_idsse.py',
              '.venv/Scripts/python.exe src/metrica_sample.py download',
              '.venv/Scripts/python.exe src/fetch_gate_data.py',
              '.venv/Scripts/python.exe src/exp_data_gate.py prepare',
              '.venv/Scripts/python.exe src/exp_data_gate.py train',
              '.venv/Scripts/python.exe src/exp_data_gate.py evaluate',
              '.venv/Scripts/python.exe src/exp_contact_gate.py',
              '.venv/Scripts/python.exe src/exp_label_gate.py',
              '.venv/Scripts/python.exe src/summarize_gates.py',
              '.venv/Scripts/python.exe -m unittest discover -s tests -v', '```', '',
              'Sürümler `requirements-gates.lock.txt`; deney tasarımı `design.json`. Frame düzeyindeki',
              'çıktılar/model dosyaları gitignore kapsamındaki `cache/gates/` altında. Yalnız toplu skorlar raporda.']
    gate.save_json(gate.REPORT / 'summary.json', summary)
    (gate.REPORT / 'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axs = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax, src, conv, title in zip(axs, gate.SOURCES, ('ctrl','pp'), ('Metrica controlled (3 matches)', 'SkillCorner player possession (20 matches)')):
        for dead, color in [('learned','#176c83'),('oracle','#cf7540')]:
            rs=[r for r in summary['learning'] if r['source']==src and r['conv']==conv and r['dead']==dead]
            if rs:
                ax.plot([2,4,7],[rs[0]['means'][n] for n in (2,4,7)],'o-',label=f'{dead} dead ball',color=color)
        ax.set(title=title,xlabel='DFL training matches',ylabel='spellF1',xticks=[2,4,7])
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(gate.REPORT / 'learning_curve.png',dpi=180)
    plt.close(fig)
    print('Wrote SUMMARY.md, summary.json, learning_curve.png', flush=True)


if __name__ == '__main__':
    main()
