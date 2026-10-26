"""Export the 25 Hz label tables and aggregate the fixed PFF experiment."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gradient
import exp_pff as exp
import exp_data_gate as gate


def export_one(match):
    out=gradient.CACHE/gradient.VERSION/'labels';out.mkdir(parents=True,exist_ok=True)
    path=gradient.CACHE/gradient.VERSION/f'{match}.npz'
    with np.load(path) as d:
        for per in d['periods']:
            dest=out/f'{match}_h{per}.csv.gz'
            if dest.exists(): continue
            tag=f'h{per}';t=d[tag+'_time'];n=d[tag+'_native'];c=d[tag+'_labels']
            valid=(n<2)&(c<2)
            pd.DataFrame(dict(half_frame25=np.arange(len(t)),video_time_s=t,
                controlled_label=np.array(['H','A','D','X'])[c],native_label=np.array(['H','A','D','X'])[n],
                team_training_eligible=valid.astype(np.int8))).to_csv(dest,index=False,float_format='%.2f',compression={'method':'gzip','mtime':0})
    return match


def data_report():
    audit=[gradient.read_json(gradient.CACHE/gradient.VERSION/f'{m}.json') for m in gradient.matches()]
    gate.save_json(exp.REPORT/'data_audit.json',audit)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(export_one,gradient.matches()))
    halves=[p for m in audit for p in m['periods']]
    counts=np.array([p['controlled_counts'] for p in halves]).sum(0)
    total=sum(p['n'] for p in halves)
    summary=dict(matches=len(audit),halves=len(halves),frames25=total,controlled_label_counts=counts.tolist(),
        controlled_label_hours=(counts/25/3600).tolist(),accepted_halves=sum(p['alignment_accepted'] for p in halves),
        rejected=[dict(match=m['match'],**p) for m in audit for p in m['periods'] if not p['alignment_accepted']],
        duplicate_rows=sum(m['raw_counts'].get('duplicate_rows',0) for m in audit),
        geometry_conflicts=sum(m['raw_counts'].get('duplicate_geometry_conflict',0) for m in audit),
        max_accepted_alignment_p95=max(p['alignment_residual_p95'] for p in halves if p['alignment_accepted']),
        median_native_control_disagree=float(np.median([p['native_control_disagree'] for p in halves if p['native_control_disagree'] is not None])),
        ignored_event_periods={m['match']:sorted(set(m['event_periods'])-{p['period'] for p in m['periods']}-{0}) for m in audit
                              if set(m['event_periods'])-{p['period'] for p in m['periods']}-{0}},
        label_dir=str(gradient.CACHE/gradient.VERSION/'labels'),
        source_hashes={name:hashlib.sha256((exp.HERE/name).read_bytes()).hexdigest() for name in ['gradient.py','exp_pff.py','report_pff.py']})
    transitions={'native':0,'controlled':0};common_live=0;disagree=0
    for match in gradient.matches():
        with np.load(gradient.CACHE/gradient.VERSION/f'{match}.npz') as d:
            for per in d['periods']:
                n=d[f'h{per}_native'];c=d[f'h{per}_labels'];t=d[f'h{per}_time']
                assert np.allclose(np.diff(t),.04,atol=1e-8),'Not a 25 Hz grid'
                for name,l in [('native',n),('controlled',c)]:
                    transitions[name]+=int(((l[1:]<2)&(l[:-1]<2)&(l[1:]!=l[:-1])).sum())
                mask=(n<2)&(c<2);common_live+=int(mask.sum());disagree+=int((n[mask]!=c[mask]).sum())
    summary.update(known_live_team_transitions=transitions,common_live_frames=common_live,
                   common_live_disagreement_frames=disagree,common_live_disagreement_fraction=disagree/max(1,common_live))
    gate.save_json(exp.REPORT/'data_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def ci(values):
    a=np.array(values);rng=np.random.default_rng(20261024)
    v=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1)
    return [float(x) for x in np.quantile(v,[.025,.975])]


def results():
    rows=json.loads((exp.REPORT/'comparison.json').read_text())
    assert len(rows)==32, f'Incomplete comparison: {len(rows)}/32'
    ref={(r['source'],r['conv'],r['dead']):r for r in rows if r['model']=='v7'}
    summary=[]
    lines=['# PFF 25 fps: kontrollü etiketler ve v7 karşılaştırması','',
           'Sabit deney: aynı 172 özellik, 400 LightGBM ağacı, stride 2, B2 / pen_team=12.',
           'Ana model DFL+PFF-controlled; PFF-only ve native etiketler tanı amaçlı. Ölü-top modeli sabit.',
           'PFF native/controlled eğitiminde aynı bilinen canlı kareler kullanılır. Dış testte ayar seçimi yok.', '',
           '| Model | Veri / konvansiyon | D | v7 spellF1 | Yeni | Δ | Eşli %95 GA | Artan maç |',
           '|---|---|---|---:|---:|---:|---|---|']
    for r in rows:
        if r['model']=='v7': continue
        baseline=ref[r['source'],r['conv'],r['dead']]
        delta={m:r['spell_by_match'][m]-baseline['spell_by_match'][m] for m in r['spell_by_match']}
        vals=list(delta.values());lo,hi=ci(vals);mean=float(np.mean(vals));wins=sum(v>0 for v in vals)
        x=dict(**r,baseline=baseline['spellF1'],delta=mean,ci=[lo,hi],positive_matches=wins,
               delta_by_match=delta,baseline_team_accuracy=baseline['team_accuracy'])
        summary.append(x)
        lines.append(f'| {r["model"]} | {r["source"]} / {r["conv"]} | {r["dead"]} | {baseline["spellF1"]:.4f} | {r["spellF1"]:.4f} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {wins}/{len(vals)} |')
    lines+=['','GA maç düzeyinde eşli bootstrap. Metrica bozulma tohumları önce maç içinde ortalanır;',
            '3 Metrica maçı bağımsız 9 maç gibi sayılmaz. Eğitim veri seçimi belirsizliği bu GA içinde değildir.',
            'Oracle D, gerçek D durumundan üretilen kuvvetli olasılık kanıtıdır; son çıktı D sınırlarını sertçe sabitlemez.',
            'Skorlar dış veri vekil etiketlerinedir; özel BAR-ATH skoru ölçülmedi.', '',
            '## Veri ve etiket çıktıları','',
            'Sayım/hizalama özeti: [data_summary.json](data_summary.json); maç bazında: [data_audit.json](data_audit.json).',
            'Etiket CSV.GZ: `cache/pff25/'+gradient.VERSION+'/labels/`. Her satır bir 25 fps karesi;',
            '`video_time_s` kaynak video zamanı, `half_frame25` yeniden örneklenen devrenin 0 tabanlı indeksi.',
            'H/A/D/X etiketlerinde X eğitim dışı/belirsizdir. X ölü top sayılmaz.',
            'Ham konumlar kullanıldı; PFF olaylardan oluşturulan yumuşatılmış top konumu özelliklere girmedi.', '',
            '## Yeniden çalıştırma','', '```powershell',
            '.venv/Scripts/python.exe src/exp_pff.py prepare --workers 6',
            '.venv/Scripts/python.exe src/report_pff.py data',
            '.venv/Scripts/python.exe src/exp_pff.py train',
            '.venv/Scripts/python.exe src/exp_pff.py evaluate',
            '.venv/Scripts/python.exe src/report_pff.py results',
            '.venv/Scripts/python.exe -m unittest discover -s tests -v','```']
    gate.save_json(exp.REPORT/'summary.json',summary)
    (exp.REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ablation=[]
    for r in rows:
        if r['model']!='dfl_pff_controlled': continue
        native=next(x for x in rows if x['model']=='dfl_pff_native' and
                    all(x[k]==r[k] for k in ('source','conv','dead')))
        delta={m:r['spell_by_match'][m]-native['spell_by_match'][m] for m in r['spell_by_match']}
        ablation.append(dict(source=r['source'],conv=r['conv'],dead=r['dead'],
            controlled=r['spellF1'],native=native['spellF1'],delta=float(np.mean(list(delta.values()))),
            ci=ci(list(delta.values())),delta_by_match=delta))
    gate.save_json(exp.REPORT/'label_ablation.json',ablation)
    print('Complete:',len(summary),'paired comparisons',flush=True)


if __name__=='__main__':
    import sys
    data_report() if sys.argv[1]=='data' else results()
