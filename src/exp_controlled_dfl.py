"""DFL controlled-label BiGRU: frozen internal selection, then external test.

Features, PFF labels, source balance and B2 are unchanged. DFL labels come
from the repository's clean-tracking control proxy; original files untouched.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('BENCH_THREADS','8')
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import warnings
import numpy as np
import torch
import bench
import dfl
import exp_seq_pff as neural
import exp_seed_balance as trained
import exp_data_gate as gate
import exp_external_ensemble as external
import exp_hybrid as hybrid
import possession_error_audit as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cache/controlled_dfl_bigru_v1'
REPORT=ROOT/'reports/controlled_dfl_bigru'
SEEDS=trained.SEEDS
CONFIGS=[c for c in neural.CONFIGS if c['kind']=='bigru']
EPOCHS=11
read,save,sha=trained.read,trained.save,trained.sha


def label_stats(native,controlled):
    if len(native)!=len(controlled):raise ValueError('Label alignment mismatch')
    if not np.array_equal(native<2,controlled<2) or not np.array_equal(native[native>=2],controlled[native>=2]):
        raise ValueError('Controlled conversion changed dead/unknown/live masks')
    def changes(y):return int(((y[:-1]<2)&(y[1:]<2)&(y[:-1]!=y[1:])).sum())
    return dict(frames=len(native),live=int((native<2).sum()),changed=int((native!=controlled).sum()),
        native_changes=changes(native),controlled_changes=changes(controlled))


def prepare():
    prior=read(neural.REPORT/'design.json');original=read(neural.OUT/'index.json')
    design=dict(date='2026-10-25',seeds=SEEDS,configs=CONFIGS,epochs=EPOCHS,
        train=prior['train'],validation=prior['validation'],
        dfl_labels='bench.trk_ctrl on clean DFL tracking; r1.5m,zmax2m,keep25 frames, two players or >=25 owner frames within spell',
        dfl_proxy_caveat='control inferred from tracking; owner frames need not be consecutive; two players do not prove a completed pass',
        pff_labels='unchanged controlled raw_control_v2 and common-live mask',
        internal_criterion='max equal-source/equal-match oracle-D B2 spellF1 of 3-seed probability ensemble',
        selection_labels='DFL trk and PFF controlled',tie_break='first config in frozen order',
        epochs_fixed_from_previous_protocol=True,scaler_seed=20261024,source_balance='exact 50:50 windows and loss',
        steps=neural.STEPS,batch=neural.BATCH,decoder='unchanged B2 pen12,dead30,minposs8,mindead50,D0=175,c48',
        refit='three seeds on all 7 DFL + 64 PFF; also fixed old bigru_64 config if another config wins',
        prior_validation_used=True,external_tuning=False,barath_public_used=False,
        original_index_sha256=sha(neural.OUT/'index.json'),
        code={n:sha(ROOT/'src'/n) for n in ('exp_controlled_dfl.py','bench.py','seq_pff.py',
            'exp_seq_pff.py','exp_data_gate.py','exp_ctrl2.py','decode2.py','possession_error_audit.py','exp_hybrid.py')})
    design=json.loads(json.dumps(design));dp=REPORT/'design.json'
    if dp.exists():assert read(dp)==design,'Frozen design changed'
    save(dp,design)
    cached=bench._load(str(Path(bench.CACHE)/'dfl_conv.pkl'))
    converted={};stats=[]
    for match in dfl.MATCHES:
        raw=dfl.parse(match);pitch=tuple(float(v) for v in raw['pitch'])
        for half in (1,2):
            tag=f'h{half}';native=raw[tag+'_labels']
            ctrl=bench.trk_ctrl(native,raw[tag+'_players'],raw[tag+'_ball'],raw[tag+'_team'],pitch)
            np.testing.assert_array_equal(ctrl,cached[match][half-1]['trk'])
            info=label_stats(native,ctrl);info.update(match=match,half=half)
            fp=OUT/'labels'/f'dfl_{match}_{tag}.npy';fp.parent.mkdir(parents=True,exist_ok=True)
            np.save(fp,ctrl.astype(np.int8));converted[f'dfl_{match}_{tag}']=fp
            stats.append(info)
        del raw;gc.collect()
    rows=[]
    for r in original:
        row=dict(r)
        if r['source']=='dfl':
            fp=converted[r['key']]
            label_stats(np.load(r['y']),np.load(fp))
            row['y']=str(fp)
        rows.append(row)
    save(OUT/'index.json',rows)
    totals={k:sum(r[k] for r in stats) for k in ('frames','live','changed','native_changes','controlled_changes')}
    save(REPORT/'label_audit.json',dict(by_half=stats,total=totals,
        changed_live_fraction=totals['changed']/totals['live'],cached_labels_recomputed_identically=True,
        labels={r['key']:sha(Path(r['y'])) for r in rows if r['source']=='dfl'},
        index_sha256=sha(OUT/'index.json'),dead_unknown_and_live_masks_unchanged=True))
    print('Label audit',totals,'changed live fraction',totals['changed']/totals['live'],flush=True)


def check():
    d=read(REPORT/'design.json')
    for n,h in d['code'].items():assert sha(ROOT/'src'/n)==h
    assert sha(neural.OUT/'index.json')==d['original_index_sha256']
    a=read(REPORT/'label_audit.json');assert sha(OUT/'index.json')==a['index_sha256']
    for r in read(OUT/'index.json'):
        if r['source']=='dfl':assert sha(Path(r['y']))==a['labels'][r['key']]
    return d


def path(config,seed,stage):return OUT/f'{config["name"]}_{stage}_s{seed}.pt'


def train(stage):
    d=check();rows=read(OUT/'index.json')
    if stage=='internal':
        rows=[r for r in rows if r['split']=='train'];configs=CONFIGS
    else:
        selection=read(REPORT/'selection.json');assert selection['design_sha256']==sha(REPORT/'design.json')
        names={selection['selected']['config']['name'],'bigru_64'}
        configs=[c for c in CONFIGS if c['name'] in names]
    data=neural.Data(rows);neural.SEED=20261024;scaler=data.scaler()
    # A pure label intervention must leave preprocessing unchanged.
    oldrows=read(neural.OUT/'index.json')
    if stage=='internal':oldrows=[r for r in oldrows if r['split']=='train']
    oldsc=neural.Data(oldrows).scaler()
    np.testing.assert_array_equal(scaler.mean,oldsc.mean);np.testing.assert_array_equal(scaler.scale,oldsc.scale)
    neural.REPORT=REPORT
    for config in configs:
        for seed in SEEDS:
            dest=path(config,seed,stage);ap=REPORT/(dest.stem+'_artifact.json')
            if ap.exists():assert sha(dest)==read(ap)['sha256'];continue
            neural.SEED=seed
            neural.train_one(config,data,scaler,EPOCHS,dest)
            save(ap,dict(path=str(dest),sha256=sha(dest),seed=seed,config=config,epochs=EPOCHS,
                stage=stage,train_matches={s:sorted({r['match'] for r in rows if r['source']==s}) for s in ('dfl','pff')},
                excluded=d['validation'] if stage=='internal' else {},scaler_identical_to_native_labels=True))


def score_internal(ph,data):
    pd=[np.where(y==2,1-1e-4,1e-4) for y in data.full_y]
    preds=gate.decode_all(pd,ph);scores={}
    for source in ('dfl','pff'):
        idx=[i for i,r in enumerate(data.rows) if r['source']==source]
        r=bench.spell_by_group([dict(group=data.rows[i]['match']) for i in idx],
            [data.full_y[i] for i in idx],[preds[i] for i in idx])
        scores[source]={m:float(v[0]) for (m,_),v in r.items()}
    means={s:float(np.mean(list(v.values()))) for s,v in scores.items()}
    return dict(balanced_spellF1=float(np.mean(list(means.values()))),by_source=means,spell_by_match=scores)


def select():
    d=check();rows=[r for r in read(OUT/'index.json') if r['split']=='val'];data=neural.Data(rows)
    neural.init_device(20261024);results=[];artifacts=[]
    for config in CONFIGS:
        members=[]
        for seed in SEEDS:
            p=path(config,seed,'internal');art=read(REPORT/(p.stem+'_artifact.json'))
            assert sha(p)==art['sha256'];artifacts.append(art)
            pp=OUT/(p.stem+'_val.pkl')
            if pp.exists():ph=bench._load(str(pp))
            else:
                net,sc,_=neural.load_model(p)
                ph=[neural.seq.predict_raw(net,sc,x,r['full_length']) for x,r in zip(data.x,rows)]
                bench._dump(ph,str(pp));del net;torch.cuda.empty_cache()
            members.append(ph)
        ph=external.average(members,[r['full_length'] for r in rows])
        result=dict(config=config,**score_internal(ph,data));results.append(result)
        print('Internal controlled',config['name'],result['balanced_spellF1'],result['by_source'],flush=True)
    # Native-label model reference scored against the SAME controlled val labels;
    # reference only, not a candidate in the requested controlled-training search.
    native=[bench._load(str(hybrid.OUT/f'bigru_internal_s{s}_validation.pkl')) for s in SEEDS]
    reference=score_internal(external.average(native,[r['full_length'] for r in rows]),data)
    selected=max(results,key=lambda r:r['balanced_spellF1'])
    result=dict(selected=selected,candidates=results,native_reference_same_validation_labels=reference,
        artifacts=artifacts,design_sha256=sha(REPORT/'design.json'),external_labels_used=False,
        frozen_at=datetime.now(timezone.utc).isoformat())
    dp=REPORT/'selection.json'
    if dp.exists():assert read(dp)['selected']==selected and read(dp)['candidates']==results
    else:save(dp,result)
    print('Selected',selected['config']['name'],'native reference',reference['balanced_spellF1'],flush=True)


def evaluate():
    check();selection=read(REPORT/'selection.json');selected=selection['selected']['config']
    assert selection['design_sha256']==sha(REPORT/'design.json')
    configs=[c for c in CONFIGS if c['name'] in {selected['name'],'bigru_64'}]
    neural.init_device(20261024);results=[];errors=[];contrasts=[];provenance=[]
    for source in gate.SOURCES:
        print('External',source,flush=True)
        blocks=bench.blocks(source);lengths=[len(b['P']) for b in blocks]
        fs=bench.features(neural.model_v7.VERSION,source)
        families={}
        for config in configs:
            members=[]
            for seed in SEEDS:
                p=path(config,seed,'final');art=read(REPORT/(p.stem+'_artifact.json'))
                assert sha(p)==art['sha256']
                pp=OUT/f'{p.stem}_{source.replace("@","-")}_probs.pkl'
                if pp.exists():ph=bench._load(str(pp))
                else:
                    net,sc,_=neural.load_model(p)
                    ph=[neural.seq.predict_raw(net,sc,neural.seq.views(f),len(f['XF'])) for f in fs]
                    bench._dump(ph,str(pp));del net;torch.cuda.empty_cache()
                members.append(ph);provenance.append(dict(source=source,config=config['name'],seed=seed,
                    model_sha256=art['sha256'],probabilities_sha256=sha(pp)))
            families[config['name']]=external.average(members,lengths)
        del fs;gc.collect()
        tag=source.replace('@','-')
        ps=dict(native_ensemble=bench._load(str(external.OUT/f'ensemble_{tag}.pkl')),
            controlled_fixed=families['bigru_64'],controlled_selected=families[selected['name']])
        labs={c:bench.labels(source,c) for c in gate.CONVS[source]}
        dead=dict(learned=bench.dead_ext('base','v3',source),
                  oracle=[np.where(y==2,1-1e-4,1e-4) for y in next(iter(labs.values()))])
        for mode,pd in dead.items():
            decoded={n:gate.decode_all(pd,ph) for n,ph in ps.items()}
            for conv,labels in labs.items():
                subset={}
                for name,pred in decoded.items():
                    r=dict(model=name,source=source,conv=conv,dead=mode,**hybrid.score_rows(source,conv,pred))
                    results.append(r);subset[name]=r
                    if name=='native_ensemble':
                        prior=next(r for r in read(external.REPORT/'results.json') if
                            (r['model'],r['source'],r['conv'],r['dead'])==('bigru_ensemble',source,conv,mode))
                        assert r['spell_by_match']==prior['spell_by_match']
                    if mode=='learned':errors.append(dict(model=name,source=source,conv=conv,
                        **audit.diagnostics(blocks,labels,pred)))
                    print(source,conv,mode,name,round(r['spellF1'],6),flush=True)
                for name in ('controlled_fixed','controlled_selected'):
                    contrasts.append(dict(model=name,source=source,conv=conv,dead=mode,
                        **external.paired_summary(subset[name]['spell_by_match'],subset['native_ensemble']['spell_by_match'])))
            for n,pred in decoded.items():bench._dump(pred,str(OUT/f'{tag}_{mode}_{n}_decoded.pkl'))
        bench._blocks.clear();del blocks,ps,families;gc.collect()
    save(REPORT/'results.json',results);save(REPORT/'contrasts.json',contrasts);save(REPORT/'errors.json',errors)
    save(REPORT/'prediction_manifest.json',dict(selection_sha256=sha(REPORT/'selection.json'),models=provenance,
        external_cells=len(results),prior_ensemble_reproduced=True))


def report():
    sel=read(REPORT/'selection.json');rs=read(REPORT/'results.json');cs=read(REPORT/'contrasts.json')
    lines=['# Controlled DFL BiGRU results','',f'Selected config: {sel["selected"]["config"]}.',
        'All ensembles average seeds 20261025/26/27. Same features, PFF labels, balance and decoder.',
        'DFL controlled labels are a tracking-derived proxy, not human ground truth.','',
        '| Internal config | DFL controlled | PFF controlled | Equal-source mean |','|---|---:|---:|---:|']
    refs=[dict(config=dict(name='native reference'),**sel['native_reference_same_validation_labels'])]+sel['candidates']
    for r in refs:lines.append(f'| {r["config"]["name"]} | {r["by_source"]["dfl"]:.6f} | {r["by_source"]["pff"]:.6f} | {r["balanced_spellF1"]:.6f} |')
    lines+=['','| Test | D | Native ensemble | Controlled fixed | Controlled selected | Selected delta [95% CI] | Wins |',
        '|---|---|---:|---:|---:|---|---|']
    for source in gate.SOURCES:
        for conv in gate.CONVS[source]:
            for mode in ('learned','oracle'):
                sub={r['model']:r for r in rs if (r['source'],r['conv'],r['dead'])==(source,conv,mode)}
                c=next(r for r in cs if (r['model'],r['source'],r['conv'],r['dead'])==('controlled_selected',source,conv,mode))
                lines.append(f'| {source}/{conv} | {mode} | '+ ' | '.join(f'{sub[n]["spellF1"]:.6f}' for n in
                    ('native_ensemble','controlled_fixed','controlled_selected'))+
                    f' | {c["delta"]:+.6f} [{c["ci95"][0]:+.6f}, {c["ci95"][1]:+.6f}] | {c["wins"]}/{c["matches"]} |')
    lines+=['','Main result: learned-D. Oracle-D is diagnostic. No external tuning.',
        'Metrica: three matches, noise draws averaged within match. SkillCorner: twenty matches.',
        '95% intervals are paired match bootstraps, conditional on fixed models.',
        'The internal split and external sets were used previously; not fresh holdouts.']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=['# Controlled DFL: error diagnostics','',
        'IoU=.5 diagnostic counts. Rates pool match counts after averaging Metrica noise seeds.',
        'Split/merge are positive-overlap proxies, not exclusive causal diagnoses.','',
        '| Test | Model | Split proxy | Merge proxy | Missed GT | Unmatched predictions | <4s missed |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in read(REPORT/'errors.json'):
        c=r['counts'];q=r['rates'];den=c['0-2s_gt']+c['2-4s_gt']
        vals=[q['split_proxy_rate'],q['merge_proxy_rate'],q['missed_rate'],q['false_discovery_rate'],
              (c['0-2s_missed']+c['2-4s_missed'])/den]
        lines.append(f'| {r["source"]}/{r["conv"]} | {r["model"]} | '+' | '.join(f'{v*100:.3f}%' for v in vals)+' |')
    (REPORT/'ERRORS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','train_internal','select','refit','evaluate','report'))
    stage=p.parse_args().stage
    if stage=='train_internal':train('internal')
    elif stage=='refit':train('final')
    else:globals()[stage]()
