"""Internal-only selection of a fixed BiGRU/LGB probability blend.

Stages prepare -> train_nn/train_lgb -> select -> evaluate -> report.
diagnose is an independent frozen external audit, never an input to selection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('BENCH_THREADS','8')
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import warnings
import numpy as np
import torch
import lightgbm as lgb
import bench
import model_v7
import exp_seq_pff as neural
import exp_seed_balance as trained
import exp_data_gate as gate
import exp_external_ensemble as external
import possession_error_audit as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cache/hybrid_v1'
REPORT=ROOT/'reports/hybrid'
SEEDS=trained.SEEDS
GRID=(0.,.25,.5,.75,1.)  # weight on the three-seed BiGRU probability mean
read,save,sha=trained.read,trained.save,trained.sha


def blend(a,b,weight):
    if not 0<=weight<=1 or len(a)!=len(b):raise ValueError('Invalid blend inputs')
    out=[]
    for x,y in zip(a,b):
        x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Unaligned or invalid probabilities')
        if not (((x>=0)&(x<=1)).all() and ((y>=0)&(y<=1)).all()):raise ValueError('Probability out of range')
        out.append(weight*x+(1-weight)*y)
    return out


def prepare():
    trained.check_design()
    prior=read(neural.REPORT/'design.json')
    selection=read(neural.REPORT/'selection.json')['selected']['bigru']
    rows=read(neural.OUT/'index.json')
    for source in ('dfl','pff'):
        assert set(prior['train'][source]).isdisjoint(prior['validation'][source])
        for split,key in (('train','train'),('val','validation')):
            assert {r['match'] for r in rows if r['source']==source and r['split']==split}==set(prior[key][source])
    design=dict(date='2026-10-25',seeds=SEEDS,grid=GRID,alpha_means='BiGRU weight; 1-alpha LGB weight',
        train=prior['train'],validation=prior['validation'],config=selection['config'],epochs=selection['epoch'],
        neural_steps=neural.STEPS,neural_batch=neural.BATCH,scaler_seed=20261024,
        lgb_rounds=400,lgb_parameters=bench.LGB_TEAM,lgb_weights='N/(2*N_source), recomputed on training subset only',
        criterion='maximize equal-source mean of equal-match full-half oracle-D B2 spellF1',
        selection_labels='DFL native, PFF controlled with common-live mask; unchanged prior internal criterion',
        tie_break='larger alpha on exact tie',source_balance='50:50 both families',
        validation_caveat='same internal split previously used for architecture selection; not a fresh holdout',
        final_models='reuse existing 3 full-data BiGRU and 3 balanced-LGB fits, equal weights within each family',
        external_weight_search=False,barath_public_used=False,
        split_sha256=sha(neural.OUT/'index.json'),previous_selection_sha256=sha(neural.REPORT/'selection.json'),
        code={n:sha(ROOT/'src'/n) for n in ('exp_hybrid.py','possession_error_audit.py','seq_pff.py',
                                          'exp_seq_pff.py','exp_data_gate.py','exp_ctrl2.py','decode2.py')})
    design=json.loads(json.dumps(design));p=REPORT/'design.json'
    if p.exists():assert read(p)==design,'Frozen design changed'
    save(p,design)
    print('Frozen 5 candidate weights, 5 DFL + 52 PFF train / 2 DFL + 12 PFF validation.',flush=True)


def check():
    d=read(REPORT/'design.json')
    for n,h in d['code'].items():assert sha(ROOT/'src'/n)==h
    assert sha(neural.OUT/'index.json')==d['split_sha256']
    assert sha(neural.REPORT/'selection.json')==d['previous_selection_sha256']
    return d


def model_path(kind,seed):return OUT/f'{kind}_internal_s{seed}.{("pt" if kind=="bigru" else "txt")}'


def train_nn():
    d=check();rows=read(neural.OUT/'index.json')
    data=neural.Data([r for r in rows if r['split']=='train'])
    neural.SEED=20261024;scaler=data.scaler();neural.REPORT=REPORT
    for seed in SEEDS:
        dest=model_path('bigru',seed);a=REPORT/(dest.stem+'_artifact.json')
        if a.exists():assert sha(dest)==read(a)['sha256'];continue
        neural.SEED=seed
        neural.train_one(d['config'],data,scaler,d['epochs'],dest)
        save(a,dict(path=str(dest),sha256=sha(dest),seed=seed,train=d['train'],excluded=d['validation'],
                    scaler_train_only=True,epochs=d['epochs'],validation_early_stopping=False))


def subset_rows():
    rows=[r for r in read(neural.OUT/'index.json') if r['split']=='train']
    meta=OUT/'lgb_rows.json';xp=OUT/'lgb_X.npy';yp=OUT/'lgb_y.npy';sp=OUT/'lgb_source.npy'
    if meta.exists():
        return np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'),np.load(sp,mmap_mode='r'),read(meta)
    counts=[int((np.load(r['y'])[::2]<2).sum()) for r in rows]
    n=2*sum(counts)
    X=np.lib.format.open_memmap(xp,mode='w+',dtype=np.float32,shape=(n,172))
    y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.int8,shape=(n,))
    source=np.lib.format.open_memmap(sp,mode='w+',dtype=np.int8,shape=(n,))
    start=0
    for r,c in zip(rows,counts):
        x=np.load(r['x'],mmap_mode='r');lab=np.load(r['y'])[::2];idx=np.flatnonzero(lab<2)
        assert len(x)==len(lab)
        for side in (0,1):
            end=start+c;X[start:end]=x[idx,side];y[start:end]=(lab[idx]==side);source[start:end]=int(r['source']=='pff');start=end
    assert start==n
    X.flush();y.flush();source.flush()
    info=dict(rows=n,rows_by_source=np.bincount(source,minlength=2).tolist(),
              matches={s:sorted({r['match'] for r in rows if r['source']==s}) for s in ('dfl','pff')},
              row_order='original index half order; home then away within each half; stride2 live frames')
    save(meta,info);return X,y,source,info


def train_lgb():
    d=check();X,y,source,info=subset_rows();assert info['matches']==d['train']
    weights=trained.source_weights(source)
    save(REPORT/'internal_lgb_weights.json',dict(**info,weight_sums=[float(weights[source==i].sum(dtype=np.float64)) for i in (0,1)]))
    for seed in SEEDS:
        dest=model_path('lgb_balanced',seed);a=REPORT/(dest.stem+'_artifact.json')
        if a.exists():assert sha(dest)==read(a)['sha256'];continue
        print('Train internal LGB',seed,flush=True)
        params=dict(bench.LGB_TEAM,seed=seed,data_random_seed=seed,feature_fraction_seed=seed,bagging_seed=seed)
        dataset=lgb.Dataset(X,label=y,weight=weights,free_raw_data=True)
        model=lgb.train(params,dataset,num_boost_round=400);model.save_model(str(dest))
        save(a,dict(path=str(dest),sha256=sha(dest),seed=seed,train=d['train'],excluded=d['validation'],parameters=params,rounds=400))
        del model,dataset;gc.collect()
        print('Completed internal LGB',seed,flush=True)


def validation_probs():
    rows=[r for r in read(neural.OUT/'index.json') if r['split']=='val']
    data=neural.Data(rows);neural.init_device(20261024);family={};artifacts=[]
    for kind in ('bigru','lgb_balanced'):
        members=[]
        for seed in SEEDS:
            path=model_path(kind,seed);art=read(REPORT/(path.stem+'_artifact.json'))
            assert sha(path)==art['sha256'];artifacts.append(art)
            pp=OUT/(path.stem+'_validation.pkl')
            if pp.exists():ph=bench._load(str(pp))
            elif kind=='bigru':
                net,sc,_=neural.load_model(path)
                ph=[neural.seq.predict_raw(net,sc,x,r['full_length']) for x,r in zip(data.x,rows)]
                del net;torch.cuda.empty_cache();bench._dump(ph,str(pp))
            else:
                model=lgb.Booster(model_file=str(path));ph=[]
                # LGB consumes full 25Hz features, not interpolated stride2 features.
                dfl_blocks={b['key']:b for b in bench.blocks('dfl@v3')}
                for r in rows:
                    if r['source']=='pff':
                        half=int(r['key'].rsplit('_h',1)[1]);fp=trained.pff.paths(r['match'],half)
                    else:
                        bb=[b for b in dfl_blocks.values() if b['group']==r['match']]
                        # Key suffix is copied verbatim by the neural materializer.
                        b=next(b for b in bb if r['key']==f'dfl_{r["match"]}_{b["key"].split("/")[-1]}')
                        fp=Path(bench.CACHE)/'feat'/bench._safe(model_v7.VERSION)/(b['key'].replace('/','_')+'.pkl')
                    f=bench._load(str(fp));assert len(f['XF'])==r['full_length']
                    ph.append(bench.predict_team(model,f));del f
                del model;bench._blocks.clear();bench._dump(ph,str(pp))
            members.append(ph)
        family[kind]=external.average(members,[r['full_length'] for r in rows])
    save(REPORT/'internal_artifacts.json',artifacts)
    return data,family


def select():
    d=check();data,family=validation_probs()
    pd=[np.where(y==2,1-1e-4,1e-4) for y in data.full_y];results=[]
    for alpha in GRID:
        ph=blend(family['bigru'],family['lgb_balanced'],alpha)
        preds=gate.decode_all(pd,ph);by={}
        for source in ('dfl','pff'):
            idx=[i for i,r in enumerate(data.rows) if r['source']==source]
            sc=bench.spell_by_group([dict(group=data.rows[i]['match']) for i in idx],
                                   [data.full_y[i] for i in idx],[preds[i] for i in idx])
            by[source]={m:float(v[0]) for (m,_),v in sc.items()}
        means={s:float(np.mean(list(v.values()))) for s,v in by.items()}
        row=dict(alpha=alpha,balanced_spellF1=float(np.mean(list(means.values()))),by_source=means,spell_by_match=by)
        results.append(row);print('Internal weight',alpha,'score',row['balanced_spellF1'],means,flush=True)
    winner=max(results,key=lambda r:(r['balanced_spellF1'],r['alpha']))
    result=dict(selected=winner,candidates=results,design_sha256=sha(REPORT/'design.json'),
        internal_artifacts_sha256=sha(REPORT/'internal_artifacts.json'),
        external_labels_used=False,frozen_at=datetime.now(timezone.utc).isoformat())
    target=REPORT/'selection.json'
    if target.exists():
        old=read(target);assert old['selected']==winner and old['candidates']==results
    else:save(target,result)
    print('FROZEN selected BiGRU weight',winner['alpha'],flush=True)


def external_probs(source):
    lengths=[len(b['P']) for b in bench.blocks(source)]
    result={};provenance=[]
    for kind in ('bigru','lgb_balanced'):
        members=[]
        for seed in SEEDS:
            path=trained.model_path(kind,seed);a=read(trained.REPORT/(path.stem+'_artifact.json'))
            assert sha(path)==a['sha256']
            pp=trained.OUT/f'{kind}_s{seed}_{source.replace("@","-")}_probs.pkl'
            members.append(bench._load(str(pp)))
            provenance.append(dict(kind=kind,seed=seed,model_sha256=a['sha256'],probabilities_sha256=sha(pp)))
        result[kind]=external.average(members,lengths)
    tag=source.replace('@','-')
    original=bench._load(str(external.OUT/f'ensemble_{tag}.pkl'))
    assert all(np.array_equal(a,b) for a,b in zip(original,result['bigru']))
    result['v7']=bench._load(str(external.OUT/f'v7_{tag}.pkl'))
    save(REPORT/f'provenance_{tag}.json',provenance)
    return result


def score_rows(source,conv,decoded):
    scores=bench.spell_by_group(bench.blocks(source),bench.labels(source,conv),decoded)
    matches={}
    for (m,s),v in scores.items():matches.setdefault(m,[]).append(v[:2])
    matches={m:np.mean(v,axis=0) for m,v in matches.items()}
    return dict(spell_by_match={m:float(v[0]) for m,v in matches.items()},
                spellF1=float(np.mean([v[0] for v in matches.values()])),
                macroF1=float(np.mean([v[1] for v in matches.values()])))


def evaluate(diagnostic=False):
    check();selection=None
    if not diagnostic:
        selection=read(REPORT/'selection.json')
        assert selection['design_sha256']==sha(REPORT/'design.json')
        assert selection['internal_artifacts_sha256']==sha(REPORT/'internal_artifacts.json')
    rows=[];errors=[];complements=[];contrasts=[]
    for source in gate.SOURCES:
        print('External',source,'diagnostic',diagnostic,flush=True)
        blocks=bench.blocks(source);ps=external_probs(source)
        if selection:ps['hybrid']=blend(ps['bigru'],ps['lgb_balanced'],selection['selected']['alpha'])
        labs={c:bench.labels(source,c) for c in gate.CONVS[source]}
        modes={'learned':bench.dead_ext('base','v3',source)}
        if not diagnostic:modes['oracle']=[np.where(y==2,1-1e-4,1e-4) for y in next(iter(labs.values()))]
        for mode,pd in modes.items():
            decoded={name:gate.decode_all(pd,ph) for name,ph in ps.items()}
            for conv,labels in labs.items():
                for name,pred in decoded.items():
                    row=dict(model=name,source=source,conv=conv,dead=mode,**score_rows(source,conv,pred));rows.append(row)
                    if name in ('v7','bigru'):
                        old=next(r for r in read(external.REPORT/'results.json') if
                            (r['model'],r['source'],r['conv'],r['dead'])==
                            ('bigru_ensemble' if name=='bigru' else name,source,conv,mode))
                        assert all(abs(v-old['spell_by_match'][m])<1e-12 for m,v in row['spell_by_match'].items())
                    if mode=='learned':errors.append(dict(model=name,source=source,conv=conv,**audit.diagnostics(blocks,labels,pred)))
                    print(source,conv,mode,name,round(row['spellF1'],6),flush=True)
                if mode=='learned':complements.append(dict(source=source,conv=conv,
                    **audit.complementary(blocks,labels,decoded['bigru'],decoded['lgb_balanced'])))
                if selection:
                    subset=[r for r in rows if (r['source'],r['conv'],r['dead'])==(source,conv,mode)]
                    hybrid=next(r for r in subset if r['model']=='hybrid')
                    for base in ('v7','bigru','lgb_balanced'):
                        baseline=next(r for r in subset if r['model']==base)
                        contrasts.append(dict(source=source,conv=conv,dead=mode,baseline=base,
                            **external.paired_summary(hybrid['spell_by_match'],baseline['spell_by_match'])))
            if selection:
                for name,pred in decoded.items():bench._dump(pred,str(OUT/f'{source.replace("@","-")}_{mode}_{name}_decoded.pkl'))
            del decoded
        bench._blocks.clear();gc.collect()
    prefix='diagnostic_' if diagnostic else ''
    save(REPORT/(prefix+'results.json'),rows);save(REPORT/(prefix+'errors.json'),errors)
    save(REPORT/(prefix+'complementarity.json'),complements)
    if selection:
        save(REPORT/'contrasts.json',contrasts)
        save(REPORT/'verification.json',dict(selected_alpha=selection['selected']['alpha'],
            selection_sha256=sha(REPORT/'selection.json'),external_cells=len(rows),
            prior_v7_and_ensemble_match_scores_reproduced=True,weights_unchanged_on_external=True,
            validation_excluded_from_six_internal_fits=True))


def report():
    selected=read(REPORT/'selection.json')['selected'];rows=read(REPORT/'results.json');cs=read(REPORT/'contrasts.json')
    lines=['# Hybrid — frozen internal selection and external evaluation','',
        f'Selected weight: BiGRU {selected["alpha"]}; balanced LGB {1-selected["alpha"]}.',
        'Three seeds averaged within each family. All settings fixed before external evaluation.','',
        '| Internal BiGRU weight | DFL native | PFF controlled | Equal-source mean |',
        '|---|---:|---:|---:|']
    for r in read(REPORT/'selection.json')['candidates']:
        lines.append(f'| {r["alpha"]:.2f} | {r["by_source"]["dfl"]:.6f} | {r["by_source"]["pff"]:.6f} | {r["balanced_spellF1"]:.6f} |')
    lines+=['','| Test | D | v7 | BiGRU ensemble | Balanced LGB ensemble | Selected hybrid | Delta vs BiGRU (95% CI) |',
            '|---|---|---:|---:|---:|---:|---|']
    for source in gate.SOURCES:
        for conv in gate.CONVS[source]:
            for mode in ('learned','oracle'):
                sub={r['model']:r for r in rows if (r['source'],r['conv'],r['dead'])==(source,conv,mode)}
                c=next(c for c in cs if (c['source'],c['conv'],c['dead'],c['baseline'])==(source,conv,mode,'bigru'))
                lines.append(f'| {source}/{conv} | {mode} | '+ ' | '.join(f'{sub[m]["spellF1"]:.6f}' for m in ('v7','bigru','lgb_balanced','hybrid'))+
                    f' | {c["delta"]:+.6f} [{c["ci95"][0]:+.6f}, {c["ci95"][1]:+.6f}] |')
    lines+=['','Oracle-D is diagnostic. Main results use identical learned-D and fixed B2 for every model.',
        'Metrica has 3 matches, noise draws averaged within match; SkillCorner has 20 matches.',
        'Both internal validation and external benchmarks were used previously; these are not untouched holdouts.',
        'DFL internal selection uses native labels as in the original neural selection, not the external controlled-label proxy.',
        'No BAR-ATH scoring, CSV replacement, threshold search or external weight sweep was performed.']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','train_nn','train_lgb','select','diagnose','evaluate','report'))
    stage=p.parse_args().stage
    if stage=='diagnose':evaluate(diagnostic=True)
    else:globals()[stage]()
