"""PFF controlled-label data expansion, fixed v7 features/model/decoder.

Primary: DFL 7 native + PFF 64 controlled. Diagnostics: PFF-only controlled,
DFL+PFF native labels. Same 400 trees, stride=2, source-unweighted sample pooling.
No external-test tuning; dead model fixed to the prior seven-DFL model.
Raw PFF tracking is already broadcast-domain; do NOT apply DFL sim.V3 again.
Large training arrays use disk-backed memmaps, keeping peak memory bounded.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault('BENCH_THREADS','8')
os.environ.setdefault('BENCH_PROCS','3')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np
import lightgbm as lgb
import bench
import model_v7
import gradient
import exp_data_gate as gate
import exp_ctrl2 as e2

ROOT=HERE.parent
OUT=ROOT/'cache/pff_experiment'/gradient.VERSION
REPORT=ROOT/'reports/pff'
MODELS=('dfl_pff_controlled','pff_controlled','dfl_pff_native')


def paths(match,per):
    return OUT/'features'/f'{match}_h{per}.pkl'


def prepare_one(match):
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    t=time.time()
    path,info=gradient.parse(match)
    d=np.load(path)
    for per in d['periods']:
        tag=f'h{per}';target=paths(match,int(per))
        if target.exists(): continue
        lab=d[tag+'_labels']
        if not ((lab==0)|(lab==1)).any(): continue
        block=dict(P=d[tag+'_players'],B=d[tag+'_ball'],team=d['team'],gk=d['gk'],pitch=tuple(d['pitch']))
        bench._block_feats((model_v7.VERSION,block,str(target)))
    d.close()
    print('prepared',match,'seconds',round(time.time()-t),
          'accepted',sum(x['alignment_accepted'] for x in info['periods']),
          'live_frames',sum(x['controlled_counts'][0]+x['controlled_counts'][1] for x in info['periods']),flush=True)
    return info


def prepare(workers=3):
    REPORT.mkdir(parents=True,exist_ok=True)
    design=dict(version=gradient.VERSION,features=model_v7.VERSION,models=MODELS,
                rounds=bench.ROUNDS,stride=bench.STRIDE,parameters=bench.LGB_TEAM,
                input='raw PFF coordinates only; no Smoothed fields; no extra synthetic degradation',
                pff_matches=gradient.matches(),baseline_model=str(gate.model_path(7,0)),
                dead_model='fixed seven-DFL base_v3',decoder=dict(pen_team=12,b2_D0=175,b2_c=48),
                source_weighting='equal frame weight; same stride 2 for both sources',
                pff_training_mask='intersection of known-live native and controlled labels (same frames for label ablation)',
                external_test_tuning=False)
    gate.save_json(REPORT/'design.json',design)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(prepare_one,gradient.matches()))
    gate.save_json(REPORT/'data_audit.json',results)


def training_items(model):
    items=[]
    if model.startswith('dfl_'):
        for b in bench.blocks('dfl@v3'):
            fpath=Path(bench.CACHE)/'feat'/bench._safe(model_v7.VERSION)/(b['key'].replace('/','_')+'.pkl')
            items.append((fpath,b['lab'],b['group']))
        bench._blocks.clear()
    conv='native' if model.endswith('native') else 'labels'
    for match in gradient.matches():
        with np.load(gradient.CACHE/gradient.VERSION/f'{match}.npz') as d:
            for per in d['periods']:
                p=paths(match,int(per))
                if p.exists():
                    lab=d[f'h{per}_{conv}'].copy()
                    common=(d[f'h{per}_native']<2)&(d[f'h{per}_labels']<2)
                    lab[~common]=3
                    items.append((p,lab,match))
    return items


def build_rows(model):
    folder=OUT/'rows';folder.mkdir(parents=True,exist_ok=True)
    xp=folder/f'{model}_X.npy';yp=folder/f'{model}_y.npy';meta=folder/f'{model}.json'
    if xp.exists() and yp.exists() and meta.exists():
        return np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'),json.loads(meta.read_text())
    items=training_items(model)
    counts=[int((l[::bench.STRIDE]<2).sum()) for _,l,_ in items]
    n=2*sum(counts)
    first=bench._load(str(items[0][0]));fk,tk=first['fk'],first['tk'];width=len(fk)+len(tk);del first
    X=np.lib.format.open_memmap(xp,mode='w+',dtype=np.float32,shape=(n,width))
    y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.int8,shape=(n,))
    start=0;by_match={}
    for (path,lab,match),count in zip(items,counts):
        f=bench._load(str(path));assert f['fk']==fk and f['tk']==tk,'Feature schema mismatch'
        idx=np.arange(0,len(lab),bench.STRIDE);idx=idx[lab[idx]<2]
        for k,team in enumerate(('XH','XA')):
            end=start+count
            X[start:end,:len(fk)]=f['XF'][idx]
            X[start:end,len(fk):]=f[team][idx]
            y[start:end]=(lab[idx]==k).astype(np.int8)
            start=end
        by_match[match]=by_match.get(match,0)+count
        del f
    assert start==n
    X.flush();y.flush()
    info=dict(rows=n,features=width,frames_by_match=by_match,feature_names=fk+tk)
    gate.save_json(meta,info)
    return X,y,info


def train():
    import hashlib
    design=json.loads((REPORT/'design.json').read_text())
    design['pff_training_mask']='intersection of known-live native and controlled labels (same frames for label ablation)'
    design['code_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                           for name in ('gradient.py','exp_pff.py')}
    gate.save_json(REPORT/'design.json',design)
    for name in MODELS:
        dest=OUT/(name+'.txt')
        if dest.exists(): continue
        t=time.time();print('building rows',name,flush=True)
        X,y,info=build_rows(name)
        print('training',name,'rows',len(y),'features',X.shape[1],flush=True)
        dataset=lgb.Dataset(X,label=y,free_raw_data=True)
        model=lgb.train(bench.LGB_TEAM,dataset,num_boost_round=bench.ROUNDS)
        model.save_model(str(dest))
        gate.save_json(REPORT/(name+'_training.json'),dict(**info,seconds=time.time()-t,
                       model=str(dest),parameters=bench.LGB_TEAM,rounds=bench.ROUNDS))
        del X,y,dataset,model;gc.collect()
        print('trained',name,round(time.time()-t),'seconds',flush=True)


def evaluate():
    target=REPORT/'comparison.json'
    results=json.loads(target.read_text()) if target.exists() else []
    for src in gate.SOURCES:
        fs=bench.features(model_v7.VERSION,src)
        learned=bench.dead_ext('base','v3',src)
        for name in ('v7',)+MODELS:
            pp=OUT/(name+'_'+src.replace('@','-')+'_probs.pkl')
            if name=='v7': ph=bench._load(gate.prob_path(src,7,0))
            elif pp.exists(): ph=bench._load(str(pp))
            else:
                model=lgb.Booster(model_file=str(OUT/(name+'.txt')))
                ph=[bench.predict_team(model,f) for f in fs]
                bench._dump(ph,str(pp))
            decoded={}
            for conv in gate.CONVS[src]:
                labs=bench.labels(src,conv)
                for dead in ('learned','oracle'):
                    ident=dict(model=name,source=src,conv=conv,dead=dead)
                    if any(all(r[k]==v for k,v in ident.items()) for r in results): continue
                    pds=learned if dead=='learned' else [np.where(l==2,1-1e-4,1e-4) for l in labs]
                    if dead not in decoded: decoded[dead]=gate.decode_all(pds,ph)
                    sc=gate.scores(src,conv,decoded[dead])
                    row=dict(**ident,spell_by_match=sc,spellF1=float(np.mean(list(sc.values()))),
                             team_accuracy=float(e2.team_acc(ph,labs)))
                    results.append(row);gate.save_json(target,results)
                    print(name,src,conv,dead,round(row['spellF1'],5),flush=True)
        del fs;bench._blocks.clear();gc.collect()


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','train','evaluate'))
    p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(args.workers)
    else: globals()[args.stage]()
