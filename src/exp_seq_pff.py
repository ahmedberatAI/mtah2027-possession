"""Frozen internal-match selection -> full-data refit -> external evaluation.

Never import external test labels/features until the two final fits are saved.
Run prepare, search, refit, evaluate, report in order. All large files stay in cache.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('BENCH_THREADS', '8')
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
import bench
import dfl
import gradient
import model_v7
import exp_pff as pff
import exp_data_gate as gate
import seq_pff as seq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'cache/sequence_pff_v1'
REPORT = ROOT/'reports/sequence_pff'
SEED = 20261024
EPOCHS = 12
STEPS = 192
BATCH = 16
CONFIGS = [dict(name=f'{kind}_{width}', kind=kind, width=width, drop=drop, lr=lr)
           for kind in ('tcn', 'bigru')
           for width, drop, lr in ((64, .15, .001), (96, .25, .0005))]


def save(path, value):
    gate.save_json(path, value)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def code_hashes():
    return {n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()
            for n in ('seq_pff.py', 'exp_seq_pff.py', 'gradient.py')}


def prepare():
    rng = np.random.default_rng(SEED)
    val = dict(dfl=sorted(rng.permutation(dfl.MATCHES)[:2].tolist()),
               pff=sorted(rng.permutation(gradient.matches())[:12].tolist()))
    train = dict(dfl=sorted(set(dfl.MATCHES)-set(val['dfl'])),
                 pff=sorted(set(gradient.matches())-set(val['pff'])))
    design = dict(seed=SEED, train=train, validation=val, configs=CONFIGS,
                  max_epochs=EPOCHS, steps_per_epoch=STEPS, batch=BATCH,
                  sample_rate=25/seq.STEP, core=seq.CORE, halo=seq.HALO,
                  source_balance='exact 50/50 per minibatch and masked per-window loss',
                  within_source='uniform match, then half, then known-live centre',
                  checkpoint_selection='lowest equal-source/equal-match fixed-window validation BCE',
                  architecture_selection='highest equal-source/equal-match full validation oracle-D B2 spellF1',
                  refit='all 7 DFL + 64 PFF; selected epoch count; fresh training-only scaler',
                  decoder='fixed v7 B2; no decoder or external-test tuning',
                  input='same 172 v7 features per team + missing indicators + padding-valid indicator',
                  pff_labels='controlled; same common-live mask as PFF LightGBM',
                  source_code=code_hashes())
    if (REPORT/'design.json').exists():
        old=read(REPORT/'design.json')
        assert old['train']==train and old['validation']==val, 'Do not change the split'
    save(REPORT/'design.json', design)
    rows=[]
    def materialize(source, match, half, fpath, labels):
        key=f'{source}_{match}_{half}'
        xp=OUT/'features'/(key+'.npy'); yp=OUT/'features'/(key+'_labels.npy')
        if not xp.exists():
            f=bench._load(str(fpath)); x=seq.views(f)
            assert x.shape[-1]==172
            np.save(xp, x); del f, x
        np.save(yp, labels.astype(np.int8))
        rows.append(dict(key=key,source=source,match=match,x=str(xp),y=str(yp),
                         full_length=len(labels),split='val' if match in val[source] else 'train'))
    for b in bench.blocks('dfl@v3'):
        fp=Path(bench.CACHE)/'feat'/bench._safe(model_v7.VERSION)/(b['key'].replace('/','_')+'.pkl')
        materialize('dfl', b['group'], b['key'].split('/')[-1], fp, b['lab'])
    bench._blocks.clear(); gc.collect()
    for match in gradient.matches():
        with np.load(gradient.CACHE/gradient.VERSION/f'{match}.npz') as data:
            for per in data['periods']:
                lab=data[f'h{per}_labels'].copy()
                common=(data[f'h{per}_native']<2)&(lab<2)
                lab[(lab<2)&~common]=3
                materialize('pff', match, f'h{per}', pff.paths(match,int(per)), lab)
        print('prepared',match,flush=True)
    save(OUT/'index.json', rows)
    print('prepared halves',len(rows),'split',train,val,flush=True)


class Data:
    def __init__(self, rows):
        self.rows=rows
        self.x=[np.load(r['x'],mmap_mode='r') for r in rows]
        self.full_y=[np.load(r['y']) for r in rows]
        self.y=[l[::seq.STEP] for l in self.full_y]
        self.live=[np.flatnonzero(l<2) for l in self.y]
        self.groups={s:{} for s in ('dfl','pff')}
        for i,r in enumerate(rows):
            if len(self.live[i]): self.groups[r['source']].setdefault(r['match'],[]).append(i)

    def scaler(self):
        rng=np.random.default_rng(SEED)
        sources=[]
        for source, groups in self.groups.items():
            samples=[]
            for match, indices in sorted(groups.items()):
                for i in indices:
                    idx=rng.choice(self.live[i],size=max(1,2000//len(indices)),replace=True)
                    samples.append(np.asarray(self.x[i][idx]).reshape(-1,172))
            sources.append(np.concatenate(samples))
        return seq.Scaler.fit(sources)

    def draw(self, rng, batch=BATCH):
        specs=[]
        for s,source in enumerate(('dfl','pff')):
            matches=sorted(self.groups[source])
            for j in range(batch//2):
                match=matches[int(rng.integers(len(matches)))]
                indices=self.groups[source][match];i=indices[int(rng.integers(len(indices)))]
                centre=int(rng.choice(self.live[i]));side=int(rng.integers(2))
                specs.append((i,centre-seq.CORE//2,side,s))
        return specs

    def batch(self, specs, scaler):
        xx=[];yy=[];mm=[];ss=[]
        for i,start,side,source in specs:
            xx.append(seq.window(self.x[i],start,side,scaler))
            lab=np.full(seq.CORE,3,np.int8)
            a,b=max(0,start),min(len(self.y[i]),start+seq.CORE)
            if b>a:lab[a-start:b-start]=self.y[i][a:b]
            yy.append((lab==side).astype(np.float32));mm.append((lab<2).astype(np.float32));ss.append(source)
        return tuple(torch.from_numpy(a) for a in (np.stack(xx),np.stack(yy),np.stack(mm),np.asarray(ss)))

    def validation_specs(self):
        # Same deterministic live-centred windows for every model and epoch.
        rng=np.random.default_rng(SEED+1);out=[]
        for i,r in enumerate(self.rows):
            if not len(self.live[i]): continue
            for centre in rng.choice(self.live[i],size=8,replace=True):
                for side in (0,1):out.append((i,int(centre)-seq.CORE//2,side,int(r['source']=='pff')))
        return out


def init_device(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed);np.random.seed(seed)
    assert torch.cuda.is_available(),'CUDA expected for this experiment'
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    return torch.device('cuda')


@torch.no_grad()
def validate(net, data, scaler, specs):
    net.eval();by_match={};device=next(net.parameters()).device
    for start in range(0,len(specs),BATCH):
        sub=specs[start:start+BATCH];x,y,m,_=data.batch(sub,scaler)
        pred=net(x.to(device))[:,seq.HALO:seq.HALO+seq.CORE].cpu()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,y,reduction='none')
        per=(loss*m).sum(1)/m.sum(1).clamp(min=1)
        for spec,v in zip(sub,per.numpy()):
            r=data.rows[spec[0]];by_match.setdefault(r['source'],{}).setdefault(r['match'],[]).append(float(v))
    src={s:float(np.mean([np.mean(v) for v in matches.values()])) for s,matches in by_match.items()}
    return dict(balanced=float(np.mean(list(src.values()))),by_source=src)


def train_one(config, data, scaler, epochs, dest, validation=None):
    device=init_device(SEED)
    net=seq.Net(345,config['kind'],config['width'],config['drop']).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=1e-4)
    rng=np.random.default_rng(SEED);history=[];best=np.inf;best_epoch=0
    specs=validation.validation_specs() if validation else None
    t0=time.time()
    for epoch in range(1,epochs+1):
        net.train();losses=[];t=time.time()
        for step in range(STEPS):
            x,y,m,s=data.batch(data.draw(rng),scaler)
            x,y,m,s=(a.to(device) for a in (x,y,m,s))
            opt.zero_grad(set_to_none=True)
            pred=net(x)[:,seq.HALO:seq.HALO+seq.CORE]
            loss=seq.balanced_loss(pred,y,m,s)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();losses.append(float(loss.detach()))
        v=validate(net,validation,scaler,specs) if validation else None
        score=v['balanced'] if v else 0.
        row=dict(epoch=epoch,train_bce=float(np.mean(losses)),validation=v,seconds=time.time()-t)
        history.append(row)
        if score<best or validation is None:
            best=score;best_epoch=epoch
            torch.save(dict(state=net.state_dict(),config=config,mean=scaler.mean,scale=scaler.scale,
                            epoch=epoch,step=seq.STEP,core=seq.CORE,halo=seq.HALO),dest)
        save(REPORT/(dest.stem+'_training.json'),dict(config=config,history=history,
             selected_epoch=best_epoch,seconds=time.time()-t0,source_code=code_hashes()))
        print(dest.stem,'epoch',epoch,'train',round(row['train_bce'],5),
              'validation',v,'seconds',round(row['seconds']),flush=True)
    del net,opt;torch.cuda.empty_cache();gc.collect()


def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    c=ck['config'];net=seq.Net(345,c['kind'],c['width'],c['drop']).cuda()
    net.load_state_dict(ck['state']);net.eval()
    return net,seq.Scaler(ck['mean'],ck['scale']),ck


def internal_score(path, data):
    net,sc,ck=load_model(path)
    ph=[seq.predict_raw(net,sc,x,r['full_length']) for x,r in zip(data.x,data.rows)]
    pds=[np.where(y==2,1-1e-4,1e-4) for y in data.full_y]
    preds=gate.decode_all(pds,ph)
    bysource={}
    for source in ('dfl','pff'):
        idx=[i for i,r in enumerate(data.rows) if r['source']==source]
        scores=bench.spell_by_group([dict(group=data.rows[i]['match']) for i in idx],
                                    [data.full_y[i] for i in idx],[preds[i] for i in idx])
        bysource[source]={m:float(v[0]) for (m,_),v in scores.items()}
    score=float(np.mean([np.mean(list(s.values())) for s in bysource.values()]))
    result=dict(config=ck['config'],epoch=ck['epoch'],balanced_spellF1=score,by_source=bysource)
    del net;torch.cuda.empty_cache()
    return result


def search():
    rows=read(OUT/'index.json');init_device(SEED)
    train=Data([r for r in rows if r['split']=='train']);val=Data([r for r in rows if r['split']=='val'])
    assert not {(r['source'],r['match']) for r in train.rows}&{(r['source'],r['match']) for r in val.rows}
    scaler=train.scaler();results=[]
    for config in CONFIGS:
        path=OUT/(config['name']+'_search.pt');report=REPORT/(config['name']+'_internal.json')
        log=REPORT/(path.stem+'_training.json')
        complete=path.exists() and log.exists() and len(read(log)['history'])==EPOCHS
        if not complete:train_one(config,train,scaler,EPOCHS,path,val)
        if report.exists():result=read(report)
        else:
            result=internal_score(path,val);save(report,result)
        results.append(result);print('internal result',result,flush=True)
    selected={kind:max((r for r in results if r['config']['kind']==kind),key=lambda r:r['balanced_spellF1'])
              for kind in ('tcn','bigru')}
    save(REPORT/'selection.json',dict(selected=selected,candidates=results,external_data_used=False,
         frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_code=code_hashes()))


def refit():
    selected=read(REPORT/'selection.json')['selected']
    data=Data(read(OUT/'index.json'));scaler=data.scaler()
    for kind,result in selected.items():
        path=OUT/(kind+'_final.pt')
        log=REPORT/(path.stem+'_training.json')
        complete=path.exists() and log.exists() and len(read(log)['history'])==result['epoch']
        if not complete:train_one(result['config'],data,scaler,result['epoch'],path)
    save(REPORT/'final_manifest.json',dict(models={kind:dict(path=str(OUT/(kind+'_final.pt')),
         sha256=hashlib.sha256((OUT/(kind+'_final.pt')).read_bytes()).hexdigest()) for kind in selected},
         selection_sha256=hashlib.sha256((REPORT/'selection.json').read_bytes()).hexdigest(),
         source_code=code_hashes(),torch=torch.__version__,cuda=torch.version.cuda,
         device=torch.cuda.get_device_name(0),train_matches={'dfl':7,'pff':64}))


def evaluate():
    # The manifest can only be created after internal selection and full refit.
    manifest=read(REPORT/'final_manifest.json');init_device(SEED)
    assert manifest['selection_sha256']==hashlib.sha256((REPORT/'selection.json').read_bytes()).hexdigest()
    path=REPORT/'comparison.json'
    rows=read(path) if path.exists() else [r for r in read(pff.REPORT/'comparison.json')
                                          if r['model'] in ('v7','dfl_pff_controlled')]
    for source in gate.SOURCES:
        feats=bench.features(model_v7.VERSION,source)
        pd=bench.dead_ext('base','v3',source)
        for kind in ('tcn','bigru'):
            pp=OUT/(kind+'_'+source.replace('@','-')+'_probs.pkl')
            if pp.exists():ph=bench._load(str(pp))
            else:
                net,sc,_=load_model(OUT/(kind+'_final.pt'))
                ph=[seq.predict_raw(net,sc,seq.views(f),len(f['XF'])) for f in feats]
                bench._dump(ph,str(pp));del net;torch.cuda.empty_cache()
            for conv in gate.CONVS[source]:
                labs=bench.labels(source,conv)
                for dead in ('learned','oracle'):
                    ident=dict(model=kind,source=source,conv=conv,dead=dead)
                    if any(all(r[k]==v for k,v in ident.items()) for r in rows):continue
                    pds=pd if dead=='learned' else [np.where(l==2,1-1e-4,1e-4) for l in labs]
                    preds=gate.decode_all(pds,ph);scs=gate.scores(source,conv,preds)
                    from exp_ctrl2 import team_acc
                    row=dict(**ident,spell_by_match=scs,spellF1=float(np.mean(list(scs.values()))),
                             team_accuracy=float(team_acc(ph,labs)))
                    rows.append(row);save(path,rows);print(ident,row['spellF1'],flush=True)
        del feats;bench._blocks.clear();gc.collect()


def report():
    from report_pff import ci
    rows=read(REPORT/'comparison.json');assert len(rows)==32
    result=[]
    lines=['# TCN / BiGRU — dış değerlendirme','',
           'Ayar seçimi yalnız ayrılmış DFL/PFF maçlarında; son eğitim tüm 7+64 maçta.',
           'DFL/PFF 50:50; aynı v7 özellik ailesi ve sabit B2. Learned-D modeli sabit.', '',
           '| Model | Test | D | spellF1 | Δ v7 | %95 GA | Δ PFF LightGBM | %95 GA |',
           '|---|---|---|---:|---:|---|---:|---|']
    for r in rows:
        if r['model'] not in ('tcn','bigru'):continue
        refs={name:next(x for x in rows if x['model']==name and
              all(x[k]==r[k] for k in ('source','conv','dead'))) for name in ('v7','dfl_pff_controlled')}
        diffs={}
        for name,ref in refs.items():
            ds={m:r['spell_by_match'][m]-ref['spell_by_match'][m] for m in r['spell_by_match']}
            diffs[name]=dict(delta=float(np.mean(list(ds.values()))),ci=ci(list(ds.values())),
                             positive_matches=sum(v>0 for v in ds.values()),by_match=ds)
        result.append(dict(**r,comparisons=diffs))
        a,b=diffs.values()
        lines.append(f'| {r["model"]} | {r["source"]}/{r["conv"]} | {r["dead"]} | {r["spellF1"]:.4f} | '
                     f'{a["delta"]:+.4f} | [{a["ci"][0]:+.4f},{a["ci"][1]:+.4f}] | '
                     f'{b["delta"]:+.4f} | [{b["ci"][0]:+.4f},{b["ci"][1]:+.4f}] |')
    lines+=['','Maç düzeyinde eşli bootstrap; Metrica tohumları maç içinde ortalanır (3 maç, 9 değil).',
            'Aralıklar tek eğitim seedinin maç belirsizliğidir; eğitim belirsizliğini kapsamaz.',
            'Özel BAR-ATH skoru ölçülmedi. PFF kontrollü etiketleri muhafazakâr vekillerdir.']
    save(REPORT/'summary.json',result)
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'features').mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True,parents=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','search','refit','evaluate','report'))
    globals()[p.parse_args().stage]()
