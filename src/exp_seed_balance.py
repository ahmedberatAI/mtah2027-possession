"""Three fresh seeds, paired source-balanced/unbalanced LightGBM controls.

Freeze previous internal-selected neural settings; no test-set selection.
Separate NN/LGB evaluation files allow independent CPU/GPU work without races.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import time
import warnings
import numpy as np
import lightgbm as lgb
import torch
import bench
import dfl
import model_v7
import exp_pff as pff
import exp_seq_pff as neural
import seq_pff as seq
import exp_data_gate as gate
from exp_ctrl2 import team_acc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cache/seed_balance_v1'
REPORT=ROOT/'reports/seed_balance'
SEEDS=(20261025,20261026,20261027)
KINDS=('lgb_unbalanced','lgb_balanced','tcn','bigru')
NN_REPORT=neural.REPORT
read=neural.read
save=gate.save_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {n:sha(ROOT/'src'/n) for n in ('exp_seed_balance.py','exp_seq_pff.py','seq_pff.py','exp_pff.py')}


def source_weights(source):
    source=np.asarray(source)
    counts=np.bincount(source,minlength=2)
    if len(counts)!=2 or (counts==0).any():raise ValueError('Exactly two nonempty sources required')
    return (len(source)/(2*counts[source])).astype(np.float32)


def model_path(kind,seed):
    return OUT/f'{kind}_s{seed}.{("pt" if kind in ("tcn","bigru") else "txt")}'


def prepare():
    selected=read(NN_REPORT/'selection.json')
    design=dict(seeds=SEEDS,neural_selection=selected['selected'],
         selection_sha256=sha(NN_REPORT/'selection.json'),
         neural_fixed=dict(epochs=11,steps=neural.STEPS,batch=neural.BATCH,
                           input_hz=12.5,output_hz=25,scaler_seed=20261024),
         seed_effects=dict(neural='initialization, dropout, training-window draws; scaler fixed',
                          lightgbm='seed, data_random_seed, feature_fraction_seed, bagging_seed'),
         lightgbm_parameters=bench.LGB_TEAM,lightgbm_rounds=400,
         lightgbm_weighting=dict(lgb_unbalanced='one per row',
             lgb_balanced='N/(2*N_source) per row; exact 50/50 source total and mean weight one'),
         controlled_contrast='paired LGB weights only; same rows, feature order, parameters and seed',
         all_training_matches=dict(dfl=7,pff=64),external_tuning=False,
         decoder='fixed B2 pen_team12; fixed learned-D and oracle-D diagnostics',
         previous_single_seed_not_in_new_mean=True,source_code=hashes())
    path=REPORT/'design.json'
    if path.exists():assert read(path)==design,'Frozen design changed'
    save(path,design)


def check_design():
    design=read(REPORT/'design.json')
    assert design['selection_sha256']==sha(NN_REPORT/'selection.json')
    assert design['source_code']==hashes(),'Code changed after freezing design'
    return design


def train_nn():
    design=check_design()
    data=neural.Data(read(neural.OUT/'index.json'))
    assert neural.SEED==20261024
    scaler=data.scaler()  # fixed preprocessing across seeds and architectures
    neural.REPORT=REPORT
    for seed in SEEDS:
        neural.SEED=seed
        for kind in ('tcn','bigru'):
            chosen=design['neural_selection'][kind];dest=model_path(kind,seed)
            done=REPORT/(dest.stem+'_artifact.json')
            if done.exists():
                assert sha(dest)==read(done)['sha256'];continue
            neural.train_one(chosen['config'],data,scaler,chosen['epoch'],dest)
            save(done,dict(kind=kind,seed=seed,path=str(dest),sha256=sha(dest),
                           epochs=chosen['epoch'],config=chosen['config'],source_code=hashes()))
    print('All six neural fits complete',flush=True)


def training_sources(info):
    # Reconstruct exact block/side order from the existing row builder and
    # verify counts against its persisted metadata before assigning weights.
    pieces=[];by_match={}
    for _,lab,match in pff.training_items('dfl_pff_controlled'):
        count=int((lab[::bench.STRIDE]<2).sum())
        pieces.append(np.full(2*count,0 if match in dfl.MATCHES else 1,np.int8))
        by_match[match]=by_match.get(match,0)+count
    source=np.concatenate(pieces)
    assert by_match==info['frames_by_match'] and len(source)==info['rows']
    return source


def train_lgb():
    check_design()
    X,y,info=pff.build_rows('dfl_pff_controlled')
    source=training_sources(info);weight=source_weights(source)
    save(REPORT/'weights.json',dict(rows=len(y),features=X.shape[1],
         rows_by_source=np.bincount(source).tolist(),weight_by_source=[float(weight[source==s][0]) for s in (0,1)],
         weight_sum_by_source=[float(weight[source==s].sum(dtype=np.float64)) for s in (0,1)],
         mean_weight=float(weight.mean(dtype=np.float64)),source_names=['DFL','PFF'],
         source_order_checked=True,feature_names=info['feature_names']))
    for seed in SEEDS:
        params=dict(bench.LGB_TEAM,seed=seed,data_random_seed=seed,
                    feature_fraction_seed=seed,bagging_seed=seed)
        for kind in ('lgb_unbalanced','lgb_balanced'):
            dest=model_path(kind,seed);done=REPORT/(dest.stem+'_artifact.json')
            if done.exists():
                assert sha(dest)==read(done)['sha256'];continue
            t=time.time();print('training',kind,seed,flush=True)
            dataset=lgb.Dataset(X,label=y,weight=weight if kind=='lgb_balanced' else None,free_raw_data=True)
            model=lgb.train(params,dataset,num_boost_round=400)
            model.save_model(str(dest))
            save(done,dict(kind=kind,seed=seed,path=str(dest),sha256=sha(dest),rounds=400,
                           parameters=params,seconds=time.time()-t,source_code=hashes()))
            print('trained',kind,seed,round(time.time()-t),'seconds',flush=True)
            del model,dataset;gc.collect()


def evaluate(family):
    check_design()
    kinds=('tcn','bigru') if family=='nn' else ('lgb_unbalanced','lgb_balanced')
    for kind in kinds:
        for seed in SEEDS:
            artifact=read(REPORT/(model_path(kind,seed).stem+'_artifact.json'))
            assert sha(model_path(kind,seed))==artifact['sha256']
    if family=='nn':neural.init_device(20261024)
    target=REPORT/(family+'_comparison.json');results=read(target) if target.exists() else []
    for source in gate.SOURCES:
        fs=bench.features(model_v7.VERSION,source)
        labs={c:bench.labels(source,c) for c in gate.CONVS[source]}
        first=labs[gate.CONVS[source][0]]
        assert all(all(np.array_equal(a==2,b==2) for a,b in zip(first,ls)) for ls in labs.values())
        dead=dict(learned=bench.dead_ext('base','v3',source),
                  oracle=[np.where(l==2,1-1e-4,1e-4) for l in first])
        for kind in kinds:
            for seed in SEEDS:
                pp=OUT/f'{kind}_s{seed}_{source.replace("@","-")}_probs.pkl'
                if pp.exists():ph=bench._load(str(pp))
                else:
                    if family=='nn':
                        net,sc,_=neural.load_model(model_path(kind,seed))
                        ph=[seq.predict_raw(net,sc,seq.views(f),len(f['XF'])) for f in fs]
                        del net;torch.cuda.empty_cache()
                    else:
                        model=lgb.Booster(model_file=str(model_path(kind,seed)))
                        ph=[bench.predict_team(model,f) for f in fs];del model
                    assert all(len(p)==len(f['XF']) and np.isfinite(p).all() for p,f in zip(ph,fs))
                    bench._dump(ph,str(pp))
                for mode,pds in dead.items():
                    pending=[c for c in labs if not any(r['model']==kind and r['seed']==seed and
                             r['source']==source and r['conv']==c and r['dead']==mode for r in results)]
                    if not pending:continue
                    decoded=gate.decode_all(pds,ph)
                    for conv in pending:
                        scores=gate.scores(source,conv,decoded)
                        row=dict(model=kind,seed=seed,source=source,conv=conv,dead=mode,
                                 spell_by_match=scores,spellF1=float(np.mean(list(scores.values()))),
                                 team_accuracy=float(team_acc(ph,labs[conv])))
                        results.append(row);save(target,results)
                        print(kind,seed,source,conv,mode,round(row['spellF1'],5),flush=True)
        del fs;bench._blocks.clear();gc.collect()
    assert len(results)==48


def crossed_ci(matrix,replicates=10000):
    """Resample seed and match axes, preserving paired model and match identity.

    Matches are drawn once per bootstrap, shared by all drawn seeds. Do not
    treat the same match under three fits as three independent matches.
    """
    a=np.asarray(matrix,float)
    if a.ndim!=2:raise ValueError('Expected seed x match matrix')
    rng=np.random.default_rng(20261028)
    si=rng.integers(0,a.shape[0],size=(replicates,a.shape[0]))
    mi=rng.integers(0,a.shape[1],size=(replicates,a.shape[1]))
    value=a[si[:,:,None],mi[:,None,:]].mean(axis=(1,2))
    return np.quantile(value,[.025,.975]).tolist()


def report():
    rows=read(REPORT/'nn_comparison.json')+read(REPORT/'lgb_comparison.json')
    assert len(rows)==96 and len({(r['model'],r['seed'],r['source'],r['conv'],r['dead']) for r in rows})==96
    refs=[r for r in read(pff.REPORT/'comparison.json') if r['model'] in ('v7','dfl_pff_controlled')]
    save(REPORT/'comparison.json',rows+refs)
    summary=[];contrasts=[]
    lines=['# Üç seed ve kaynak dengesi — tüm sonuçlar','',
           'Skorlar seed başına maç ortalaması spellF1; ± örnek standart sapma (ddof=1).',
           'Yeni seedler: 20261025, 20261026, 20261027. Önceki tek seed bu ortalamaya girmez.','',
           '| Test | D | Model | Ortalama ± SS | Min–maks | Seed skorları |',
           '|---|---|---|---|---|---|']
    def matrix(kind,source,conv,dead):
        matches=sorted(next(r for r in rows if r['source']==source)['spell_by_match'])
        if kind in ('v7','dfl_pff_controlled'):
            r=next(r for r in refs if (r['model'],r['source'],r['conv'],r['dead'])==(kind,source,conv,dead))
            return np.tile([r['spell_by_match'][m] for m in matches],(3,1)),matches
        selected={r['seed']:r for r in rows if (r['model'],r['source'],r['conv'],r['dead'])==(kind,source,conv,dead)}
        assert set(selected)==set(SEEDS)
        return np.array([[selected[s]['spell_by_match'][m] for m in matches] for s in SEEDS]),matches
    pairs=(('lgb_balanced','lgb_unbalanced'),('tcn','lgb_balanced'),('bigru','lgb_balanced'),
           ('tcn','lgb_unbalanced'),('bigru','lgb_unbalanced'),('tcn','v7'),('bigru','v7'),('bigru','tcn'))
    for source in gate.SOURCES:
        for conv in gate.CONVS[source]:
            for dead in ('learned','oracle'):
                for kind in KINDS:
                    a,matches=matrix(kind,source,conv,dead);seed_scores=a.mean(1)
                    item=dict(model=kind,source=source,conv=conv,dead=dead,
                         mean=float(seed_scores.mean()),sd=float(seed_scores.std(ddof=1)),
                         min=float(seed_scores.min()),max=float(seed_scores.max()),
                         scores_by_seed=dict(zip(map(str,SEEDS),seed_scores.tolist())),
                         match_scores_by_seed={str(s):dict(zip(matches,a[i].tolist())) for i,s in enumerate(SEEDS)})
                    summary.append(item)
                    lines.append(f'| {source}/{conv} | {dead} | {kind} | {item["mean"]:.4f} ± {item["sd"]:.4f} | '
                         f'{item["min"]:.4f}–{item["max"]:.4f} | '+', '.join(f'{v:.4f}' for v in seed_scores)+' |')
                for kind,base in pairs:
                    a,matches=matrix(kind,source,conv,dead);b,_=matrix(base,source,conv,dead);delta=a-b
                    contrasts.append(dict(model=kind,baseline=base,source=source,conv=conv,dead=dead,
                         delta=float(delta.mean()),seed_deltas=delta.mean(1).tolist(),
                         crossed_seed_match_ci95=crossed_ci(delta),
                         positive_matches_seed_mean=int((delta.mean(0)>0).sum()),
                         positive_matches_each_seed=(delta>0).sum(1).tolist(),matches=len(matches)))
    lines+=['','## Eşli farklar','',
            '| Test | D | Fark | Ortalama Δ | Seed+maç %95 aralık | Artan maç (seed ort.) |',
            '|---|---|---|---:|---|---:|']
    for c in contrasts:
        lo,hi=c['crossed_seed_match_ci95']
        lines.append(f'| {c["source"]}/{c["conv"]} | {c["dead"]} | {c["model"]} − {c["baseline"]} | '
                     f'{c["delta"]:+.4f} | [{lo:+.4f},{hi:+.4f}] | {c["positive_matches_seed_mean"]}/{c["matches"]} |')
    lines+=['','Aralıklar seed ve maç eksenlerini birlikte örnekleyen 10.000 bootstrap tekrarından.',
            'Yalnız üç seed vardır; aralıklar keşifseldir. Metrica 3, SkillCorner 20 bağımsız maçtır.',
            'Metrica bozulma tohumları önce maç içinde ortalanır. Seed skorlarının ortalaması ensemble değildir.',
            'Dengeli/normal LGB çifti yalnız kaynak ağırlığını değiştirir. Sinir ağı karşılaştırması',
            'normalizasyon, pencere örnekleme, örnekleme hızı ve mimari farklarını birlikte içerir.']
    save(REPORT/'summary.json',summary);save(REPORT/'contrasts.json',contrasts)
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Complete: 96 new evaluation cells, 16 fixed reference cells, 64 contrasts',flush=True)


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train_nn','train_lgb','evaluate','report'))
    parser.add_argument('--family',choices=('nn','lgb'))
    args=parser.parse_args()
    if args.stage=='evaluate':
        assert args.family is not None
        evaluate(args.family)
    else:globals()[args.stage]()
