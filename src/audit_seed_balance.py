"""Audit model seeds, fixed controls, predictions and additive contrasts."""
import hashlib
import numpy as np
import torch
import lightgbm as lgb
import bench
import exp_seed_balance as exp


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    design=exp.check_design();models=[];means=[];scales=[];state_hashes={k:[] for k in ('tcn','bigru')}
    for seed in exp.SEEDS:
        params=[]
        for kind in exp.KINDS:
            path=exp.model_path(kind,seed);a=exp.read(exp.REPORT/(path.stem+'_artifact.json'))
            assert a['seed']==seed and a['kind']==kind and exp.sha(path)==a['sha256']
            if kind in ('tcn','bigru'):
                c=torch.load(path,map_location='cpu',weights_only=False)
                selected=design['neural_selection'][kind]
                assert c['epoch']==selected['epoch'] and c['config']==selected['config']
                state_hashes[kind].append(state_hash(c['state']))
                means.append(c['mean']);scales.append(c['scale'])
                history=exp.read(exp.REPORT/(path.stem+'_training.json'))['history']
                assert len(history)==11 and all(x['validation'] is None for x in history)
            else:
                model=lgb.Booster(model_file=str(path));assert model.num_trees()==400 and model.num_feature()==172
                for key in ('seed','data_random_seed','feature_fraction_seed','bagging_seed'):
                    assert a['parameters'][key]==seed
                params.append(a['parameters'])
            models.append(dict(model=kind,seed=seed,sha256=a['sha256']))
        assert params[0]==params[1],'Paired LGB parameters differ'
    assert all(len(set(v))==3 for v in state_hashes.values())
    assert all(np.array_equal(means[0],v) for v in means)
    assert all(np.array_equal(scales[0],v) for v in scales)
    weights=exp.read(exp.REPORT/'weights.json')
    mass=np.array(weights['weight_sum_by_source']);assert np.max(np.abs(mass/mass.sum()-.5))<1e-7
    assert abs(weights['mean_weight']-1)<1e-7
    predictions=[]
    for source in exp.gate.SOURCES:
        blocks=bench.blocks(source);lengths=[len(b['P']) for b in blocks]
        for kind in exp.KINDS:
            fingerprints=[]
            for seed in exp.SEEDS:
                pp=exp.OUT/f'{kind}_s{seed}_{source.replace("@","-")}_probs.pkl'
                probs=bench._load(str(pp));assert len(probs)==len(lengths)
                assert all(len(p)==n and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p,n in zip(probs,lengths))
                h=hashlib.sha256()
                for p in probs:h.update(p.tobytes())
                fingerprints.append(h.hexdigest())
            assert len(set(fingerprints))==3,'Seed predictions were duplicated'
            predictions.append(dict(model=kind,source=source,frames=sum(lengths),distinct_seed_predictions=True))
        bench._blocks.clear()
    rows=exp.read(exp.REPORT/'comparison.json');assert len(rows)==112
    contrasts=exp.read(exp.REPORT/'contrasts.json')
    for c in contrasts:
        if c['model'] not in ('tcn','bigru') or c['baseline']!='lgb_unbalanced':continue
        others=[x for x in contrasts if all(x[k]==c[k] for k in ('source','conv','dead'))]
        balance=next(x['delta'] for x in others if x['model']=='lgb_balanced' and x['baseline']=='lgb_unbalanced')
        residual=next(x['delta'] for x in others if x['model']==c['model'] and x['baseline']=='lgb_balanced')
        assert abs(c['delta']-balance-residual)<1e-12
    exp.save(exp.REPORT/'verification.json',dict(models=models,predictions=predictions,
        source_weight_equal=True,paired_lgb_parameters_identical=True,fixed_scaler_across_seeds=True,
        neural_state_hashes=state_hashes,all_seeds_distinct=True,evaluation_cells=112,
        contrast_additivity=True,code_and_selection_frozen=True))
    print('Verified 12 models, distinct seeds, fixed scaler/controls, 24 prediction caches, 112 cells and additive contrasts.')


if __name__=='__main__':main()
