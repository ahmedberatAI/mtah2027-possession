"""Independent checks of held-out provenance, source balance and frozen choice."""
import numpy as np
import torch
import lightgbm as lgb
import exp_hybrid as exp


def main():
    d=exp.check();models=[];means=[];scales=[]
    rows=exp.read(exp.neural.OUT/'index.json')
    tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='val']
    assert not {(r['source'],r['match']) for r in tr}&{(r['source'],r['match']) for r in va}
    exp.neural.SEED=20261024
    expected=exp.neural.Data(tr).scaler()
    for seed in exp.SEEDS:
        for kind in ('bigru','lgb_balanced'):
            p=exp.model_path(kind,seed);a=exp.read(exp.REPORT/(p.stem+'_artifact.json'))
            assert a['sha256']==exp.sha(p) and a['seed']==seed
            assert a['train']==d['train'] and a['excluded']==d['validation']
            if kind=='bigru':
                ck=torch.load(p,map_location='cpu',weights_only=False)
                assert ck['epoch']==11 and ck['config']==d['config']
                np.testing.assert_array_equal(ck['mean'],expected.mean)
                np.testing.assert_array_equal(ck['scale'],expected.scale)
                log=exp.read(exp.REPORT/(p.stem+'_training.json'))
                assert len(log['history'])==11 and all(r['validation'] is None for r in log['history'])
            else:
                model=lgb.Booster(model_file=str(p))
                assert model.num_trees()==400 and model.num_feature()==172
                for k in ('seed','data_random_seed','feature_fraction_seed','bagging_seed'):
                    assert a['parameters'][k]==seed
            models.append(dict(kind=kind,seed=seed,sha256=a['sha256']))
    X,y,source,info=exp.subset_rows()
    # Independently walk the saved raw views and confirm row counts, ordering,
    # both team views, labels and source identity at each half's endpoints.
    offset=0
    for r in tr:
        raw=np.load(r['x'],mmap_mode='r');lab=np.load(r['y'])[::2];ids=np.flatnonzero(lab<2)
        for side in (0,1):
            if len(ids):
                for j in (0,len(ids)-1):
                    np.testing.assert_array_equal(X[offset+j],raw[ids[j],side])
                    assert y[offset+j]==(lab[ids[j]]==side)
                    assert source[offset+j]==int(r['source']=='pff')
            offset+=len(ids)
    assert offset==len(y)
    weights=exp.read(exp.REPORT/'internal_lgb_weights.json')
    mass=np.array(weights['weight_sums'])
    np.testing.assert_allclose(mass/mass.sum(),[.5,.5],rtol=1e-6)
    selection=exp.read(exp.REPORT/'selection.json')
    winner=max(selection['candidates'],key=lambda r:(r['balanced_spellF1'],r['alpha']))
    assert selection['selected']==winner
    assert {r['alpha'] for r in selection['candidates']}==set(exp.GRID)
    for candidate in selection['candidates']:
        means=[]
        for s in ('dfl','pff'):
            assert set(candidate['spell_by_match'][s])==set(d['validation'][s])
            means.append(np.mean(list(candidate['spell_by_match'][s].values())))
        assert abs(candidate['balanced_spellF1']-np.mean(means))<1e-12
    result=exp.read(exp.REPORT/'results.json')
    assert len(result)==32 and len({(r['model'],r['source'],r['conv'],r['dead']) for r in result})==32
    verify=exp.read(exp.REPORT/'verification.json')
    assert verify['selection_sha256']==exp.sha(exp.REPORT/'selection.json')
    if winner['alpha'] in (0,1):
        end='bigru' if winner['alpha']==1 else 'lgb_balanced'
        for r in result:
            if r['model']!='hybrid':continue
            base=next(x for x in result if x['model']==end and all(x[k]==r[k] for k in ('source','conv','dead')))
            assert base['spell_by_match']==r['spell_by_match']
    exp.save(exp.REPORT/'independent_audit.json',dict(models=models,
        train_validation_disjoint=True,scalers_recomputed_from_train_only=True,
        lgb_row_alignment_verified=True,source_weights_equal=True,
        internal_selection_recomputed=True,external_cells=32,
        selected_alpha=winner['alpha'],selection_sha256=exp.sha(exp.REPORT/'selection.json')))
    print('Verified 6 held-out models, training-only scalers, row alignment, 50/50 weights, internal selection and 32 external cells.')


if __name__=='__main__':main()
