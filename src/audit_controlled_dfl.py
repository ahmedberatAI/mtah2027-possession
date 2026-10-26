"""Verify controlled-label intervention and add paired error-rate contrasts."""
import numpy as np
import torch
import exp_controlled_dfl as e


def main():
    d=e.check();rows=e.read(e.OUT/'index.json');old=e.read(e.neural.OUT/'index.json')
    assert len(rows)==len(old)
    for a,b in zip(rows,old):
        assert all(a[k]==b[k] for k in a if k!='y')
        if a['source']=='pff':assert a['y']==b['y']
        else:e.label_stats(np.load(b['y']),np.load(a['y']))
    sel=e.read(e.REPORT/'selection.json')
    assert sel['selected']==max(sel['candidates'],key=lambda r:r['balanced_spellF1'])
    expected={s:set(d['validation'][s]) for s in ('dfl','pff')}
    for r in sel['candidates']:
        vals=[]
        for s in expected:
            assert set(r['spell_by_match'][s])==expected[s]
            vals.append(np.mean(list(r['spell_by_match'][s].values())))
        assert abs(r['balanced_spellF1']-np.mean(vals))<1e-12
    artifacts=[]
    for stage in ('internal','final'):
        sub=[r for r in rows if r['split']=='train'] if stage=='internal' else rows
        e.neural.SEED=20261024;sc=e.neural.Data(sub).scaler()
        train={s:sorted({r['match'] for r in sub if r['source']==s}) for s in ('dfl','pff')}
        configs=e.CONFIGS if stage=='internal' else [c for c in e.CONFIGS if c['name'] in {'bigru_64',sel['selected']['config']['name']}]
        for c in configs:
            for seed in e.SEEDS:
                path=e.path(c,seed,stage);a=e.read(e.REPORT/(path.stem+'_artifact.json'))
                assert a['sha256']==e.sha(path) and a['train_matches']==train
                if stage=='internal':
                    assert all(set(train[s]).isdisjoint(d['validation'][s]) for s in train)
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['epoch']==11 and ck['config']==c
                np.testing.assert_array_equal(sc.mean,ck['mean']);np.testing.assert_array_equal(sc.scale,ck['scale'])
                log=e.read(e.REPORT/(path.stem+'_training.json'))
                assert len(log['history'])==11 and all(r['validation'] is None for r in log['history'])
                artifacts.append(dict(stage=stage,config=c['name'],seed=seed,sha256=a['sha256']))
    # Native-label fixed-config control has identical preprocessing and settings.
    for seed in e.SEEDS:
        oldck=torch.load(e.trained.model_path('bigru',seed),map_location='cpu',weights_only=False)
        ck=torch.load(e.path(e.CONFIGS[0],seed,'final'),map_location='cpu',weights_only=False)
        assert ck['config']==oldck['config'] and ck['epoch']==oldck['epoch']
        np.testing.assert_array_equal(ck['mean'],oldck['mean']);np.testing.assert_array_equal(ck['scale'],oldck['scale'])
    manifest=e.read(e.REPORT/'prediction_manifest.json')
    assert manifest['selection_sha256']==e.sha(e.REPORT/'selection.json')
    rs=e.read(e.REPORT/'results.json');assert len(rs)==24
    if sel['selected']['config']['name']=='bigru_64':
        for r in rs:
            if r['model']!='controlled_selected':continue
            fixed=next(x for x in rs if x['model']=='controlled_fixed' and all(x[k]==r[k] for k in ('source','conv','dead')))
            assert r['spell_by_match']==fixed['spell_by_match']
    errors=e.read(e.REPORT/'errors.json');contrasts=[]
    mapping={'split_proxy_rate':('split_proxy','gt'),'merge_proxy_rate':('merge_proxy','pred'),
             'missed_rate':('missed','gt'),'false_discovery_rate':('false_positive','pred')}
    for r in errors:
        if r['model']=='native_ensemble':continue
        base=next(x for x in errors if x['model']=='native_ensemble' and all(x[k]==r[k] for k in ('source','conv')))
        for metric,(num,den) in mapping.items():
            a={m:c[num]/c[den] for m,c in r['counts_by_match'].items()}
            b={m:c[num]/c[den] for m,c in base['counts_by_match'].items()}
            c=e.external.paired_summary(a,b)
            c['matches_with_reduction']=c.pop('losses');c['matches_with_increase']=c.pop('wins')
            contrasts.append(dict(model=r['model'],source=r['source'],conv=r['conv'],metric=metric,
                aggregation='equal matches; negative delta is reduction',**c))
    e.save(e.REPORT/'error_contrasts.json',contrasts)
    e.save(e.REPORT/'independent_audit.json',dict(artifacts=artifacts,
        feature_paths_and_pff_labels_unchanged=True,train_validation_disjoint=True,
        scalers_recomputed_from_training_only=True,fixed_config_native_control_identical_preprocessing=True,
        selection_recomputed=True,external_cells=24,selection_sha256=e.sha(e.REPORT/'selection.json')))
    print('Verified',len(artifacts),'fits, label-only fixed-config control, scalers, split, internal selection and 24 external cells.')


if __name__=='__main__':main()
