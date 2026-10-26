"""Frozen BAR-ATH inference, followed by aggregate-only public scoring.

No public-label tuning, hand annotation, or hosted processing of match data.
Raw assets, features, probabilities and CSVs remain in gitignored local paths.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import warnings
import numpy as np
import lightgbm as lgb
import torch
import atd
import bench
import evaluate as ev
import model_v7
import make_submission
import exp_seed_balance as trained
import exp_seq_pff as neural
import seq_pff as seq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cache/barath_public_v1'
REPORT=ROOT/'reports/barath_public'
SUB=ROOT/'submissions/barath_public_v1'
save=trained.save
read=trained.read
sha=trained.sha


def assets():
    return [Path(atd.paths(src)[key]) for src in ('tactical','broadcast')
            for key in (('meta','raw','homography','smart_tagging') if src=='tactical'
                        else ('meta','raw','homography'))]


def pack_probs(blocks, probabilities):
    if len(blocks)!=len(probabilities):raise ValueError('Half count mismatch')
    out={}
    for b,p in zip(blocks,probabilities):
        p=np.asarray(p)
        if len(p)!=b['b']-b['a']+1:raise ValueError('Frame alignment mismatch')
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid probabilities')
        if b['a'] in out:raise ValueError('Duplicate half start')
        out[b['a']]=(None,p)
    return out


def prepare():
    missing=[str(p) for p in assets() if not p.is_file()]
    save(REPORT/'input_status.json',dict(ready=not missing,missing=missing))
    if missing:raise FileNotFoundError('BAR-ATH input assets are missing; see reports/barath_public/input_status.json')
    models=[]
    for kind in ('bigru','lgb_balanced'):
        for seed in trained.SEEDS:
            p=trained.model_path(kind,seed)
            artifact=read(trained.REPORT/(p.stem+'_artifact.json'))
            assert sha(p)==artifact['sha256']
            models.append(dict(name=f'{kind}_s{seed}',kind=kind,seed=seed,path=str(p),sha256=sha(p)))
    design=dict(date='2026-10-25',models=models,baseline='exact model_v7 team training recipe, 7 DFL natural match order',
                features=model_v7.VERSION,decoder_and_evidence=model_v7.DEFAULT,
                source_assets=[dict(name=str(p.relative_to(Path(atd.DATA_DIR))),bytes=p.stat().st_size,sha256=sha(p)) for p in assets()],
                prior_model_selection_sha256=sha(neural.REPORT/'selection.json'),
                public_tuning=False,ensemble=False,all_predictions_frozen_before_public_scoring=True,
                source_code={n:sha(ROOT/'src'/n) for n in ('exp_barath_public.py','model_v7.py','model_v5.py','model_v3.py','seq_pff.py')},
                scorer_sha256=sha(Path(ev.OFFICIAL)/'evaluation/possession_metrics.py'))
    dp=REPORT/'design.json'
    if dp.exists():assert read(dp)==design,'Inputs, models or code changed after freezing'
    save(dp,design)
    make_submission.merged_features()
    bench._barath_inputs()
    bench.features(model_v7.VERSION,'barath',procs=2)
    print('Prepared BAR-ATH assets and two feature halves; raw data kept local.',flush=True)


def predict():
    design=read(REPORT/'design.json')
    assert design['source_code']['exp_barath_public.py']==sha(Path(__file__))
    f=make_submission.merged_features();assert f['n']==atd.N_FRAMES['tactical']
    blocks=bench.blocks('barath');fs=bench.features(model_v7.VERSION,'barath',procs=2)
    # Build the real v7 baseline, not the shuffled-order gate approximation.
    base=OUT/'v7_team.txt'
    if not base.exists():
        fd=bench.features(model_v7.VERSION,'dfl@v3')
        model=bench.fit_team(fd,bench.labels('dfl@v3','native'),params={'num_threads':8})
        model.save_model(str(base));del fd,model;bench._blocks.pop('dfl@v3',None);gc.collect()
    jobs=[dict(name='v7',kind='v7',seed=None,path=str(base),sha256=sha(base))]+design['models']
    neural.init_device(20261024)
    manifest=[]
    for job in jobs:
        assert sha(job['path'])==job['sha256']
        pp=OUT/(job['name']+'_probs.pkl')
        if pp.exists():
            packed=bench._load(str(pp))
            ph=[packed[b['a']][1] for b in blocks]
        else:
            if job['kind']=='bigru':
                net,sc,_=neural.load_model(Path(job['path']))
                ph=[seq.predict_raw(net,sc,seq.views(ff),len(ff['XF'])) for ff in fs]
                del net;torch.cuda.empty_cache()
            else:
                model=lgb.Booster(model_file=job['path'])
                ph=[bench.predict_team(model,ff) for ff in fs];del model
            bench._dump(pack_probs(blocks,ph),str(pp))
        pack_probs(blocks,ph)
        labels=model_v7.predict(f,probs_path=str(pp))
        assert len(labels)==atd.N_FRAMES['tactical']
        assert not (labels[atd.graded_mask(len(labels))]==3).any()
        dest=SUB/(job['name']+'.csv')
        ev.write(labels,str(dest))
        manifest.append(dict(**job,prediction=str(dest),prediction_sha256=sha(dest),probabilities=str(pp)))
        print('Saved and validated',job['name'],'199319 rows',flush=True)
    save(REPORT/'prediction_manifest.json',dict(predictions=manifest,design_sha256=sha(REPORT/'design.json'),
         frozen_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))


def score():
    manifest=read(REPORT/'prediction_manifest.json')
    assert len(manifest['predictions'])==7
    assert manifest['design_sha256']==sha(REPORT/'design.json')
    # Public labels are first used here, after every prediction has been frozen.
    rows=[]
    for job in manifest['predictions']:
        assert sha(job['prediction'])==job['prediction_sha256']
        labels=ev.pm.label_codes(ev.pm.read_label_csv(job['prediction']))
        result=ev.score(labels)
        rows.append(dict(model=job['name'],kind=job['kind'],seed=job['seed'],**result))
        print(job['name'],ev.fmt(result),flush=True)
    baseline=next(r for r in rows if r['kind']=='v7')
    summary=[]
    for kind in ('bigru','lgb_balanced'):
        chosen=[r for r in rows if r['kind']==kind]
        a=np.array([r['spellF1'] for r in chosen])
        summary.append(dict(model=kind,mean=float(a.mean()),sd=float(a.std(ddof=1)),
             min=float(a.min()),max=float(a.max()),delta_v7=float(a.mean()-baseline['spellF1']),
             seed_scores={str(r['seed']):r['spellF1'] for r in chosen}))
    save(REPORT/'results.json',dict(baseline=baseline,individual=rows,summary=summary,
         public_gt_sha256=sha(ev.GT_PUBLIC),prediction_manifest_sha256=sha(REPORT/'prediction_manifest.json')))
    lines=['# BAR-ATH açık bölüm — 25 Ekim 2026','',
           'Sabit v7 ölü-top kanalı ve B2; yalnız takım modeli değişir. Açık bölümde ayar seçilmedi.',
           'Tüm yedi tahmin dosyası skorlama başlamadan üretildi ve hash ile donduruldu.','',
           '| Model | spellF1 | Δ v7 | macroF1 | GT possession | Tahmin possession |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["spellF1"]:.4f} | {r["spellF1"]-baseline["spellF1"]:+.4f} | '
                     f'{r["macroF1"]:.4f} | {r["n_gt"]} | {r["n_pred"]} |')
    lines+=['','| Aile | 3-seed ortalama ± SS | Δ v7 | Min–maks |','|---|---:|---:|---|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["mean"]:.4f} ± {r["sd"]:.4f} | {r["delta_v7"]:+.4f} | '
                     f'{r["min"]:.4f}–{r["max"]:.4f} |')
    lines+=['','Bu sonuç yalnız yaklaşık beş dakikalık açık GT bölümünündür; tam maç gizli skoru değildir.',
            'Üç seed aynı açık bölümü ölçer; üç bağımsız maç sayılmaz. Seed ortalaması ensemble değildir.',
            'Sabit ayarlar ve modeller: design.json; yerel CSV dosyaları: prediction_manifest.json.',
            'Ham takip, özellikler, olasılıklar ve kare etiketleri yalnız gitignored yerel klasörlerde kalır.',
            'Yeni elle etiket, public ayarı veya yarışmaya gönderim yapılmadı.','',
            '```powershell','.venv/Scripts/python.exe src/exp_barath_public.py prepare',
            '.venv/Scripts/python.exe src/exp_barath_public.py predict',
            '.venv/Scripts/python.exe src/exp_barath_public.py score','```']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    for p in (OUT,REPORT,SUB):p.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','predict','score'))
    globals()[parser.parse_args().stage]()
