"""User-requested equal arithmetic probability mean, then unchanged v7/B2.

Frame-level assets stay local. Build/freeze the full CSV before scoring it.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import atd
import bench
import evaluate as ev
import make_submission
import model_v7
import exp_barath_public as previous

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cache/barath_bigru_ensemble'
REPORT=ROOT/'reports/barath_bigru_ensemble'
SUB=ROOT/'submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv'
SEEDS=(20261025,20261026,20261027)
read,save,sha=previous.read,previous.save,previous.sha


def equal_probability_mean(members,halves):
    if len(members)!=3:raise ValueError('Exactly three seed members required')
    keys={a for _,a,_ in halves}
    if any(set(m)!=keys for m in members):raise ValueError('Half keys differ')
    out={}
    for _,a,b in halves:
        values=[]
        for m in members:
            pd,ph=m[a]
            if pd is not None:raise ValueError('Only team probabilities may change')
            ph=np.asarray(ph,dtype=np.float64)
            if ph.shape!=(b-a+1,):raise ValueError('Half length differs')
            if not np.isfinite(ph).all() or not ((ph>=0)&(ph<=1)).all():
                raise ValueError('Invalid member probabilities')
            values.append(ph)
        out[a]=(None,np.mean(np.stack(values),axis=0,dtype=np.float64))
    return out


def build():
    old=read(previous.REPORT/'prediction_manifest.json')
    prior_design=read(previous.REPORT/'design.json')
    assert old['design_sha256']==sha(previous.REPORT/'design.json')
    assert prior_design['decoder_and_evidence']==model_v7.DEFAULT
    for name in ('model_v7.py','model_v5.py','model_v3.py'):
        assert prior_design['source_code'][name]==sha(ROOT/'src'/name)
    jobs=[next(j for j in old['predictions'] if j['kind']=='bigru' and j['seed']==s) for s in SEEDS]
    members=[];metadata=[]
    f=make_submission.merged_features()
    for j in jobs:
        assert sha(j['path'])==j['sha256'] and sha(j['prediction'])==j['prediction_sha256']
        # Check cached probabilities still reproduce the already frozen seed
        # prediction; this does not read or use public ground truth.
        regenerated=model_v7.predict(f,probs_path=j['probabilities'])
        frozen=ev.pm.label_codes(ev.pm.read_label_csv(j['prediction']))
        assert np.array_equal(regenerated,frozen),'Cached seed probability provenance mismatch'
        members.append(bench._load(j['probabilities']))
        metadata.append(dict(seed=j['seed'],weight=1/3,model_sha256=j['sha256'],
                             probabilities=j['probabilities'],probabilities_sha256=sha(j['probabilities']),
                             source_csv_sha256=j['prediction_sha256']))
    design=dict(date='2026-10-25',members=metadata,operation='arithmetic mean of P(home) at each frame before decoding',
                decoder_and_evidence=model_v7.DEFAULT,public_weight_search=False,public_scores_previously_seen=True,
                code_sha256=sha(Path(__file__)),merged_features_sha256=sha(ROOT/'cache/features_merged.pkl'),
                original_prediction_manifest_sha256=sha(previous.REPORT/'prediction_manifest.json'))
    dp=REPORT/'design.json'
    if dp.exists():assert read(dp)==design,'Frozen ensemble design differs'
    save(dp,design)
    mean=equal_probability_mean(members,atd.HALVES)
    pp=OUT/'equal_probabilities.pkl';bench._dump(mean,str(pp))
    labels=model_v7.predict(f,probs_path=str(pp))
    assert len(labels)==atd.N_FRAMES['tactical']
    assert (labels[~atd.graded_mask()]==3).all() and (labels[atd.graded_mask()]<3).all()
    ev.write(labels,str(SUB))
    save(REPORT/'prediction_manifest.json',dict(csv=str(SUB),rows=len(labels),csv_sha256=sha(SUB),
         probabilities=str(pp),probabilities_sha256=sha(pp),design_sha256=sha(dp),
         frozen_at=datetime.now(timezone.utc).isoformat(),source_predictions_reproduced=True))
    print('Built and validated equal-weight ensemble: 199319 rows; source seed predictions reproduced.',flush=True)


def score():
    manifest=read(REPORT/'prediction_manifest.json')
    assert manifest['csv_sha256']==sha(SUB) and manifest['design_sha256']==sha(REPORT/'design.json')
    labels=ev.pm.label_codes(ev.pm.read_label_csv(str(SUB)))
    result=ev.score(labels)
    old=read(previous.REPORT/'results.json')
    seed_summary=next(r for r in old['summary'] if r['model']=='bigru')
    result.update(delta_v7=result['spellF1']-old['baseline']['spellF1'],
                  delta_seed_mean=result['spellF1']-seed_summary['mean'],
                  delta_best_seed=result['spellF1']-seed_summary['max'],
                  public_gt_sha256=sha(ev.GT_PUBLIC),csv_sha256=sha(SUB))
    save(REPORT/'results.json',result)
    print('Ensemble',ev.fmt({k:v for k,v in result.items() if 'sha256' not in k}),flush=True)
    lines=['# Eşit ağırlıklı BiGRU ensemble — 25 Ekim 2026','',
           'Üç seed (20261025/26/27) için kare bazındaki P(home) olasılıklarının aritmetik ortalaması alındı.',
           'Ağırlıkların her biri 1/3. Aynı v7 ölü-top kanalı ve B2; ağırlık/dekoder araması yapılmadı.',
           'Kaynak olasılıklar eski seed CSV’lerini yeniden üretti. Ensemble CSV’si skorlamadan önce donduruldu.','',
           '| Model / özet | Açık bölüm spellF1 |','|---|---:|',
           f'| v7 | {old["baseline"]["spellF1"]:.4f} |',
           f'| Üç ayrı BiGRU skorunun ortalaması | {seed_summary["mean"]:.4f} |',
           f'| En yüksek ayrı seed (karşılaştırma amaçlı) | {seed_summary["max"]:.4f} |',
           f'| **Eşit ağırlıklı olasılık ensemble** | **{result["spellF1"]:.4f}** |','',
           f'Ensemble − v7: **{result["delta_v7"]:+.4f}**; ensemble − ayrı seed skor ortalaması: '
           f'**{result["delta_seed_mean"]:+.4f}**. macroF1: **{result["macroF1"]:.4f}**.',
           f'Açık bölümde GT possession: {result["n_gt"]}; tahmin: {result["n_pred"]}.','',
           'Tam-maç dosyası: `submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv`.',
           '199.319 veri satırı, sıfır tabanlı frame indeksi ve frame,label başlığı. Kare düzeyi içerik gitignored yereldir.',
           'Bu skor yalnız yaklaşık beş dakikalık açık GT bölümünündür; tam maçın gizli skoru bilinmiyor.',
           'Önceki public sonuçlar zaten görülmüştü. Bu deney kullanıcının istediği sabit 1/3 ağırlıklı birleşimi ölçer.',
           'Submission dosyası hazırlanmıştır; yarışma platformuna yüklenmemiştir.','',
           '```powershell','.venv/Scripts/python.exe src/exp_barath_ensemble.py build',
           '.venv/Scripts/python.exe src/exp_barath_ensemble.py score',
           '.venv/Scripts/python.exe external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/barath_bigru_ensemble/bigru_ensemble_equal.csv',
           '.venv/Scripts/python.exe external/more-than-a-hack-2027/evaluation/run_evaluation.py --gt public --results submissions/barath_bigru_ensemble --out cache/barath_bigru_ensemble/official --bootstrap 1000 --no-detail','```']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    for path in (OUT,REPORT,SUB.parent):path.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('build','score'))
    globals()[p.parse_args().stage]()
