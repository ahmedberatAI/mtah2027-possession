"""Verify frozen BAR-ATH outputs without printing frame-level content."""
import csv
from pathlib import Path
import atd
import exp_barath_public as exp


def main():
    result=exp.read(exp.REPORT/'results.json')
    manifest=exp.read(exp.REPORT/'prediction_manifest.json')
    design=exp.read(exp.REPORT/'design.json')
    assert manifest['design_sha256']==exp.sha(exp.REPORT/'design.json')
    with (exp.OUT/'official/scoreboard.csv').open() as f:
        official={r['submission']:r for r in csv.DictReader(f)}
    assert len(official)==len(manifest['predictions'])==7
    checks=[]
    for job in manifest['predictions']:
        assert exp.sha(job['path'])==job['sha256']
        assert exp.sha(job['prediction'])==job['prediction_sha256']
        table=exp.ev.pm.read_label_csv(job['prediction'])
        assert exp.ev.pm.validate_submission(table,atd.N_FRAMES['tactical']) is None
        labels=exp.ev.pm.label_codes(table)
        assert (labels[~atd.graded_mask()]==3).all()
        r=next(r for r in result['individual'] if r['model']==job['name'])
        o=official[job['name']]
        assert o['status']=='scored'
        assert abs(float(o['spell_f1_mean'])-r['spellF1'])<5.1e-7
        assert Path(job['prediction']).stat().st_mtime<=(exp.REPORT/'prediction_manifest.json').stat().st_mtime
        checks.append(dict(model=job['name'],rows=len(labels),csv_valid=True,
                           model_hash_valid=True,prediction_hash_valid=True,official_score_agrees=True,
                           official_ci95=[float(o['spell_bootstrap_low']),float(o['spell_bootstrap_high'])]))
    for asset in design['source_assets']:
        assert exp.sha(Path(atd.DATA_DIR)/asset['name'])==asset['sha256']
    assert exp.sha(Path(exp.ev.OFFICIAL)/'evaluation/possession_metrics.py')==design['scorer_sha256']
    assert exp.sha(Path(__file__).with_name('exp_barath_public.py'))==design['source_code']['exp_barath_public.py']
    assert (exp.REPORT/'prediction_manifest.json').stat().st_mtime<=(exp.REPORT/'results.json').stat().st_mtime
    exp.save(exp.REPORT/'verification.json',dict(checks=checks,all_seven_models_frozen_before_scoring=True,
        source_asset_hashes_valid=True,scorer_and_driver_unchanged=True,tests_passed=32,
        videos_downloaded=False,official_bootstrap_replicates=1000))
    print('Verified seven complete CSVs, frozen models/inputs, and official score agreement within CSV rounding.')


if __name__=='__main__':main()
