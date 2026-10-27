"""Fixed equal-probability BiGRU ensemble versus the actual v7 artifact.

No training or parameter search. Reuse the frozen external seed probabilities,
verify their old scores, and rescore both methods on identical inputs/labels/D.
"""
import gc
from pathlib import Path
import warnings
import numpy as np
import lightgbm as lgb
import bench
import model_v7
import exp_data_gate as gate
import exp_seed_balance as trained
import exp_barath_public as barath
from exp_ctrl2 import team_acc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cache/external_bigru_ensemble'
REPORT = ROOT / 'reports/external_bigru_ensemble'
read, save, sha = trained.read, trained.save, trained.sha


def average(members, lengths):
    if len(members) != 3 or any(len(m) != len(lengths) for m in members):
        raise ValueError('Exactly three aligned prediction lists required')
    result = []
    for i, n in enumerate(lengths):
        values = [np.asarray(m[i], dtype=np.float64) for m in members]
        if any(p.shape != (n,) or not np.isfinite(p).all() or
               not ((p >= 0) & (p <= 1)).all() for p in values):
            raise ValueError('Invalid probability values or alignment')
        result.append(np.mean(np.stack(values), axis=0, dtype=np.float64))
    return result


def paired_summary(a, b):
    assert set(a) == set(b)
    keys = sorted(a)
    delta = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(20261025)
    samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return dict(delta=float(delta.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                wins=int((delta > 1e-12).sum()), ties=int((np.abs(delta) <= 1e-12).sum()),
                losses=int((delta < -1e-12).sum()), matches=len(keys),
                delta_by_match={k: float(v) for k, v in zip(keys, delta)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    trained.check_design()
    manifest = read(barath.REPORT / 'prediction_manifest.json')
    v7 = next(j for j in manifest['predictions'] if j['kind'] == 'v7')
    assert sha(v7['path']) == v7['sha256']
    members = []
    for seed in trained.SEEDS:
        p = trained.model_path('bigru', seed)
        artifact = read(trained.REPORT / (p.stem + '_artifact.json'))
        assert sha(p) == artifact['sha256']
        j = next(j for j in manifest['predictions'] if j['kind'] == 'bigru' and j['seed'] == seed)
        assert j['sha256'] == artifact['sha256']
        members.append(dict(seed=seed, weight=1/3, model_sha256=artifact['sha256']))
    probability_paths = [trained.OUT / f'bigru_s{s}_{src.replace("@", "-")}_probs.pkl'
                         for src in gate.SOURCES for s in trained.SEEDS]
    design = dict(date='2026-10-25', members=members, v7_model=v7['path'], v7_sha256=v7['sha256'],
        operation='float64 arithmetic mean of P(home), then fixed B2',
        sources=gate.SOURCES, conventions=gate.CONVS, features=model_v7.VERSION,
        decoder=dict(pen_team=12, pen_dead=30, min_poss=8, min_dead=50, D0=175, c=48),
        dead='same fixed seven-DFL learned-D for both; oracle-D diagnostic only',
        aggregation='equal matches; average three Metrica noise seeds within match first',
        uncertainty='paired match bootstrap 10000 draws; fixed trained ensemble, no training-seed resampling',
        external_tuning=False, previous_individual_external_results_already_seen=True,
        probabilities={str(p): sha(p) for p in probability_paths},
        code={n: sha(ROOT/'src'/n) for n in ('exp_external_ensemble.py', 'bench.py',
                                            'exp_data_gate.py', 'exp_ctrl2.py', 'decode2.py')})
    # JSON-normalize tuples before comparing with a persisted design.
    import json
    design = json.loads(json.dumps(design))
    dp = REPORT/'design.json'
    if dp.exists():
        assert read(dp) == design, 'Frozen inputs changed'
    save(dp, design)
    old_rows = read(trained.REPORT/'nn_comparison.json')
    old_refs = read(trained.pff.REPORT/'comparison.json')
    rows, contrasts, checks, prediction_manifest = [], [], [], []
    for source in gate.SOURCES:
        print('Preparing', source, flush=True)
        blocks = bench.blocks(source)
        lengths = [len(b['P']) for b in blocks]
        seed_probs = [bench._load(str(trained.OUT/f'bigru_s{s}_{source.replace("@", "-")}_probs.pkl'))
                      for s in trained.SEEDS]
        ensemble = average(seed_probs, lengths)
        pp = OUT/f'ensemble_{source.replace("@", "-")}.pkl'
        bench._dump(ensemble, str(pp))
        vp = OUT/f'v7_{source.replace("@", "-")}.pkl'
        if vp.exists():
            baseline = bench._load(str(vp))
        else:
            fs = bench.features(model_v7.VERSION, source)
            mdl = lgb.Booster(model_file=v7['path'])
            baseline = [bench.predict_team(mdl, f) for f in fs]
            bench._dump(baseline, str(vp))
            del fs, mdl
            gc.collect()
        average([baseline]*3, lengths)  # same shape/range checks
        prediction_manifest.append(dict(source=source, frames=sum(lengths), blocks=len(lengths),
            ensemble_sha256=sha(pp), v7_probabilities_sha256=sha(vp)))
        save(REPORT/'prediction_manifest.json', prediction_manifest)
        labs = {c: bench.labels(source, c) for c in gate.CONVS[source]}
        first = labs[gate.CONVS[source][0]]
        assert all(len(ls) == len(lengths) and all(len(l) == n for l, n in zip(ls, lengths))
                   for ls in labs.values())
        assert all(all(np.array_equal(a == 2, b == 2) for a, b in zip(first, ls)) for ls in labs.values())
        dead = dict(learned=bench.dead_ext('base', 'v3', source),
                    oracle=[np.where(l == 2, 1-1e-4, 1e-4) for l in first])
        models = [('v7', baseline), ('bigru_ensemble', ensemble)] + [
            (f'bigru_s{s}', p) for s, p in zip(trained.SEEDS, seed_probs)]
        for mode, pd in dead.items():
            for name, ph in models:
                decoded = gate.decode_all(pd, ph)
                for conv in labs:
                    scores = gate.scores(source, conv, decoded)
                    row = dict(model=name, source=source, conv=conv, dead=mode,
                        spell_by_match=scores, spellF1=float(np.mean(list(scores.values()))),
                        team_accuracy=float(team_acc(ph, labs[conv])))
                    rows.append(row)
                    if name.startswith('bigru_s'):
                        seed = int(name.removeprefix('bigru_s'))
                        old = next(r for r in old_rows if (r['model'], r['seed'], r['source'], r['conv'], r['dead'])
                                   == ('bigru', seed, source, conv, mode))
                        assert set(scores) == set(old['spell_by_match'])
                        assert all(abs(v-old['spell_by_match'][m]) < 1e-12 for m,v in scores.items())
                        checks.append(dict(source=source, conv=conv, dead=mode, seed=seed, old_scores_reproduced=True))
                    print(source, conv, mode, name, round(row['spellF1'], 6), flush=True)
                del decoded
        for conv in labs:
            for mode in dead:
                subset = [r for r in rows if (r['source'], r['conv'], r['dead']) == (source, conv, mode)]
                ens = next(r for r in subset if r['model'] == 'bigru_ensemble')
                base = next(r for r in subset if r['model'] == 'v7')
                seed_rows = [r for r in subset if r['model'].startswith('bigru_s')]
                seed_mean = {m: float(np.mean([r['spell_by_match'][m] for r in seed_rows])) for m in ens['spell_by_match']}
                legacy = next(r for r in old_refs if (r['model'], r['source'], r['conv'], r['dead']) == ('v7',source,conv,mode))
                contrasts.append(dict(source=source, conv=conv, dead=mode,
                    v7=base['spellF1'], ensemble=ens['spellF1'], seed_score_mean=float(np.mean(list(seed_mean.values()))),
                    vs_v7=paired_summary(ens['spell_by_match'], base['spell_by_match']),
                    vs_seed_mean=paired_summary(ens['spell_by_match'], seed_mean),
                    previous_gate_v7=legacy['spellF1'], delta_v7_vs_previous_gate=base['spellF1']-legacy['spellF1']))
        save(REPORT/'results.json', rows)
        save(REPORT/'contrasts.json', contrasts)
        bench._blocks.clear()
        del blocks, seed_probs, ensemble, baseline, models, labs, dead
        gc.collect()
    assert len(rows) == 40 and len(checks) == 24 and len(contrasts) == 8
    save(REPORT/'verification.json', dict(cells=40, previous_seed_cells_reproduced=checks,
        model_hashes_verified=True, actual_barath_v7_artifact=True, exact_equal_probability_average=True,
        identical_dead_and_decoder=True, design_sha256=sha(dp)))
    lines = ['# BiGRU ensemble: external evaluation', '',
        'Fixed equal probability mean of seeds 20261025/26/27; no new training or tuning.',
        'v7 is the exact artifact used in the BAR-ATH comparison, not the older gate reference.',
        'Same test inputs, label conventions, fixed learned-D and B2 for both.',
        'Metrica: 3 matches, 3 noise seeds averaged within each match. SkillCorner: 20 matches.',
        'Paired match bootstrap 10000 draws; Metrica intervals are fragile with only three matches.', '',
        '| Test | D | v7 | Ensemble | Delta | Delta 95% CI | Wins / matches | Separate seed score mean |',
        '|---|---|---:|---:|---:|---|---|---:|']
    for c in contrasts:
        d=c['vs_v7']
        lines.append(f'| {c["source"]}/{c["conv"]} | {c["dead"]} | {c["v7"]:.6f} | {c["ensemble"]:.6f} | '
                     f'{d["delta"]:+.6f} | [{d["ci95"][0]:+.6f}, {d["ci95"][1]:+.6f}] | '
                     f'{d["wins"]}/{d["matches"]} | {c["seed_score_mean"]:.6f} |')
    lines += ['', 'Oracle-D uses true dead-ball labels and is diagnostic, not a deployable result.',
        'The external protocol uses learned-D because BAR-ATH camera-specific rule-D is unavailable here.',
        'These external sets have been used in prior experiments; this is not a new untouched holdout.',
        'The ensemble is fixed; match-bootstrap intervals do not include training-data or model-selection uncertainty.',
        'Label conventions within a source are alternative views of the same matches, not independent datasets.',
        'No inference about the private full-match BAR-ATH score is possible.', '',
        'Reproduce: `.venv/Scripts/python.exe src/exp_external_ensemble.py`']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
