"""Fixed-protocol 2/4/7-match learning curve; no BAR-ATH data or tuning.

Seven cyclic, nested subsets balance inclusion of each DFL match. Same v7
features, native labels, V3 noise, 400 trees and B2 decoder for every fit.
Dead model always uses seven DFL matches, isolating TEAM training size.
Metrica noise seeds are averaged within game before match-level summaries.
Run prepare, train, evaluate in order. All frame-level outputs stay in cache.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import lightgbm as lgb
import bench
import dfl
import model_v7
import exp_ctrl2 as e2

ROOT = HERE.parent
OUT = ROOT / 'cache/gates'
REPORT = ROOT / 'reports/gates'
SOURCES = ('metrica@v3', 'sc@v3')
CONVS = {'metrica@v3': ('ctrl', 'events'), 'sc@v3': ('pp', 'phase')}
DEC = (30.0, 8, 50)


def decode_all(pds, phs):
    """Same decoder, independent halves in threads; warm JIT before sharing."""
    def one(pair):
        return e2.decode_all(DEC, [pair[0]], [pair[1]])[0]
    first = one((pds[0], phs[0]))
    with ThreadPoolExecutor(max_workers=int(os.environ.get('GATE_DECODE_THREADS', '6'))) as pool:
        return [first] + list(pool.map(one, zip(pds[1:], phs[1:])))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def designs():
    # Order fixed before seeing outcomes. Each match occurs k times for size k.
    order = np.random.default_rng(20261024).permutation(dfl.MATCHES).tolist()
    return [(k, r, (order[r:] + order[:r])[:k]) for k in (2, 4) for r in range(7)] + [(7, 0, order)]


def model_path(k, r):
    return OUT / f'team_n{k}_r{r}.txt'


def prob_path(src, k, r):
    return str(OUT / f'ph_{src}_n{k}_r{r}.pkl')


def prepare():
    for src in ('dfl@v3',) + SOURCES:
        t = time.time()
        print('prepare', src, flush=True)
        bench.features(model_v7.VERSION, src)
        gc.collect()
        # Fixed seven-match dead model, compatible with the old bench.
        if src in SOURCES:
            bench.dead_ext('base', 'v3', src)
            for conv in CONVS[src]:
                bench.labels(src, conv)
        bench._blocks.clear()
        gc.collect()
        print('prepared', src, round(time.time() - t), 's', flush=True)


def train():
    fs = bench.features(model_v7.VERSION, 'dfl@v3')
    bl = bench.blocks('dfl@v3')
    labs = [b['lab'] for b in bl]
    save_json(REPORT / 'design.json', dict(designs=designs(), features=model_v7.VERSION,
              rounds=bench.ROUNDS, stride=bench.STRIDE, threads=bench.THREADS,
              decoder=dict(pen_team=12, pen_dead=30, min_poss=8, min_dead=50, b2_D0=175, b2_c=48)))
    for k, r, matches in designs():
        path = model_path(k, r)
        if path.exists():
            continue
        ids = [i for i, b in enumerate(bl) if b['group'] in matches]
        t = time.time()
        print('train', k, r, matches, flush=True)
        mdl = bench.fit_team([fs[i] for i in ids], [labs[i] for i in ids])
        mdl.save_model(str(path))
        del mdl
        gc.collect()
        print('trained', k, r, round(time.time() - t), 's', flush=True)


def scores(src, conv, preds):
    by_group = bench.spell_by_group(bench.blocks(src), bench.labels(src, conv), preds)
    by_match = {}
    for (match, noise_seed), result in by_group.items():
        by_match.setdefault(match, []).append(float(result[0]))
    return {m: float(np.mean(v)) for m, v in by_match.items()}


def evaluate():
    path = REPORT / 'learning_curve.json'
    results = json.loads(path.read_text()) if path.exists() else []
    for src in SOURCES:
        fs = bench.features(model_v7.VERSION, src)
        pds = bench.dead_ext('base', 'v3', src)
        for k, r, matches in sorted(designs(), key=lambda d: (-d[0], d[1])):
            ppath = prob_path(src, k, r)
            if os.path.isfile(ppath):
                phs = bench._load(ppath)
            else:
                mdl = lgb.Booster(model_file=str(model_path(k, r)))
                phs = [bench.predict_team(mdl, f) for f in fs]
                bench._dump(phs, ppath)
            decoded = {}
            for conv in CONVS[src]:
                labs = bench.labels(src, conv)
                for mode in ('learned', 'oracle'):
                    if any(x['source'] == src and x['n'] == k and x['rep'] == r and
                           x['conv'] == conv and x['dead'] == mode for x in results):
                        continue
                    pd_ = pds if mode == 'learned' else [np.where(l == 2, 1-1e-4, 1e-4) for l in labs]
                    if mode not in decoded:
                        decoded[mode] = decode_all(pd_, phs)
                    else:
                        # These convention pairs change team labels only.
                        assert all(np.array_equal(l == 2, first == 2) for l, first in
                                   zip(labs, bench.labels(src, CONVS[src][0])))
                    sc = scores(src, conv, decoded[mode])
                    row = dict(source=src, n=k, rep=r, conv=conv, dead=mode,
                               matches=matches, spell_by_match=sc, mean=float(np.mean(list(sc.values()))),
                               team_accuracy=float(e2.team_acc(phs, labs)))
                    results.append(row)
                    save_json(path, results)
                    print(src, k, r, conv, mode, round(row['mean'], 5), flush=True)
        del fs
        bench._blocks.clear()
        gc.collect()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare', 'train', 'evaluate'))
    p.add_argument('--source', choices=SOURCES)
    args = p.parse_args()
    if args.source:
        SOURCES = (args.source,)
    globals()[args.stage]()
