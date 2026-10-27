"""CPU contact-stream gate extending exp_anchor beyond passes.

This is a synthetic value-of-information experiment, NOT a video detector.
Reference contacts and evaluation labels share event sources (optimistic).
Primary: event precision=.70, contact recall=.70, team accuracy=.95,
timing SD=3 frames, beta=4 fixed from the prior experiment (no tuning).
Also team70 (perfect event precision, .70 team accuracy), perfect ceiling,
beta=2 sensitivity, and legacy passes. Wrong contacts may be uncontrolled.
No ground-truth ownership is used to relabel or filter contacts.
"""
import gc
import json
from pathlib import Path
import warnings
import numpy as np
import bench
import exp_anchor as anchor
import exp_ctrl2 as e2
import exp_data_gate as gate


def corrupt_contacts(anc, n, rng, precision, recall, acc, jitter):
    a = anc[(anc[:, 0] >= 0) & (anc[:, 0] < n)]
    a = a[rng.random(len(a)) < recall].copy()
    # Precision refers to existence of a real event, independently of team.
    false_n = int(round(len(a) * (1 / precision - 1)))
    flips = rng.random(len(a)) >= acc
    a[flips, 1] = 1 - a[flips, 1]
    a[:, 0] += np.rint(rng.normal(0, jitter, len(a))).astype(np.int64)
    a = a[(a[:, 0] >= 0) & (a[:, 0] < n)]
    false = np.column_stack((rng.integers(0, n, false_n), rng.integers(0, 2, false_n)))
    if anc.shape[1] == 3:
        # Assumes perfect contact-type recognition for retained events: optimistic
        # typed sensitivity, not part of the predeclared primary gate.
        false = np.column_stack((false, rng.choice(anc[:, 2], false_n) if len(anc) else np.zeros(false_n)))
    return np.concatenate((a, false)), dict(true=len(a), false=false_n, wrong_team=int(flips.sum()))


def typed_evidence(a, n):
    windows = {-1: (12, 3), 0: (3, 3), 1: (3, 12)}
    # Use the existing cap within type and then cap the combined signal.
    return np.clip(sum(anchor.evidence(a[a[:, 2] == k, :2], n, *w) for k, w in windows.items()), -1, 1)


def main():
    out = gate.REPORT / 'contacts.json'
    results = json.loads(out.read_text()) if out.exists() else []
    scenarios = [('perfect', 1., 1., 1., 0., (0,)),
                 ('precision70', .7, .7, .95, 3., (0, 1, 2)),
                 ('team70', 1., .7, .7, 3., (0, 1, 2))]
    for src in gate.SOURCES:
        bl = bench.blocks(src)
        phs = bench._load(gate.prob_path(src, 7, 0))
        pds = bench.dead_ext('base', 'v3', src)
        refs = {}
        decoded = {}
        for conv in gate.CONVS[src]:
            labs = bench.labels(src, conv)
            for dm in ('learned', 'oracle'):
                ds = pds if dm == 'learned' else [np.where(l == 2, 1-1e-4, 1e-4) for l in labs]
                if dm not in decoded:
                    decoded[dm] = gate.decode_all(ds, phs)
                else:
                    assert all(np.array_equal(l == 2, first == 2) for l, first in
                               zip(labs, bench.labels(src, gate.CONVS[src][0])))
                refs[conv, dm] = (ds, gate.scores(src, conv, decoded[dm]))
        for mode in ('passes', 'contacts', 'contacts_typed'):
            fn = anchor.metrica_anchors if src.startswith('metrica') else anchor.sc_anchors
            ancs = [fn(b, contacts=mode != 'passes', typed=mode == 'contacts_typed') for b in bl]
            # Hold fusion fixed to the earlier pass experiment. Event expansion
            # must not be confounded with a simultaneous change of window.
            # Limitation: pre-contact evidence is less natural for interceptions.
            window = (12, 3)
            for name, precision, recall, acc, jitter, seeds in scenarios:
                for seed in seeds:
                    evs, stats = [], []
                    for i, (a, p) in enumerate(zip(ancs, phs)):
                        a, st = corrupt_contacts(a, len(p), np.random.default_rng(seed * 10000 + i), precision, recall, acc, jitter)
                        evs.append(typed_evidence(a, len(p)) if mode == 'contacts_typed' else anchor.evidence(a, len(p), *window))
                        stats.append(st)
                    for beta in ((2., 4.) if name == 'precision70' else (4.,)):
                        ps = [e2.sig(e2.logit(p) + beta * ev) for p, ev in zip(phs, evs)]
                        decoded = {}
                        for (conv, dm), (ds, ref) in refs.items():
                            ident = dict(source=src, anchors=mode, scenario=name, seed=seed, beta=beta, conv=conv, dead=dm)
                            if any(all(r.get(k) == v for k, v in ident.items()) for r in results):
                                continue
                            if dm not in decoded:
                                decoded[dm] = gate.decode_all(ds, ps)
                            sc = gate.scores(src, conv, decoded[dm])
                            delta = {m: sc[m] - ref[m] for m in sc}
                            row = dict(**ident, baseline=ref, spell_by_match=sc, delta_by_match=delta,
                                       delta=float(np.mean(list(delta.values()))), raw_contact_count=sum(map(len, ancs)),
                                       noise_counts={k: sum(s[k] for s in stats) for k in ('true', 'false', 'wrong_team')})
                            results.append(row)
                            gate.save_json(out, results)
                            print(src, mode, name, seed, beta, conv, dm, round(row['delta'], 5), flush=True)
        bench._blocks.clear()
        gc.collect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=gate.SOURCES)
    args = parser.parse_args()
    if args.source:
        gate.SOURCES = (args.source,)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
