"""B2/no-B2 sensitivity at fixed pen=12, across both label conventions.

This cannot identify the private competition GT's upstream annotation method.
It only quantifies the cost of the convention assumption on external games.
"""
import json
import numpy as np
import bench
import exp_data_gate as gate


def main():
    results = []
    for src in gate.SOURCES:
        ph = bench._load(gate.prob_path(src, 7, 0))
        learned = bench.dead_ext('base', 'v3', src)
        labs0 = bench.labels(src, gate.CONVS[src][0])
        for dm in ('learned', 'oracle'):
            pd = learned if dm == 'learned' else [np.where(l == 2, 1-1e-4, 1e-4) for l in labs0]
            b2 = gate.decode_all(pd, ph)
            plain = [bench.decode(d, h, 12, bench.DECODERS['v5']) for d, h in zip(pd, ph)]
            for conv in gate.CONVS[src]:
                sa, sb = gate.scores(src, conv, plain), gate.scores(src, conv, b2)
                row = dict(source=src, conv=conv, dead=dm, plain=sa, b2=sb,
                           delta=float(np.mean([sb[m] - sa[m] for m in sa])))
                results.append(row)
                print(src, conv, dm, 'B2 delta', round(row['delta'], 5), flush=True)
        bench._blocks.clear()
    gate.save_json(gate.REPORT / 'label_sensitivity.json', results)


if __name__ == '__main__':
    main()
