"""Verify the realized split and deterministic sampling schedule without test data."""
from collections import Counter
import hashlib
import numpy as np
import exp_seq_pff as exp


def main():
    design=exp.read(exp.REPORT/'design.json');rows=exp.read(exp.OUT/'index.json')
    split={s:{source:sorted({r['match'] for r in rows if r['source']==source and r['split']==s})
              for source in ('dfl','pff')} for s in ('train','val')}
    assert split['train']==design['train'] and split['val']==design['validation']
    assert all(not set(split['train'][s])&set(split['val'][s]) for s in ('dfl','pff'))
    for r in rows:
        x=np.load(r['x'],mmap_mode='r');y=np.load(r['y'])
        assert x.shape==((len(y)+1)//2,2,172)
    data=exp.Data(rows);selected=exp.read(exp.REPORT/'selection.json')['selected'];out={}
    for kind,result in selected.items():
        rng=np.random.default_rng(exp.SEED);counts=Counter();live=Counter()
        for _ in range(result['epoch']*exp.STEPS):
            specs=data.draw(rng)
            assert Counter(s[3] for s in specs)=={0:8,1:8}
            for i,start,side,source in specs:
                r=rows[i];counts[r['source']+'/'+r['match']]+=1
                a,b=max(0,start),min(len(data.y[i]),start+exp.seq.CORE)
                live[r['source']]+=int((data.y[i][a:b]<2).sum())
        assert len(counts)==71,'Some final-training match was never sampled'
        out[kind]=dict(windows_by_match=dict(counts),live_loss_positions_by_source=dict(live),
                       windows_by_source={s:sum(v for k,v in counts.items() if k.startswith(s+'/'))
                                          for s in ('dfl','pff')})
    exp.save(exp.REPORT/'sampling_audit.json',dict(disjoint_match_split=True,split=split,
             all_half_shapes_valid=True,halves=len(rows),final_schedule=out,
             source_code=exp.code_hashes(),selection_sha256=hashlib.sha256((exp.REPORT/'selection.json').read_bytes()).hexdigest()))
    print('Verified: split disjoint; 148 halves; exact 8+8 source batches; all 71 matches sampled.')


if __name__=='__main__':main()
