"""Aggregate possession error diagnostics; no frame-level exports.

IoU=.5 counts are diagnostic, distinct from the headline five-IoU spellF1.
Merge/split overlap counts are structural proxies, not exclusive error classes.
"""
import numpy as np
import train_dfl as td
pm = td.pm
BUCKETS = ((0,50,'0-2s'),(50,100,'2-4s'),(100,200,'4-8s'),
           (200,375,'8-15s'),(375,float('inf'),'15+s'))


def matched(gs, ps):
    mg, mp = np.zeros(len(gs),bool), np.zeros(len(ps),bool)
    for iou,i,j in pm.spell_overlaps(gs,ps):
        if iou < .5: break
        if not mg[i] and not mp[j]: mg[i]=mp[j]=True
    return mg,mp


def overlaps(a,b):
    return [[j for j,(c,d,l) in enumerate(b) if k==l and min(z,d)>max(x,c)]
            for x,z,k in a]


def one(gt,pred):
    runs=pm.valid_runs(gt)
    pred=pm.fill_abstentions(pred,runs)
    gs=pm.possession_spells(gt,runs);ps=pm.possession_spells(pred,runs)
    mg,mp=matched(gs,ps)
    go,po=overlaps(gs,ps),overlaps(ps,gs)
    out=dict(gt=len(gs),pred=len(ps),matched=int(mg.sum()),missed=int((~mg).sum()),
        false_positive=int((~mp).sum()),missed_no_same_team_overlap=sum(not x for x in go),
        spurious_no_same_team_overlap=sum(not x for x in po),
        merge_proxy=sum(len(x)>=2 for x in po),split_proxy=sum(len(x)>=2 for x in go))
    for lo,hi,name in BUCKETS:
        mask=np.array([lo<=b-a<hi for a,b,k in gs],bool)
        out[name+'_gt']=int(mask.sum());out[name+'_missed']=int((mask&~mg).sum())
    latency=[];nt=np_=miss=spur=0
    # Never permit a transition pairing across an unknown-label gap.
    for run in runs:
        gtr=pm.transitions(gt,[run]);ptr=pm.transitions(pred,[run])
        pairs,mm,ss=pm.match_transitions(gtr,ptr,25,typed='to')
        nt+=len(gtr);np_+=len(ptr);miss+=len(mm);spur+=len(ss)
        latency += [p[0]-g[0] for g,p in pairs]
    out.update(transition_gt=nt,transition_pred=np_,transition_matched=len(latency),
        transition_missed=miss,transition_spurious=spur,
        latency_sum_frames=float(sum(latency)),abs_latency_sum_frames=float(sum(abs(x) for x in latency)),
        early_transitions=sum(x<0 for x in latency),late_transitions=sum(x>0 for x in latency))
    return out,mg


def diagnostics(blocks,labs,preds):
    bygroup={}
    for b,g,p in zip(blocks,labs,preds):
        key=(b['group'],b.get('seed',0));row,_=one(g,p)
        target=bygroup.setdefault(key,{})
        for k,v in row.items():target[k]=target.get(k,0)+v
    bymatch={}
    for (m,s),r in bygroup.items():bymatch.setdefault(m,[]).append(r)
    bymatch={m:{k:float(np.mean([r[k] for r in rr])) for k in rr[0]} for m,rr in bymatch.items()}
    counts={k:float(sum(r[k] for r in bymatch.values())) for k in next(iter(bymatch.values()))}
    def rate(num,den):return counts[num]/counts[den] if counts[den] else None
    rates=dict(missed_rate=rate('missed','gt'),false_discovery_rate=rate('false_positive','pred'),
        merge_proxy_rate=rate('merge_proxy','pred'),split_proxy_rate=rate('split_proxy','gt'),
        transition_recall=rate('transition_matched','transition_gt'),
        transition_precision=rate('transition_matched','transition_pred'),
        mean_latency_seconds=rate('latency_sum_frames','transition_matched'),
        mean_abs_latency_seconds=rate('abs_latency_sum_frames','transition_matched'))
    for k in ('mean_latency_seconds','mean_abs_latency_seconds'):
        if rates[k] is not None:rates[k]/=25
    for _,_,name in BUCKETS:rates[name+'_missed_rate']=rate(name+'_missed',name+'_gt')
    return dict(counts=counts,rates=rates,counts_by_match=bymatch,
                pooling='sum across matches after averaging Metrica noise realizations within match')


def complementary(blocks,labs,gru,lgb):
    groups={}
    for b,g,p,q in zip(blocks,labs,gru,lgb):
        runs=pm.valid_runs(g);gs=pm.possession_spells(g,runs)
        flags=[]
        for x in (p,q):
            flags.append(matched(gs,pm.possession_spells(pm.fill_abstentions(x,runs),runs))[0])
        a,c=flags;key=(b['group'],b.get('seed',0));r=groups.setdefault(key,np.zeros(4,float))
        r+=np.array([(a&c).sum(),(~a&c).sum(),(a&~c).sum(),(~a&~c).sum()])
    matches={}
    for (m,s),v in groups.items():matches.setdefault(m,[]).append(v)
    total=np.sum([np.mean(v,axis=0) for v in matches.values()],axis=0)
    return dict(zip(('both_match','only_lgb_matches','only_bigru_matches','neither_matches'),total.tolist()))
