"""Reproduce the frozen BAR-ATH BiGRU ensemble from shared weights only.

Requires the participant's own BAR-ATH assets and official evaluator checkout.
No DFL/PFF raw data, training, private cache paths or old probability files needed.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import atd
import bench
import evaluate as ev
import make_submission
import model_v7
import seq_pff as seq

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    ap.add_argument('--out',type=Path,default=ROOT/'submissions/shared_bigru/bigru_ensemble_equal.csv')
    ap.add_argument('--score',action='store_true',help='Optional public-window check after prediction')
    args=ap.parse_args()
    folder=ROOT/'models/bigru_dfl_pff_v1'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4);torch.manual_seed(20261024);np.random.seed(20261024)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
        torch.cuda.manual_seed_all(20261024)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    f=make_submission.merged_features()
    Path(bench.INP).mkdir(parents=True,exist_ok=True)
    bench._barath_inputs()
    blocks=bench.blocks('barath');fs=bench.features(model_v7.VERSION,'barath',procs=2)
    members=[]
    for item in manifest['models']:
        path=folder/item['file']
        if sha(path)!=item['sha256']:raise ValueError('Checkpoint hash mismatch: '+item['file'])
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if (ck['step'],ck['core'],ck['halo'])!=(seq.STEP,seq.CORE,seq.HALO):raise ValueError('Window settings differ')
        c=ck['config'];net=seq.Net(345,c['kind'],c['width'],c['drop']).to(args.device)
        net.load_state_dict(ck['state']);net.eval();sc=seq.Scaler(ck['mean'],ck['scale'])
        ph=[]
        for ff in fs:
            if ff['fk']+ff['tk']!=manifest['feature_names']:raise ValueError('Feature order differs')
            ph.append(seq.predict_raw(net,sc,seq.views(ff),len(ff['XF'])))
        members.append(ph);del net
        if args.device=='cuda':torch.cuda.empty_cache()
        print('Predicted',item['file'],flush=True)
    probs={}
    for i,b in enumerate(blocks):
        probs[b['a']]=(None,np.mean(np.stack([m[i].astype(np.float64) for m in members]),axis=0))
    pp=ROOT/'cache/shared_bigru/barath_equal_probabilities.pkl'
    pp.parent.mkdir(parents=True,exist_ok=True);bench._dump(probs,str(pp))
    labels=model_v7.predict(f,probs_path=str(pp))
    ev.write(labels,str(args.out.resolve()))
    actual=sha(args.out);exact=actual==manifest['expected_csv_sha256']
    record=dict(csv=str(args.out.resolve()),sha256=actual,expected_sha256=manifest['expected_csv_sha256'],
        exact_match=exact,rows=len(labels),device=args.device,torch=torch.__version__,
        probabilities=str(pp.relative_to(ROOT)))
    if args.score:record['public_score']=ev.score(labels)
    (pp.parent/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
    if not exact:raise RuntimeError('CSV differs from frozen reference; inspect device/version/features before submitting')


if __name__=='__main__':main()
