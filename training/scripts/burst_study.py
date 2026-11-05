"""Reproducible paired-seed rule evaluation for the isolated burst preview."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from burst_rules import CONFIGS, DEFAULT, make_policy, run_pair
from burst_tactics import make_safe_policy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--configs',nargs='+',choices=CONFIGS,default=[DEFAULT.name])
    p.add_argument('--pairs',nargs='+',default=['attack:attack','attack:safe','safe:safe'])
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--confirmation',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh output file; retain previous evidence.')
    seed_base=2_830_000_000 if args.confirmation else 2_800_000_000
    rng_seed=20261214 if args.confirmation else 20261210
    seeds=(np.random.default_rng(rng_seed).choice(10_000_000 if args.confirmation else 100_000_000,args.count,replace=False)+seed_base).tolist()
    source_dir=Path(__file__).resolve().parent
    result=dict(purpose='fresh rule confirmation' if args.confirmation else 'diagnostic rule screen',
        seeds=seeds,rng_seed=rng_seed,configs={},sources={name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
        for name in ('burst_rules.py','burst_tactics.py','burst_study.py')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for name in args.configs:
        c=CONFIGS[name];result['configs'][name]={}
        policy=lambda kind:make_safe_policy(c) if kind=='safe' else make_policy(c,kind)
        for pair in args.pairs:
            a,b=pair.split(':')
            row=run_pair(c,policy(a),policy(b),seeds)
            result['configs'][name][pair]=row
            args.out.write_text(json.dumps(result,separators=(',',':'))+'\n')
            s=row['summary'];print(name,pair,s['wins'],s['losses'],s['draws'],'median',s['median_decisive_returns'],flush=True)
    result['complete']=True
    args.out.write_text(json.dumps(result,separators=(',',':'))+'\n')


if __name__=='__main__':main()
