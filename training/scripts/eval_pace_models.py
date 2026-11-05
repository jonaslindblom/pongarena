"""Freeze two plain v4 networks, then evaluate on separate model-test seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from competition_eval import load_mlp
from pace_study import CONFIGS, make_policy, run_pair
from policy import ROOT, load_builtin
from train_pace import export_v4

OUT = ROOT/'training/results/pace-study/training'
SOURCES = {
    'pacer': OUT/'dagger-continuation-s412/best.pt',
    'apprentice': OUT/'dagger-s411/round-6.pt',
}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    args = p.parse_args()
    manifest = OUT/'manifest.json'
    if args.freeze:
        if manifest.exists():raise FileExistsError('Models already frozen.')
        files = {}
        for name,source in SOURCES.items():
            dest = OUT/'candidate'/name
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest/'best.pt')
            export_v4(load_mlp(source),dest/'best.onnx')
            for path in (source,dest/'best.pt',dest/'best.onnx'):
                files[str(path.relative_to(ROOT))] = digest(path)
        for source in ('pace_study.py','eval_pace_models.py'):
            path=ROOT/'training/scripts'/source
            files[str(path.relative_to(ROOT))]=digest(path)
        seeds=(np.random.default_rng(20261203).choice(100_000_000,64,replace=False)+2_600_000_000).tolist()
        manifest.write_text(json.dumps(dict(files=files,seeds=seeds,rng_seed=20261203,
            purpose='Untouched model test after development-only checkpoint selection',
            architecture=[6,128,128,3],training='DAgger: geometric labels during training only; no fixed inference features'),indent=2)+'\n')
        print('Frozen',manifest)
        return
    frozen=json.loads(manifest.read_text())
    def verify():
        for name,h in frozen['files'].items():assert digest(ROOT/name)==h,name
    verify()
    final=OUT/'confirmation.json'
    if final.exists():raise FileExistsError('Do not overwrite a reserved model evaluation.')
    torch.set_num_threads(1)
    rules=CONFIGS['pace13-v9-scaled']
    nets={name:load_mlp(OUT/'candidate'/name/'best.pt') for name in SOURCES}
    policies=dict(nets,attack=make_policy(rules,'attack'),defend=make_policy(rules,'defend'),
        contender=load_mlp(ROOT/'training/results/fully-learned/candidate/best.pt'),
        v2=load_mlp(ROOT/'training/results/v2/ppo.pt'),tracker=load_builtin('tracker'),anticipator=load_builtin('anticipator'))
    pairs=[f'pacer:{name}' for name in policies]+['apprentice:tracker','apprentice:anticipator','apprentice:contender']
    result=dict(seeds=frozen['seeds'],manifest_sha256=digest(manifest),pairs={})
    for pair in pairs:
        a,b=pair.split(':')
        row=run_pair(rules,policies[a],policies[b],frozen['seeds'],capture_replays=5 if pair in ('pacer:attack','pacer:apprentice','apprentice:tracker') else 0)
        result['pairs'][pair]=row
        final.write_text(json.dumps(result,separators=(',',':'))+'\n')
        s=row['summary'];print(pair,s['wins'],s['losses'],s['draws'],'median',s['median_decisive_returns'],flush=True)
    verify()
    result['complete']=True
    final.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print('Completed',final)


if __name__=='__main__':main()
