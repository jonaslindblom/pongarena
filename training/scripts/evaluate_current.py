"""Evaluate current Classic v4 policies on explicit paired serve seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from competition_eval import validate_seeds
from pace_study import CONFIGS, make_policy, run_pair
from policy import ROOT, OnnxPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--seeds-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opponents', nargs='+', default=['robust-strong-v4', 'pacer-v4', 'attack', 'defend', 'interceptor'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output path.')
    seeds = validate_seeds(json.loads(args.seeds_file.read_text()))
    catalog = json.loads((ROOT/'training/catalog.json').read_text())
    models = {model['id']: ROOT/model['file'] for model in catalog['models']}
    if len(set(args.opponents)) != len(args.opponents) or any(name not in models and name not in ('attack','defend','interceptor') for name in args.opponents):
        parser.error('Choose distinct current model IDs or attack, defend, interceptor.')
    paths = [args.model.resolve()] + [models[name] for name in args.opponents if name in models]
    subprocess.run(['node','--import','tsx','training/scripts/check_models.ts',*map(str,paths)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    torch.set_num_threads(1)
    rules = CONFIGS['pace13-v9-scaled']
    actor = OnnxPolicy(args.model)
    opponents = {name: OnnxPolicy(models[name]) if name in models else make_policy(rules,name) for name in args.opponents}
    result = dict(rules='pong-v4',contract=catalog['contract'],seeds=seeds,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        opponent_sha256={name:hashlib.sha256(models[name].read_bytes()).hexdigest() for name in args.opponents if name in models},
        opponents={name:run_pair(rules,actor,opponent,seeds) for name,opponent in opponents.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
