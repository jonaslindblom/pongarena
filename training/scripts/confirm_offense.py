"""Freeze and evaluate the isolated four-rule study on reserved matched serves."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from competition_eval import load_mlp
from offense_eval import make_policy, run_pair
from offense_rules import CONFIGS
from policy import ROOT

STUDY = ROOT / 'training/results/offense-study'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files():
    paths = [STUDY / 'protocol.json', ROOT / 'src/game.ts',
             ROOT / 'training/scripts/env.py', ROOT / 'training/v3/reference.py',
             ROOT / 'training/results/fully-learned/candidate/best.pt',
             ROOT / 'training/results/v2/ppo.pt']
    paths += [ROOT / 'training/scripts' / name for name in
              ['offense_rules.py', 'offense_eval.py', 'train_offense.py', 'confirm_offense.py', 'competition_eval.py', 'policy.py']]
    for name in CONFIGS:
        paths += [STUDY / 'training/search' / name / file for file in
                  ['best.pt', 'best.onnx', 'config.json', 'selection.json']]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    manifest_path = STUDY / 'manifest.json'
    if args.freeze:
        if manifest_path.exists():
            raise FileExistsError('The final evaluation is already frozen.')
        manifest_path.write_text(json.dumps(dict(
            purpose='Single frozen final comparison; no tuning on final outcomes',
            files=frozen_files()), indent=2) + '\n')
        print('Frozen', manifest_path)
        return
    manifest = json.loads(manifest_path.read_text())
    assert frozen_files() == manifest['files'], 'Frozen inputs changed; do not silently reuse reserved results.'
    protocol = json.loads((STUDY / 'protocol.json').read_text())
    seeds = protocol['final_seeds']
    output = STUDY / 'evaluation/confirmation.json'
    if output.exists():
        raise FileExistsError('Confirmation already exists; retain the original run.')
    torch.set_num_threads(1)
    initial = load_mlp(ROOT / 'training/results/fully-learned/candidate/best.pt')
    v2 = load_mlp(ROOT / 'training/results/v2/ppo.pt')
    result = dict(protocol='protocol.json', manifest_sha256=digest(manifest_path),
                  seed_count=len(seeds), seeds=seeds, configs={})
    pairs = ['attack:attack', 'attack:defend', 'defend:defend', 'attack:delayed',
             'initial:attack', 'initial:defend', 'trained:attack', 'trained:defend',
             'trained:initial', 'trained:v2', 'trained:trained']
    for name, rules in CONFIGS.items():
        trained = load_mlp(STUDY / 'training/search' / name / 'best.pt')
        result['configs'][name] = {}
        for pair in pairs:
            def make(kind):
                if kind == 'initial':
                    return initial
                if kind == 'trained':
                    return trained
                if kind == 'v2':
                    return v2
                return make_policy(rules, 'defend', delay=5) if kind == 'delayed' else make_policy(rules, kind)
            a, b = pair.split(':')
            started = time.monotonic()
            row = run_pair(rules, make(a), make(b), seeds,
                           capture_replays=5 if pair in ('trained:attack', 'attack:defend') else 0)
            row['seconds'] = time.monotonic() - started
            result['configs'][name][pair] = row
            output.write_text(json.dumps(result, indent=2) + '\n')
            s = row['summary']
            print(name, pair, s['wins'], s['losses'], s['draws'],
                  'median returns', s['median_decisive_returns'], flush=True)
    assert frozen_files() == manifest['files'], 'Inputs changed while evaluation was running.'
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Completed', output)


if __name__ == '__main__':
    main()
