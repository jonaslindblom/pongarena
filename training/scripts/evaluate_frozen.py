"""Evaluate the preselected export on one reserved block, without selecting weights."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from competition_eval import evaluate_competition
from policy import OnnxPolicy, ROOT


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Frozen artifact changed: {path}')
    return actual


def apply_protocol(result, protocol):
    """A paired win can hide a drawn ordinary game; enforce the declared unit."""
    if 'ordinary_match_win_rate' in protocol['acceptance']:
        for row in result['opponents'].values():
            games = sum(row[key] for key in ['match_wins', 'match_losses', 'match_draws'])
            row['ordinary_match_win_rate'] = row['match_wins'] / games
            row['acceptance']['criteria']['majority_ordinary_games'] = row['ordinary_match_win_rate'] > .5
            row['acceptance']['passed'] = all(row['acceptance']['criteria'].values())
        result['primary_targets_pass_this_block'] = all(
            result['opponents'][name]['acceptance']['passed'] for name in protocol['primary_targets'])
        result['acceptance_protocol'] = protocol['acceptance']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--results', type=Path, default=ROOT / 'training/results/training-improvement')
    args = parser.parse_args()
    directory = args.results
    protocol = json.loads((directory / 'protocol.json').read_text())
    frozen = json.loads((directory / 'candidate/freeze.json').read_text())
    model = ROOT / frozen['onnx']
    model_hash = verify(model, frozen['onnx_sha256'])
    verify(ROOT / frozen['checkpoint'], frozen['checkpoint_sha256'])
    rules = protocol['rules_unchanged']
    verify(ROOT / rules['engine'], rules['sha256'])
    verify(ROOT / rules['contract'], rules['contract_sha256'])
    v2 = protocol['primary_targets']['selfplay-v2']
    verify(ROOT / v2['file'], v2['sha256'])
    verify(ROOT / 'training/scripts/env.py', protocol['primary_targets']['attacker']['source_sha256'])
    seed_file = directory / 'holdout-seeds.json'
    verify(seed_file, protocol['holdout']['sha256'])
    block = json.loads(seed_file.read_text())['blocks'][args.block]
    output = directory / 'holdouts' / f'{block["name"]}.json'
    if output.exists():
        raise FileExistsError(f'Result already recorded: {output}')
    torch.set_num_threads(1)
    result = evaluate_competition(OnnxPolicy(model), block['seeds'], native_reference=True)
    apply_protocol(result, protocol)
    result.update(model=str(model.relative_to(ROOT)), model_sha256=model_hash, block=block['name'])
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {k: row[k] for k in ['wins', 'losses', 'draws',
          'paired_match_win_rate', 'point_margin_ci95', 'acceptance']}
          for name, row in result['opponents'].items()}, indent=2))


if __name__ == '__main__':
    main()
