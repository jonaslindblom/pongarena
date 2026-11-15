"""Freeze and replay the current five-model Findings comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import secrets
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from evaluate import INTERVAL_METHOD, onnx_policy, pair
from game import RULES_ID

MODEL_IDS = ('quill', 'small', 'oneshot', 'large', 'counter')
EVALUATOR_FILES = ('training/game.py', 'training/evaluate.py', 'training/models.py',
                   'experiments/quill-training/current-comparison.py')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    models = []
    for name in MODEL_IDS:
        path = ROOT / 'public/models' / (name + '.onnx')
        models.append({'id': name, 'name': {'oneshot': 'OneShot'}.get(name, name.title()),
                       'sha256': sha256(path),
                       'params': sum(math.prod(tensor.dims) for tensor in onnx.load(path).graph.initializer)})
    return {'models': models, 'evaluator_sha256': {file: sha256(ROOT / file) for file in EVALUATOR_FILES}}


def write_new(path, data):
    with path.open('x') as handle:
        handle.write(json.dumps(data, indent=2) + '\n')


def freeze(path):
    # Freeze every entrant before drawing any serves. This comparison does not select a model.
    frozen = inputs()
    excluded = set(json.loads((HERE / 'results.json').read_text())['seeds'])
    for file in (ROOT / 'experiments/one-prompt-weights').rglob('seeds.json'):
        excluded.update(json.loads(file.read_text()))
    seeds = set()
    while len(seeds) < 512:
        seed = 2_800_000_000 + secrets.randbelow(2 ** 32 - 2_800_000_000)
        if seed not in excluded:
            seeds.add(seed)
    if inputs() != frozen:
        raise ValueError('Frozen model or evaluator changed while drawing serves.')
    write_new(path, {'schema': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                     'rules_id': RULES_ID, 'contract_id': 'dense-mlp', **frozen,
                     'seeds': sorted(seeds), 'rallies_per_side': 5,
                     'seed_method': '512 distinct random uint32 seeds above 2.8 billion, drawn after freezing all five models; excludes the earlier study and prompt experiment serves.',
                     'selection': 'Small was selected on development results before this plan. All five retained models are compared; these results select no checkpoint.'})


def evaluate(plan_path, output):
    plan = json.loads(plan_path.read_text())
    frozen = {key: plan[key] for key in ('models', 'evaluator_sha256')}
    if inputs() != frozen:
        raise ValueError('The frozen model bytes or evaluator changed.')
    seeds = plan['seeds']
    if len(seeds) != 512 or len(set(seeds)) != 512 or any(not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError('Expected 512 distinct uint32 seeds.')
    policies = {name: onnx_policy(ROOT / 'public/models' / (name + '.onnx')) for name in MODEL_IDS}
    matchups = []
    for model, opponent in itertools.combinations(MODEL_IDS, 2):
        result = pair(policies[model], policies[opponent], seeds)
        margins = np.asarray(result['paired_point_margin'])
        scores = (margins > 0).astype(float) + .5 * (margins == 0)
        samples = np.random.default_rng(51083).choice(scores, (5000, len(seeds))).mean(1)
        result.update(league_score=float(scores.mean()),
                      league_score_ci95=np.quantile(samples, [.025, .975]).tolist(),
                      contest_wins=int((margins > 0).sum()),
                      contest_draws=int((margins == 0).sum()),
                      contest_losses=int((margins < 0).sum()))
        matchups.append({'model': model, 'opponent': opponent, **result})
        print(f"{model} vs {opponent}: rally {result['rally_score']:.6f}, league {result['league_score']:.6f}", flush=True)
    if inputs() != frozen:
        raise ValueError('Frozen model or evaluator changed during the comparison.')
    write_new(output, {'plan': plan, 'runtime': {'numpy': np.__version__, 'onnxruntime': ort.__version__},
                       'rally_interval_method': INTERVAL_METHOD,
                       'league_interval_method': '95% percentile bootstrap; 5000 resamples of whole paired serve sets, using 1/0.5/0 for a set win/draw/loss. Describes serve-set uncertainty, not training variation.',
                       'matchups': matchups})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('freeze')
    create.add_argument('plan', type=Path)
    replay = commands.add_parser('evaluate')
    replay.add_argument('plan', type=Path)
    replay.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.plan)
    else:
        evaluate(args.plan, args.output)
