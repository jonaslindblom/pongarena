"""Replay the frozen Quill-sized student's held-out comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from evaluate import onnx_policy, pair


def comparison(model, opponent, seeds):
    result = pair(onnx_policy(model), onnx_policy(opponent), seeds)
    margins = np.asarray(result['paired_point_margin'])
    scores = (margins > 0).astype(float) + .5 * (margins == 0)
    samples = np.random.default_rng(51083).choice(scores, (5000, len(seeds))).mean(1)
    result.update(league_score=float(scores.mean()),
                  league_score_ci95=np.quantile(samples, [.025, .975]).tolist(),
                  contest_wins=int((margins > 0).sum()),
                  contest_draws=int((margins == 0).sum()),
                  contest_losses=int((margins < 0).sum()))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output file.')
    saved = json.loads((HERE / 'results.json').read_text())
    model = ROOT / 'public/models/small.onnx'
    if hashlib.sha256(model.read_bytes()).hexdigest() != saved['primary_sha256']:
        raise ValueError('The frozen student bytes changed.')
    results = {}
    for name, expected in saved['opponent_sha256'].items():
        opponent = ROOT / 'public/models' / (name + '.onnx')
        if hashlib.sha256(opponent.read_bytes()).hexdigest() != expected:
            raise ValueError('The frozen opponent bytes changed.')
        results[name] = comparison(model, opponent, saved['seeds'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in row.items() if not k.startswith('paired_')}
                      for name, row in results.items()}, indent=2))
