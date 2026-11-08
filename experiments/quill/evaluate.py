"""Compare frozen handwritten ONNX variants; no training or weight selection.

The vectorized simulator is the repository's training/game.py physics, without
importing any training policy, optimizer, teacher, or PyTorch dependency. Both
sides replay the same five serves. Use exploration results to choose a design,
then freeze it before running a fresh holdout. This is a CPU experiment report,
not the published browser/WASM league.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from game import Arena, RULES_ID, serve  # noqa: E402

INTERVAL_METHOD = (
    '95% percentile bootstrap; 4096 resamples of whole shared serve seeds, '
    'retaining both sides, all five rallies, and every opponent together. '
    'Intervals measure serve-set uncertainty, not certainty about model ranks.'
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_policy(path):
    """Freeze original model bytes and batch the same dense inference graph."""
    path = Path(path).resolve()
    data = path.read_bytes()
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model)
    if (len(model.graph.input) != 1 or len(model.graph.output) != 1 or
            model.graph.input[0].name != 'observation' or
            model.graph.output[0].name != 'action_scores'):
        raise ValueError(f'{path}: expected observation and action_scores ports')
    for port, width in [(model.graph.input[0], 6), (model.graph.output[0], 3)]:
        shape = port.type.tensor_type.shape.dim
        if (port.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or
                len(shape) != 2 or shape[0].dim_value != 1 or shape[1].dim_value != width):
            raise ValueError(f'{path}: expected fixed float32 [1, {width}] port')
        shape[0].ClearField('dim_value')
        shape[0].dim_param = 'batch'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=['CPUExecutionProvider'])

    def policy(observations):
        scores = session.run(['action_scores'], {'observation': observations})[0]
        if scores.shape != (len(observations), 3) or not np.isfinite(scores).all():
            raise ValueError(f'{path}: model returned invalid action scores')
        return scores.argmax(axis=1)

    parameters = sum(int(np.prod(tensor.dims)) for tensor in model.graph.initializer)
    macs = sum(int(np.prod(tensor.dims)) for tensor in model.graph.initializer
               if len(tensor.dims) == 2)
    return policy, {'path': str(path), 'sha256': sha256(data),
                    'parameters': parameters, 'macs': macs}


def paired_outcomes(candidate, opponent, seeds):
    """Return [side, seed, rally] winners (0/1) or draws (2)."""
    seeds = list(seeds)
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Use distinct uint32 serve seeds')
    frames = []
    for _side in (0, 1):
        for seed in seeds:
            frame = None
            for _rally in range(5):
                frame = serve(seed, frame)
                frames.append(frame)
    sides = np.repeat([0, 1], len(seeds) * 5)
    arena = Arena(len(frames), loop_detection=True)
    arena.load(frames)
    outcomes = np.full(len(frames), -1, dtype=np.int8)
    for _tick in range(1600):
        active = outcomes < 0
        left, right = arena.obs(0), arena.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        candidate_actions = np.ones(len(frames), dtype=np.int64)
        opponent_actions = np.ones(len(frames), dtype=np.int64)
        candidate_actions[active] = candidate(ours[active])
        opponent_actions[active] = opponent(theirs[active])
        actions = np.stack([np.where(sides == 0, candidate_actions, opponent_actions),
                            np.where(sides == 0, opponent_actions, candidate_actions)], axis=1)
        outcome, _ = arena.step(actions, active)
        outcomes[active] = outcome[active]
        if np.all(outcomes >= 0):
            break
    if np.any(outcomes < 0):
        raise RuntimeError('The simulator returned incomplete rallies')
    return outcomes.reshape(2, len(seeds), 5)


def interval(values, resamples):
    """Keep the league convention: no interval without observed variation."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    return np.quantile(values[resamples].mean(axis=1), [.025, .975]).tolist()


def summarize(outcomes, resamples):
    outcomes = np.asarray(outcomes)
    if (outcomes.ndim != 3 or outcomes.shape[0] != 2 or outcomes.shape[2] != 5 or
            not np.isin(outcomes, [0, 1, 2]).all()):
        raise ValueError('Expected ten completed paired rallies per serve seed')
    sides = np.array([0, 1])[:, None, None]
    won = outcomes == sides
    lost = (outcomes < 2) & ~won
    drawn = outcomes == 2
    by_seed_margin = (won.astype(int) - lost.astype(int)).sum(axis=(0, 2))
    rally_scores = .5 + by_seed_margin / 20
    contest_scores = .5 + np.sign(by_seed_margin) / 2
    return {
        'rally_wins': int(won.sum()), 'rally_draws': int(drawn.sum()),
        'rally_losses': int(lost.sum()), 'rally_score': float(rally_scores.mean()),
        'rally_score_ci95': interval(rally_scores, resamples),
        'contest_wins': int((by_seed_margin > 0).sum()),
        'contest_draws': int((by_seed_margin == 0).sum()),
        'contest_losses': int((by_seed_margin < 0).sum()),
        'contest_score': float(contest_scores.mean()),
        'contest_score_ci95': interval(contest_scores, resamples),
        'paired_point_margin': by_seed_margin.tolist(),
        'paired_rally_score': rally_scores.tolist(),
        'paired_contest_score': contest_scores.tolist(),
    }


def named_paths(values):
    records = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or not name.strip() or not path or name in records:
            raise ValueError('Use distinct nonempty NAME=PATH model entries')
        records[name] = path
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--opponent', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--seed', type=int, required=True, help='RNG seed, recorded for replay')
    parser.add_argument('--count', type=int, default=32, help='Shared paired serve seed sets')
    parser.add_argument('--label', choices=['exploratory', 'holdout'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh report path; existing results are not overwritten')
    if not 1 <= args.count <= 512 or args.seed < 0:
        raise ValueError('Use 1–512 serve seed sets and a nonnegative RNG seed')
    candidate_paths, opponent_paths = named_paths(args.candidate), named_paths(args.opponent)
    cache = {}
    candidates, opponents = {}, {}
    for paths, destination in [(candidate_paths, candidates), (opponent_paths, opponents)]:
        for name, path in paths.items():
            key = str(Path(path).resolve())
            if key not in cache:
                cache[key] = load_policy(key)
            destination[name] = cache[key]
    # Freeze every model above before sampling. Separate ranges make exploratory
    # and holdout runs disjoint even when the caller reuses its RNG seed.
    low, high = ((1_000_000_000, 2_000_000_000) if args.label == 'exploratory'
                 else (2_000_000_000, 2**32))
    seeds = sorted((np.random.default_rng(args.seed).choice(
        high - low, args.count, replace=False) + low).tolist())
    resamples = np.random.default_rng(831701).integers(0, len(seeds), (4096, len(seeds)))
    report = {
        'schema': 'pongarena-handwritten-experiment-v1', 'label': args.label,
        'rules_id': RULES_ID, 'backend': 'ONNX Runtime CPU, dynamic batch, one thread',
        'numpy_version': np.__version__, 'onnx_version': onnx.__version__,
        'onnxruntime_version': ort.__version__, 'rng_seed': args.seed, 'seeds': seeds,
        'rallies_per_side': 5, 'interval_method': INTERVAL_METHOD,
        'evaluator_sha256': sha256(Path(__file__).read_bytes() +
                                   (ROOT / 'training/game.py').read_bytes()),
        'candidates': {name: record for name, (_, record) in candidates.items()},
        'opponents': {name: record for name, (_, record) in opponents.items()},
        'results': {}, 'aggregate': {},
    }
    started = time.monotonic()
    for name, (candidate, _) in candidates.items():
        report['results'][name] = {}
        for opponent_name, (opponent, _) in opponents.items():
            outcomes = paired_outcomes(candidate, opponent, seeds)
            result = summarize(outcomes, resamples)
            report['results'][name][opponent_name] = result
            print(f'{name} vs {opponent_name}: rally {result["rally_score"]:.1%}; '
                  f'contest {result["contest_score"]:.1%} '
                  f'({result["contest_wins"]}/{result["contest_draws"]}/{result["contest_losses"]})',
                  flush=True)
        pair_results = report['results'][name].values()
        aggregate = {}
        for metric in ['rally', 'contest']:
            per_seed = np.mean([result[f'paired_{metric}_score'] for result in pair_results], axis=0)
            aggregate[f'{metric}_score'] = float(per_seed.mean())
            aggregate[f'{metric}_score_ci95'] = interval(per_seed, resamples)
            aggregate[f'paired_{metric}_score'] = per_seed.tolist()
        report['aggregate'][name] = aggregate
    report['elapsed_seconds'] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.out} in {report["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
