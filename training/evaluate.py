"""Evaluate frozen controllers on paired sides and five identical seeded serves."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from game import Arena, RULES_ID, serve
from models import decide
from teachers import teacher

DEVELOPMENT_SEEDS = [1100000149, 1200000311, 1300000487, 1400000621,
                     1500000793, 1600000931, 1700001097, 1800001249]
DEVELOPMENT_SEEDS += [1_000_000_000 + int(x) for x in
                      np.random.default_rng(901837).choice(1_000_000_000, 24, replace=False)]
FINAL_SEEDS = [2_510_000_000 + int(x) for x in
               np.random.default_rng(612907).choice(90_000_000, 128, replace=False)]
INTERVAL_METHOD = ('95% percentile bootstrap of mean rally score; 4096 resamples of serve seeds, '
                   'keeping both sides and all five rallies together. Measures serve-set uncertainty, '
                   'not variation between training runs.')


def summarize(outcomes, sides, seed_count):
    """Treat each paired serve seed as one observation, not ten independent rallies."""
    outcomes, sides = np.asarray(outcomes), np.asarray(sides)
    if seed_count < 1 or outcomes.shape != (seed_count * 10,) or sides.shape != outcomes.shape:
        raise ValueError('Expected ten paired rallies per evaluation seed.')
    if not np.isin(outcomes, [0, 1, 2]).all() or not np.isin(sides, [0, 1]).all():
        raise ValueError('Evaluation requires completed rallies and valid player sides.')
    won = outcomes == sides
    lost = (outcomes < 2) & ~won
    draws = outcomes == 2
    by_seed = (won.astype(int) - lost.astype(int)).reshape(2, seed_count, 5).sum((0, 2))
    paired_scores = .5 + by_seed / 20
    samples = np.random.default_rng(831701).choice(paired_scores, (4096, seed_count)).mean(axis=1)
    interval = np.quantile(samples, [.025, .975]).tolist()
    return {'wins': int(won.sum()), 'draws': int(draws.sum()), 'losses': int(lost.sum()),
            'rally_score': float(paired_scores.mean()), 'rally_score_ci95': interval,
            'point_margin': float(by_seed.mean() / 10),
            'paired_point_margin': by_seed.tolist(), 'paired_rally_score': paired_scores.tolist()}


def onnx_policy(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # The submission has a fixed batch of one. A temporary evaluation graph can
    # use a dynamic batch without changing any operators or weights.
    import onnx
    graph = onnx.load(path)
    for port in [graph.graph.input[0], graph.graph.output[0]]:
        port.type.tensor_type.shape.dim[0].ClearField('dim_value')
        port.type.tensor_type.shape.dim[0].dim_param = 'batch'
    session = ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider'])
    return lambda x: session.run(None, {'observation': x})[0].argmax(-1)


def pair(a, b, seeds):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError('Evaluation seeds must be distinct uint32 values.')
    frames, sides = [], []
    for side in (0, 1):
        for seed in seeds:
            frame = None
            for _ in range(5):
                frame = serve(seed, frame)
                frames.append(frame)
                sides.append(side)
    sides = np.asarray(sides)
    arena = Arena(len(frames), loop_detection=True)
    arena.load(frames)
    outcomes = np.full(len(frames), -1)
    for _ in range(1600):
        active = outcomes < 0
        left, right = arena.obs(0), arena.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        ca, cb = np.ones(len(frames), int), np.ones(len(frames), int)
        ca[active], cb[active] = decide(a, ours[active]), decide(b, theirs[active])
        actions = np.stack([np.where(sides == 0, ca, cb), np.where(sides == 0, cb, ca)], 1)
        outcome, _ = arena.step(actions, active)
        outcomes[active] = outcome[active]
        if (outcomes >= 0).all():
            break
    assert (outcomes >= 0).all()
    return summarize(outcomes, sides, len(seeds))


def field(policy, seeds):
    opponents = {kind: teacher(kind) for kind in ('attack', 'defend', 'interceptor')}
    return {name: pair(policy, opponent, seeds) for name, opponent in opponents.items()}


def development_field(policy, seeds=DEVELOPMENT_SEEDS):
    """Checkpoint selection cannot consume final or fresh evaluation seeds."""
    if not seeds or any(not 1_000_000_000 <= seed < 2_000_000_000 for seed in seeds):
        raise ValueError('Checkpoint selection requires development seeds in [1e9, 2e9).')
    return field(policy, seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    split = parser.add_mutually_exclusive_group()
    split.add_argument('--seeds', choices=['development', 'final'], default='final')
    split.add_argument('--seed', type=int, help='Generate 128 fresh evaluation seeds from this RNG seed')
    parser.add_argument('--opponent', type=Path, help='Also compare against this ONNX after freezing your model')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path.')
    seeds = ([int(x) + 2_000_000_000 for x in np.random.default_rng(args.seed).choice(400_000_000, 128, replace=False)]
             if args.seed is not None else DEVELOPMENT_SEEDS if args.seeds == 'development' else FINAL_SEEDS)
    data = {'rules_id': RULES_ID, 'model': args.model.name, 'sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'seeds': seeds, 'rallies_per_side': 5, 'interval_method': INTERVAL_METHOD,
            'opponents': field(onnx_policy(args.model), seeds)}
    if args.opponent:
        data['additional_opponent'] = {
            'model': args.opponent.name,
            'sha256': hashlib.sha256(args.opponent.read_bytes()).hexdigest(),
            'results': pair(onnx_policy(args.model), onnx_policy(args.opponent), seeds)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: v for k, v in data.items() if k in ('opponents', 'additional_opponent')}, indent=2))


if __name__ == '__main__':
    main()
