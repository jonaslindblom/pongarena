"""CMA-ES outcome refinement of an independently trained Quill-sized policy."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from distill import HERE, WIDTHS, development, digest, learned_teacher
from export import export_model
from game import Arena, serve
from models import network, decide
from teachers import teacher


def population_scores(vectors, seeds, opponents):
    """Evaluate candidates together on identical paired serves and opponent fields."""
    vectors = np.asarray(vectors, dtype=np.float32)
    population = len(vectors)
    frames, sides, opponent_ids = [], [], []
    for opponent_id in range(len(opponents)):
        for side in (0, 1):
            for seed in seeds:
                frame = None
                for _ in range(5):
                    frame = serve(seed, frame)
                    frames.append(frame)
                    sides.append(side)
                    opponent_ids.append(opponent_id)
    count = len(frames)
    sides = np.tile(sides, population)
    opponent_ids = np.tile(opponent_ids, population)
    arena = Arena(population * count, loop_detection=True)
    arena.load(frames * population)
    outcomes = np.full(arena.n, -1)
    layers, offset = [], 0
    for a, b in zip(WIDTHS, WIDTHS[1:]):
        weight = vectors[:, offset:offset + a*b].reshape(population, b, a)
        offset += a*b
        bias = vectors[:, offset:offset + b].reshape(population, 1, b)
        offset += b
        layers.append((torch.from_numpy(weight.copy()), torch.from_numpy(bias.copy())))
    if offset != vectors.shape[1]:
        raise ValueError('Expected exactly 163 student parameters.')
    for _ in range(1600):
        active = outcomes < 0
        left, right = arena.obs(0), arena.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        with torch.no_grad():
            value = torch.from_numpy(ours.reshape(population, count, 6))
            for i, (weight, bias) in enumerate(layers):
                value = torch.bmm(value, weight.transpose(1, 2)) + bias
                if i < len(layers) - 1:
                    value = value.relu()
            action = value.argmax(-1).numpy().flatten()
        other = np.ones(arena.n, int)
        for i, opponent in enumerate(opponents):
            mask = active & (opponent_ids == i)
            if mask.any():
                other[mask] = decide(opponent, theirs[mask])
        outcome, _ = arena.step(np.stack([np.where(sides == 0, action, other),
                                           np.where(sides == 0, other, action)], 1), active)
        outcomes[active] = outcome[active]
        if (outcomes >= 0).all():
            break
    if (outcomes < 0).any():
        raise RuntimeError('Incomplete search evaluation.')
    margins = ((outcomes == sides).astype(int) - ((outcomes < 2) & (outcomes != sides)))
    margins = margins.reshape(population, len(opponents), 2, len(seeds), 5).sum((2, 4))
    league = ((margins > 0) + .5 * (margins == 0)).mean((1, 2))
    rally = (.5 + margins / 20).mean((1, 2))
    return np.stack([league, rally], 1)


def search(parent, output, seed, generations=96, population=24, serve_sets=8, sigma=.08):
    import cma
    if output.exists():
        raise FileExistsError('Choose a fresh output directory.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(parent / 'model.pt', map_location='cpu', weights_only=True)
    if saved['widths'] != WIDTHS or saved['activation'] != 'Relu':
        raise ValueError('Search must preserve the exact student architecture.')
    policy = network(WIDTHS, 'Relu')
    policy.load_state_dict(saved['state_dict'])
    opponents = {kind: teacher(kind) for kind in ('attack', 'defend', 'interceptor')}
    opponents['large'] = learned_teacher(HERE.parent / 'public/models/large.onnx')
    initial = nn.utils.parameters_to_vector(policy.parameters()).detach().numpy().copy()
    output.mkdir(parents=True)
    config = dict(seed=seed, generations=generations, population=population,
                  serve_sets=serve_sets, sigma=sigma, parent_sha256=digest(parent / 'model.onnx'),
                  widths=WIDTHS, activation='Relu', cma_version=cma.__version__)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    optimizer = cma.CMAEvolutionStrategy(initial, sigma, {'seed': seed, 'popsize': population,
        'verbose': -9, 'verb_log': 0, 'maxiter': generations})
    rng = np.random.default_rng(seed)
    best = (-1., -1.)
    started = time.monotonic()

    def select(vector, generation):
        nonlocal best
        nn.utils.vector_to_parameters(torch.from_numpy(np.asarray(vector, dtype=np.float32).copy()), policy.parameters())
        score, results = development(policy, opponents)
        if score > best:
            best = score
            torch.save({'widths': WIDTHS, 'activation': 'Relu', 'state_dict': policy.state_dict()}, output / 'model.pt')
            export_model(policy, output / 'model.onnx', 'Relu', 'Quill-sized student',
                         'Independent imitation and outcome learning, followed by CMA-ES')
            (output / 'selection.json').write_text(json.dumps(dict(phase='search', generation=generation,
                development_score=score, opponents=results), indent=2) + '\n')

    select(initial, 0)
    for generation in range(1, generations + 1):
        vectors = optimizer.ask()
        seeds = rng.choice(1_000_000_000, serve_sets, replace=False).tolist()
        scores = population_scores(vectors, seeds, list(opponents.values()))
        fitness = scores[:, 0] + .25 * scores[:, 1]
        optimizer.tell(vectors, (-fitness).tolist())
        if generation % 8 == 0 or generation == generations:
            select(vectors[int(fitness.argmax())], generation)
            select(optimizer.mean, generation)
        row = dict(generation=generation, training_score=scores[int(fitness.argmax())].tolist(),
                   development_score=best, sigma=float(optimizer.sigma))
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    (output / 'run.json').write_text(json.dumps(dict(seconds=time.monotonic() - started,
        best_development_score=best, candidate_evaluations=generations*population,
        training_rallies=generations*population*serve_sets*10*len(opponents),
        sha256=digest(output / 'model.onnx')), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--generations', type=int, default=96)
    parser.add_argument('--sigma', type=float, default=.08)
    args = parser.parse_args()
    search(args.resume, args.out, args.seed, args.generations, sigma=args.sigma)
