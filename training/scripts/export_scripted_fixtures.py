"""Freeze the evaluated Classic v4 scripts for browser-port conformance tests.

Run: .venv/bin/python training/scripts/export_scripted_fixtures.py
Actions come from the original NumPy policies, movement-path digests from their
dynamic-programming table, and full matches from the independent v4 scalar port.
No browser implementation is imported or used to construct expected results.
"""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from offense_eval import contact_paths
from pace_study import CONFIGS, make_policy

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'tests/fixtures/scripted-v4.json'
spec = importlib.util.spec_from_file_location('classic_scalar', ROOT / 'training/v4/reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
KINDS = ('attack', 'defend', 'interceptor')
POLICIES = {kind: make_policy(CONFIGS['pace13-v9-scaled'], kind) for kind in KINDS}
SCORES = [[int(i == action) for i in range(3)] for action in range(3)]


def state(frame):
    """Fixed field order and integer spelling make cross-language hashing exact."""
    return [frame[key] for key in ('tick', 'rally', 'rallyTick', 'x', 'y', 'vx', 'vy')] + [
        *frame['paddles'], *frame['points'], frame['returns'],
        (None, 'serve', 'return', 'point', 'draw').index(frame['event']),
        frame.get('actor', -1), (None, 'loop', 'timeout').index(frame.get('drawReason'))]


def match(seed, kinds):
    frame, loops = ref.serve(seed), ref.LoopDetector()
    digest = hashlib.sha256()
    actions, endings = [], []
    while True:
        digest.update((','.join(map(str, state(frame))) + '\n').encode())
        if ref.ended_rally(frame):
            endings.append(state(frame))
            if ref.ended_match(frame):
                break
            frame = ref.serve(seed, frame)
            continue
        chosen = [int(POLICIES[kind](np.asarray([ref.observation(frame, side)], dtype=np.float32))[0])
                  for side, kind in enumerate(kinds)]
        actions.append(str(3 * chosen[0] + chosen[1]))
        frame = ref.advance(frame, [SCORES[a] for a in chosen], loops)
    return dict(seed=seed, kinds=kinds, actions=''.join(actions),
                trajectory_sha256=digest.hexdigest(), endings=endings)


def main():
    # Raw physical states are compact; the fixtures also exercise float32
    # normalization and decoding at every speed, walls and clamped paddle edges.
    rng = np.random.default_rng(20261105)
    raw = np.stack([rng.integers(15, 346, 3000), rng.integers(4, 157, 3000),
                    rng.integers(5, 14, 3000) * rng.choice([-1, 1], 3000),
                    rng.integers(-9, 10, 3000), rng.integers(15, 146, 3000),
                    rng.integers(15, 146, 3000)], axis=1).tolist()
    for speed, side, own, height in itertools.product(range(5, 14), [-1, 1],
            [15, 16, 17, 79, 80, 81, 143, 144, 145], [4, 5, 79, 80, 81, 155, 156]):
        raw.append([16 if side < 0 else 344, height, side * speed, 0, own, 80])
    observations = np.asarray(raw, dtype=np.float64)
    observations = ((observations - [180, 80, 0, 0, 80, 80]) / [180, 80, 6, 6, 80, 80]).astype(np.float32)
    labels = np.stack([POLICIES[kind](observations) for kind in KINDS], axis=1)
    probes = [row + list(map(int, label)) for row, label in zip(raw, labels)]

    # Exactly representable half-integer decoded fields cover NumPy ties-to-even.
    halves = np.asarray(list(itertools.product([-.125, .125], [-.03125, .03125],
        [-1.25, 1.25], [-.25, .25], [-.78125, -.03125, .03125, .78125], [0])), dtype=np.float32)
    half_labels = np.stack([POLICIES[kind](halves) for kind in KINDS], axis=1)
    rounding = [row.tolist() + list(map(int, label)) for row, label in zip(halves, half_labels)]

    paths = contact_paths()
    path_hashes = [hashlib.sha256(paths[ticks, :, :, 3].tobytes()).hexdigest() for ticks in range(1, 67)]
    pairs = list(itertools.permutations(KINDS, 2))
    matches = [match(seed, pair) for seed in (24, 902) for pair in pairs]
    matches += [match(24, (kind, kind)) for kind in KINDS]
    sources = ['training/scripts/pace_study.py', 'training/scripts/offense_eval.py',
               'training/scripts/env.py', 'training/v4/reference.py']
    data = dict(config='pace13-v9-scaled', kinds=KINDS,
        sources={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
        contact_path_sha256=path_hashes, probes=probes, rounding=rounding, matches=matches)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    print(f'Wrote {OUTPUT.relative_to(ROOT)}: {len(probes)} physical probes, '
          f'{len(rounding)} rounding probes, {len(matches)} matches, '
          f'{sum(len(row["actions"]) for row in matches)} match decisions.')


if __name__ == '__main__':
    main()
