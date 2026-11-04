"""Verify batched physics on every published transition and random complete matches."""
import json
from pathlib import Path
import numpy as np
from env import Arena
from reference import advance, serve, observation, ended_rally


def verify(env, frames, actions):
    env.load(frames)
    for side in (0, 1):
        np.testing.assert_array_equal(env.obs(side), np.array([observation(f, side) for f in frames], dtype=np.float32))
    out, hit = env.step(actions)
    expected = [advance(f, [[int(a == i) for i in range(3)] for a in pair]) for f, pair in zip(frames, actions)]
    np.testing.assert_array_equal(env.ball, [[f[k] for k in ('x', 'y', 'vx', 'vy')] for f in expected])
    np.testing.assert_array_equal(env.paddles, [f['paddles'] for f in expected])
    np.testing.assert_array_equal(env.ticks, [f['rallyTick'] for f in expected])
    np.testing.assert_array_equal(env.returns, [f['returns'] for f in expected])
    np.testing.assert_array_equal(out, [f['actor'] if f['event'] == 'point' else 2 if f['event'] == 'draw' else -1 for f in expected])
    return expected


def main():
    fixtures = json.loads((Path(__file__).resolve().parents[1] / 'v1/fixtures.json').read_text())
    frames, actions = [], []
    for match in fixtures['matches']:
        prev = match['initial']['state']
        for row in match['steps']:
            if row['actions'] is not None:
                frames.append(prev)
                actions.append(row['actions'])
            prev = row['state']
    for case in fixtures['cases']:
        frames.append(case['before'])
        actions.append([int(np.argmax(s)) for s in case['scores']])
    verify(Arena(len(frames)), frames, np.array(actions))
    rng = np.random.default_rng(4815)
    env = Arena(32)
    states = [serve(int(s)) for s in rng.integers(0, 2**32, 32)]
    for _ in range(1000):
        actions = rng.integers(0, 3, (32, 2))
        states = verify(env, states, actions)
        states = [serve(int(rng.integers(0, 2**32))) if ended_rally(f) else f for f in states]
    print(f'PASS: {len(frames)} fixture transitions + 32,000 randomized transitions, including float32 observations.')


if __name__ == '__main__':
    main()
