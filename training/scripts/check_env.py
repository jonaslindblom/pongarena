"""Verify batched physics on every published transition and random complete matches."""
import json
from pathlib import Path
import numpy as np
from env import Arena, scripted
from reference import advance, serve, observation, ended_rally, LoopDetector


def compare(env, frames, out):
    np.testing.assert_array_equal(env.ball, [[f[k] for k in ('x', 'y', 'vx', 'vy')] for f in frames])
    np.testing.assert_array_equal(env.paddles, [f['paddles'] for f in frames])
    np.testing.assert_array_equal(env.ticks, [f['rallyTick'] for f in frames])
    np.testing.assert_array_equal(env.returns, [f['returns'] for f in frames])
    np.testing.assert_array_equal(out, [f['actor'] if f['event'] == 'point' else 2 if f['event'] == 'draw' else -1 for f in frames])
    np.testing.assert_array_equal(env.draw_reason, [1 if f.get('drawReason') == 'loop' else 2 if f.get('drawReason') == 'timeout' else 0 for f in frames])


def verify(env, frames, actions):
    env.load(frames)
    for side in (0, 1):
        np.testing.assert_array_equal(env.obs(side), np.array([observation(f, side) for f in frames], dtype=np.float32))
    out, hit = env.step(actions)
    expected = [advance(f, [[int(a == i) for i in range(3)] for a in pair]) for f, pair in zip(frames, actions)]
    compare(env, expected, out)
    return expected


def check_deterministic_and_sampled_modes():
    # A centered horizontal exchange repeats exactly. Frozen policies may end it;
    # stochastic training must retain the opportunity to choose a different move.
    initial = dict(serve(24), x=180, y=80, vx=-5, vy=0)
    deterministic = Arena(1, loop_detection=True)
    sampled = Arena(1)
    deterministic.load([initial])
    sampled.load([initial])
    frame, loops = initial, LoopDetector()
    still = np.ones((1, 2), dtype=int)
    for _ in range(1600):
        result, _ = deterministic.step(still)
        frame = advance(frame, [[0, 1, 0], [0, 1, 0]], loops)
        compare(deterministic, [frame], result)
        if result[0] >= 0:
            break
    assert frame['drawReason'] == 'loop' and frame['rallyTick'] < 800
    for _ in range(1600):
        result, _ = sampled.step(still)
        if sampled.ticks[0] < 1600:
            assert result[0] == -1
    assert result[0] == 2 and sampled.draw_reason[0] == 2
    assert not sampled.seen[0]
    deterministic.load([initial])
    assert not deterministic.seen[0]
    deterministic.seen[0].add((1, 2, 3, 4, 5, 6))
    deterministic.reset(np.array([0]))
    assert not deterministic.seen[0]

    # Completed vector lanes must stay frozen while a longer rally continues.
    env = Arena(2, loop_detection=True)
    env.load([initial, initial])
    before = (env.ball[0].copy(), env.paddles[0].copy(), env.ticks[0], env.returns[0])
    for _ in range(50):
        env.step(np.zeros((2, 2), int), np.array([False, True]))
    np.testing.assert_array_equal(env.ball[0], before[0])
    np.testing.assert_array_equal(env.paddles[0], before[1])
    assert env.ticks[0] == before[2] and env.returns[0] == before[3] and not env.seen[0]
    print('PASS: exact loop vs stochastic continuation, history reset, and frozen completed lanes.')


def check_deterministic_matches():
    env = Arena(32, seed=730, loop_detection=True)
    frames = [serve(int(seed)) for seed in env.seeds]
    loops = [LoopDetector() for _ in frames]
    env.load(frames)
    count = 0
    for _ in range(1600):
        active = np.array([not ended_rally(f) for f in frames])
        if not active.any():
            break
        selected = np.stack([scripted(env.obs(), attack=True), scripted(env.obs(1))], 1)
        out, _ = env.step(selected, active)
        for i in np.flatnonzero(active):
            frames[i] = advance(frames[i], [[int(a == k) for k in range(3)] for a in selected[i]], loops[i])
            count += 1
        expected_out = np.array([f['actor'] if f['event'] == 'point' else 2 if f['event'] == 'draw' else -1 for f in frames])
        expected_out[~active] = -1
        np.testing.assert_array_equal(out, expected_out)
        np.testing.assert_array_equal(env.ball, [[f[k] for k in ('x', 'y', 'vx', 'vy')] for f in frames])
        np.testing.assert_array_equal(env.paddles, [f['paddles'] for f in frames])
        np.testing.assert_array_equal(env.draw_reason[active], [1 if frames[i].get('drawReason') == 'loop' else 2 if frames[i].get('drawReason') == 'timeout' else 0 for i in np.flatnonzero(active)])
    assert all(ended_rally(f) for f in frames)
    assert any(f.get('drawReason') == 'loop' for f in frames)
    print(f'PASS: {count:,} deterministic transitions against independent scalar loop detection.')


def check_published_sequences(fixtures):
    count = 0
    for sequence in fixtures['matches'] + fixtures['traces']:
        env = Arena(1, loop_detection=sequence.get('loopDetection', False))
        env.load([sequence['initial']['state']])
        for row in sequence['steps']:
            if row['actions'] is None:
                env.load([row['state']])
            else:
                out, _ = env.step(np.array([row['actions']]))
                compare(env, [row['state']], out)
                count += 1
            for side in (0, 1):
                np.testing.assert_array_equal(env.obs(side)[0], np.array(row['observations'][side], dtype=np.float32))
    for case in fixtures['cases']:
        env = Arena(1, loop_detection=case.get('loopDetection', False))
        env.load([case['before']])
        for f in case.get('seenContacts', []):
            if f['rally'] == case['before']['rally'] and f['event'] == 'return':
                env.seen[0].add(tuple(f[k] for k in ('x', 'y', 'vx', 'vy')) + tuple(f['paddles']))
        out, _ = env.step(np.array([[int(np.argmax(s)) for s in case['scores']]]))
        compare(env, [case['expected']['state']], out)
    print(f'PASS: {count:,} complete published action transitions and {len(fixtures["cases"])} boundary cases, including loop histories.')


def main():
    fixtures = json.loads((Path(__file__).resolve().parents[1] / 'v3/fixtures.json').read_text())
    frames, actions = [], []
    for match in fixtures['matches']:
        if match.get('loopDetection'):
            continue
        prev = match['initial']['state']
        for row in match['steps']:
            if row['actions'] is not None:
                frames.append(prev)
                actions.append(row['actions'])
            prev = row['state']
    for case in fixtures['cases']:
        if case.get('loopDetection'):
            continue
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
    check_published_sequences(fixtures)
    check_deterministic_and_sampled_modes()
    check_deterministic_matches()


if __name__ == '__main__':
    main()
