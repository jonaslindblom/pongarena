"""Check that experimental measurements preserve v2 and never invent draws.

Run with the training virtual environment. No game, model or result files change.
"""
import json

import numpy as np
import torch

from check_env import verify
from env import scripted
from policy import ROOT, OnnxPolicy, actions
from reference import ended_rally, serve
from rule_study import CONFIGS, Rules, StudyArena, load_policies, run, script, shot, summarize


class DeadlineArena(StudyArena):
    """Restore v2's deadline only when comparing with the frozen reference."""
    def step(self, selected):
        out, hit = super().step(selected)
        return np.where((out < 0) & (self.ticks >= 800), 2, out), hit


def check_reference_and_scripts():
    fixtures = json.loads((ROOT / 'training/v2/fixtures.json').read_text())
    frames, selected = [], []
    for match in fixtures['matches']:
        previous = match['initial']['state']
        for row in match['steps']:
            if row['actions'] is not None:
                frames.append(previous)
                selected.append(row['actions'])
            previous = row['state']
    for case in fixtures['cases']:
        frames.append(case['before'])
        selected.append([int(np.argmax(scores)) for scores in case['scores']])
    verify(DeadlineArena(len(frames), Rules()), frames, np.array(selected))

    rng = np.random.default_rng(4815)
    env = DeadlineArena(32, Rules())
    frames = [serve(int(seed)) for seed in rng.integers(0, 2**32, 32)]
    decisions = 0
    for _ in range(1000):
        env.load(frames)
        for side in (0, 1):
            obs = env.obs(side)
            for attack in (False, True):
                np.testing.assert_array_equal(script(obs, Rules(), attack), scripted(obs, attack))
                decisions += len(frames)
        frames = verify(env, frames, rng.integers(0, 3, (32, 2)))
        frames = [serve(int(rng.integers(0, 2**32))) if ended_rally(f) else f for f in frames]
    print(f'PASS: {len(selected)} fixture + 32,000 random transitions; {decisions:,} baseline script decisions.')


def check_deadline():
    # A point on the exact deadline must not become a draw; a return cannot win.
    start = dict(serve(24), rallyTick=799, x=17, y=30, vx=-5, vy=0)
    env = StudyArena(1, Rules())
    env.load([start])
    out, _ = env.step(np.array([[1, 1]]))
    assert out.tolist() == [1] and env.ticks.tolist() == [800]
    record = dict(winner=[1] * 10, point_tick=[800] * 10, point_returns=[0] * 10,
                  cycle_tick=[0] * 10)
    sides, receiver = np.zeros(10, int), np.ones(10, int)
    assert summarize(record, 799, sides, receiver)['draws'] == 10
    assert summarize(record, 800, sides, receiver)['losses'] == 10
    env.load([dict(start, y=80)])
    out, hit = env.step(np.array([[1, 1]]))
    assert out.tolist() == [-1] and hit.tolist() == [0]
    record.update(winner=[-1] * 10, point_tick=[0] * 10, point_returns=[0] * 10)
    assert summarize(record, 800, sides, receiver)['draws'] == 10
    # Continuing an unresolved rally is the whole point of the deadline study.
    assert env.step(np.array([[1, 1]]))[0].tolist() == [-1]
    print('PASS: exact-deadline points count; last-tick returns draw; physics can continue past 800.')


def check_rounding_and_mirror():
    integer_offsets = np.arange(-15, 16, dtype=float)
    np.testing.assert_array_equal(shot(integer_offsets, Rules())[1],
                                  np.floor(integer_offsets / 3 + .5))
    np.testing.assert_array_equal(shot(-integer_offsets, Rules())[1],
                                  -shot(integer_offsets, Rules())[1])
    # Fractional speeds introduce reachable half ties: explicitly preserve and
    # expose the difference instead of silently correcting the candidate rule.
    assert shot(np.array([-1.5, 1.5]), Rules())[1].tolist() == [0, 1]
    assert shot(np.array([-1.5, 1.5]), Rules(symmetric_rounding=True))[1].tolist() == [-1, 1]

    rng = np.random.default_rng(941)
    for name in ['v2', 'paddle2.25-symmetric', 'paddle2.5-symmetric', 'paddle2.25-cap14']:
        rules = CONFIGS[name]
        n = 512
        env, mirrored = StudyArena(n, rules), StudyArena(n, rules)
        frames = []
        for index in range(n):
            f = serve(index)
            f.update(x=20 if index % 2 else 340, vx=-5 if index % 2 else 5,
                     y=int(rng.integers(4, 157)), vy=int(rng.integers(-5, 6)))
            grid = 1 if name == 'v2' else 4
            f['paddles'] = (rng.integers(15 * grid, 145 * grid + 1, 2) / grid).tolist()
            frames.append(f)
        reflected = [dict(f, y=160-f['y'], vy=-f['vy'], paddles=[160-y for y in f['paddles']])
                     for f in frames]
        env.load(frames)
        mirrored.load(reflected)
        selected = rng.integers(0, 3, (n, 2))
        out, hit = env.step(selected)
        mirrored_out, mirrored_hit = mirrored.step(2-selected)
        np.testing.assert_array_equal(out, mirrored_out)
        np.testing.assert_array_equal(hit, mirrored_hit)
        expected_ball = env.ball.copy()
        expected_ball[:, 1] = 160 - expected_ball[:, 1]
        expected_ball[:, 3] *= -1
        np.testing.assert_array_equal(mirrored.ball, expected_ball)
        np.testing.assert_array_equal(mirrored.paddles, 160-env.paddles)
    print('PASS: integer v2 and explicitly symmetric fractional rules preserve vertical reflection; ordinary half ties do not.')


def check_baseline_results(policies):
    mirror = json.loads((ROOT / 'training/results/v2/matchups.json').read_text())
    opponents = json.loads((ROOT / 'training/results/v2/selfplay-test.json').read_text())
    seeds = np.arange(2_100_000_000, 2_100_000_100)
    for pair, expected in [('NN', mirror['opponents']['selfplay']),
                           ('NA', opponents['opponents']['attacker'])]:
        rows, _ = run(Rules(), pair, seeds, policies, [800])
        row = rows[0]
        for key in ['wins', 'losses', 'draws', 'match_wins', 'match_losses', 'match_draws',
                    'median_decisive_returns', 'mean_ticks']:
            assert row[key] == expected[key], (pair, key, row[key], expected[key])
        assert row['early_rate'] == expected['early_finish_rate']
    print('PASS: baseline NN/NA summary metrics reproduce the released evaluation across 2,000 rallies.')


def check_cycle_shortcuts(policies):
    cycle_count = 0
    for config, pair in [('v2', 'NN'), ('v2', 'DD'), ('v2', 'AH'), ('paddle2.25-symmetric', 'AH')]:
        args = (CONFIGS[config], pair, [24, 902, 3_100_000_041], policies, [800, 1600, 3200])
        shortcut_rows, shortcut = run(*args)
        full_rows, full = run(*args, detect_cycles=False)
        for key in ['winner', 'point_tick', 'point_returns']:
            np.testing.assert_array_equal(shortcut[key], full[key], err_msg=f'{config} {pair} {key}')
        for a, b in zip(shortcut_rows, full_rows):
            for key in ['wins', 'losses', 'draws', 'mean_ticks', 'median_decisive_returns',
                        'match_wins', 'match_losses', 'match_draws']:
                assert a[key] == b[key], (config, pair, key, a[key], b[key])
        cycles = np.array(shortcut['cycle_tick']) > 0
        cycle_count += int(cycles.sum())
        assert np.all(np.array(shortcut['cycle_period'])[cycles] > 0)
        assert np.all(np.array(full['winner'])[cycles] == -1)
        np.testing.assert_array_equal(np.array(shortcut['cycle_tick'])[cycles],
                                      (np.array(shortcut['cycle_start']) + np.array(shortcut['cycle_period']))[cycles])
    assert cycle_count > 0, 'The test must exercise actual cycle exits, not just identical straight-line runs.'
    print(f'PASS: {cycle_count} cycle shortcuts agree with full 3,200-tick simulations, including delayed-policy memory.')


def check_exported_actions(policies):
    # Probe real played states, including fractional paddle positions that never
    # appeared in v2 training. Neither synthetic feature noise nor unbounded
    # post-point states can establish parity for these experimental trajectories.
    batches = []
    for name in ['v2', 'paddle2.25-symmetric', 'paddle2.5-symmetric', 'larger-paddle']:
        rules = CONFIGS[name]
        env = StudyArena(64, rules)
        lanes = np.arange(env.n) % 4
        for tick in range(256):
            left, right = env.obs(), env.obs(1)
            if tick % 16 == 0:
                batches.extend([left.copy(), right.copy()])
            a = np.where(lanes == 3, script(left, rules, True), actions(policies['N'], left))
            b = actions(policies['P'], right)
            b = np.where(lanes == 0, actions(policies['N'], right), b)
            b = np.where(lanes == 2, script(right, rules, True), b)
            b = np.where(lanes == 3, script(right, rules), b)
            out, _ = env.step(np.stack([a, b], axis=1))
            env.reset(np.flatnonzero(out >= 0))
    obs = np.concatenate(batches)
    assert obs.shape == (8192, 6)
    for code, name in [('N', 'selfplay-v2'), ('P', 'apprentice-v2')]:
        with torch.no_grad():
            expected = policies[code](torch.from_numpy(obs)).numpy()
        exported = OnnxPolicy(ROOT / f'public/models/{name}.onnx')
        actual = np.array([exported.session.run(None, {'observation': row[None]})[0][0] for row in obs])
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1), err_msg=name)
        error = float(np.max(np.abs(actual-expected)))
        print(f'PASS: {name}: 8,192 reachable observations, 100% PyTorch/native ONNX action agreement; max score error {error:.9g}.')


def main():
    torch.set_num_threads(1)
    check_reference_and_scripts()
    check_deadline()
    check_rounding_and_mirror()
    policies = load_policies()
    check_baseline_results(policies)
    check_cycle_shortcuts(policies)
    check_exported_actions(policies)


if __name__ == '__main__':
    main()
