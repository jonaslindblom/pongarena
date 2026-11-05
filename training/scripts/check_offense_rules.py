"""Independent scalar checks and portable action fixtures for candidate rules.

This is experiment verification, not a revision of the frozen v3 package.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from env import Arena
from offense_rules import CONFIGS, OffenseArena, shot
from reference import advance, observation, serve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'training/results/offense-study/verification'


def scalar_step(before, actions, rules, seen=None):
    """Separate scalar implementation; deliberately does not call shot()."""
    f = dict(before, paddles=list(before['paddles']), points=list(before['points']))
    f.pop('actor', None)
    f.pop('drawReason', None)
    for side in (0, 1):
        f['paddles'][side] = max(15, min(145, before['paddles'][side] + 2 * (actions[side] - 1)))
    f.update(x=before['x'] + before['vx'], y=before['y'] + before['vy'],
             tick=before['tick'] + 1, rallyTick=before['rallyTick'] + 1, event=None)
    if f['y'] < 4:
        f['y'], f['vy'] = 8 - f['y'], -f['vy']
    elif f['y'] > 156:
        f['y'], f['vy'] = 312 - f['y'], -f['vy']
    side = 0 if f['vx'] < 0 else 1
    contact = (side == 0 and f['x'] <= 15) or (side == 1 and f['x'] >= 345)
    if contact:
        d = f['y'] - f['paddles'][side]
        if abs(d) > 15:
            f.update(event='point', actor=1 - side)
            f['points'][1 - side] += 1
        else:
            dy = f['paddles'][side] - before['paddles'][side]
            movement = 1 if dy > 0 else -1 if dy < 0 else 0
            f.update(x=15 if side == 0 else 345,
                     vx=(1 if side == 0 else -1) * (5 + math.floor(8 * abs(d) / 15 + .5)),
                     vy=math.floor(rules.vertical_max * d / 15 + .5) + rules.deflection * movement,
                     event='return', actor=side, returns=before['returns'] + 1)
    if seen is not None and f['event'] == 'return':
        key = (f['x'], f['y'], f['vx'], f['vy'], *f['paddles'])
        if key in seen:
            f.update(event='draw', drawReason='loop')
            f.pop('actor', None)
        seen.add(key)
    if f['rallyTick'] >= 1600 and f['event'] not in ('point', 'draw'):
        f.update(event='draw', drawReason='timeout')
        f.pop('actor', None)
    f['scores'] = [[int(i == a) for i in range(3)] for a in actions]
    return f


def compare(env, frames, selected, seen=None):
    env.load(frames)
    if seen is not None:
        env.seen = [set(s) for s in seen]
    expected = [scalar_step(f, a, env.rules, set(s) if s is not None else None)
                for f, a, s in zip(frames, selected, seen if seen is not None else [None] * len(frames))]
    for side in (0, 1):
        np.testing.assert_array_equal(env.obs(side), np.array([observation(f, side) for f in frames], np.float32))
    out, hit = env.step(np.asarray(selected))
    np.testing.assert_array_equal(env.ball, [[f[k] for k in ('x', 'y', 'vx', 'vy')] for f in expected])
    np.testing.assert_array_equal(env.paddles, [f['paddles'] for f in expected])
    np.testing.assert_array_equal(env.ticks, [f['rallyTick'] for f in expected])
    np.testing.assert_array_equal(env.returns, [f['returns'] for f in expected])
    np.testing.assert_array_equal(out, [f['actor'] if f['event'] == 'point' else 2 if f['event'] == 'draw' else -1 for f in expected])
    np.testing.assert_array_equal(hit, [(0 if f['vx'] > 0 else 1) if f['returns'] > old['returns'] else -1
                                      for old, f in zip(frames, expected)])
    np.testing.assert_array_equal(env.draw_reason, [1 if f.get('drawReason') == 'loop' else 2 if f.get('drawReason') == 'timeout' else 0 for f in expected])
    return expected


def exhaustive_contacts():
    # Every legal integer pre-contact paddle center, contact height, and own
    # action, for both sides. This includes all edge hits and adjacent misses.
    p, y, action, side = np.meshgrid(np.arange(15, 146), np.arange(4, 157), np.arange(3), np.arange(2), indexing='ij')
    p, y, action, side = [a.ravel() for a in (p, y, action, side)]
    n = len(p)
    selected = np.ones((n, 2), int)
    selected[np.arange(n), side] = action
    frame = serve(1)
    frames = [dict(frame, x=28 if s == 0 else 332, y=int(h), vx=-13 if s == 0 else 13, vy=0,
                   paddles=[int(center), 80] if s == 0 else [80, int(center)]) for center, h, s in zip(p, y, side)]
    baseline = Arena(n)
    baseline.load(frames)
    expected_out, expected_hit = baseline.step(selected)
    for name, rules in CONFIGS.items():
        env = OffenseArena(n, rules)
        compare(env, frames, selected)
        if name == 'baseline':
            np.testing.assert_array_equal(env.ball, baseline.ball)
            np.testing.assert_array_equal(env.paddles, baseline.paddles)
            env.load(frames)
            out, hit = env.step(selected)
            np.testing.assert_array_equal(out, expected_out)
            np.testing.assert_array_equal(hit, expected_hit)
    return n


def edge_cases():
    base = dict(serve(7), x=17, y=80, vx=-5, vy=0)
    cases = [
        ('negative_edge', dict(base, y=67), [2, 1]),
        ('positive_edge', dict(base, y=93), [0, 1]),
        ('partial_top_clamp', dict(base, y=15, paddles=[16, 80]), [0, 1]),
        ('full_top_clamp', dict(base, y=15, paddles=[15, 80]), [0, 1]),
        ('partial_bottom_clamp', dict(base, y=145, paddles=[144, 80]), [2, 1]),
        ('full_bottom_clamp', dict(base, y=145, paddles=[145, 80]), [2, 1]),
        ('wall_before_contact', dict(base, y=5, vy=-4, paddles=[15, 80]), [1, 1]),
        ('wall_equality', dict(base, x=100, y=8, vy=-4), [1, 1]),
        ('point_at_limit', dict(base, y=130, rallyTick=1599), [1, 1]),
        ('return_at_limit', dict(base, rallyTick=1599), [1, 1]),
        ('loop_at_limit', dict(base, rallyTick=1599), [1, 1]),
    ]
    result = []
    for name, rules in CONFIGS.items():
        env = OffenseArena(len(cases), rules, loop_detection=True)
        seen = [set() for _ in cases]
        seen[-1].add((15, 80, 5, 0, 80, 80))
        after = compare(env, [f for _, f, _ in cases], [a for _, _, a in cases], seen)
        assert after[-3]['event'] == 'point'
        assert after[-2]['drawReason'] == 'timeout'
        assert after[-1]['drawReason'] == 'loop'
        assert after[7]['y'] == 4 and after[7]['vy'] == -4
        if name == 'motion2':
            assert [f['vy'] for f in after[2:6]] == [-2, 0, 2, 0]
        result.extend(dict(rules=name, name=title, before=f, actions=a, after=g,
                           seen_before=[list(k) for k in previous])
                      for (title, f, a), g, previous in zip(cases, after, seen))
    # Rounding is specified as floor(x + .5), including negative half ties;
    # integer reachable offsets have no half ties because the denominator is 15.
    assert shot(-1.5, 0, CONFIGS['baseline'])[1] == 0
    assert shot(1.5, 0, CONFIGS['baseline'])[1] == 1
    return result


def trajectories():
    rng = np.random.default_rng(6712)
    count = 0
    fixtures = []
    for name, rules in CONFIGS.items():
        frames = [serve(int(seed)) for seed in rng.integers(0, 2**32, 32)]
        env = OffenseArena(32, rules)
        for _ in range(400):
            selected = rng.integers(0, 3, (32, 2))
            expected = compare(env, frames, selected)
            if name == 'baseline':
                frozen = [advance(f, [[int(i == a) for i in range(3)] for a in pair]) for f, pair in zip(frames, selected)]
                assert expected == frozen
            count += len(frames)
            frames = [serve(int(rng.integers(0, 2**32))) if f['event'] in ('point', 'draw') else f for f in expected]
        for seed in (11, 2026, 4294967295):
            f = serve(seed)
            trace = dict(rules=name, seed=seed, loop_detection=True, initial=f, steps=[])
            seen = set()
            for _ in range(1600):
                # Generate reproducible contacts, with occasional direction
                # changes. Consumers replay the stored actions, not this policy.
                ticks = math.ceil(abs((15 if f['vx'] < 0 else 345) - f['x']) / abs(f['vx']))
                z = (f['y'] + ticks * f['vy'] - 4) % 304
                landing = 4 + (z if z <= 152 else 304 - z)
                selected = []
                for side in (0, 1):
                    target = landing if side == int(f['vx'] > 0) else 80
                    error = target - f['paddles'][side]
                    selected.append(0 if error < -1 else 2 if error > 1 else 1)
                f = scalar_step(f, selected, rules, seen)
                trace['steps'].append(dict(actions=selected, after=f))
                if f['event'] in ('point', 'draw'):
                    break
            fixtures.append(trace)
    return count, fixtures


def mask_and_symmetry():
    rng = np.random.default_rng(941)
    checks = 0
    for rules in CONFIGS.values():
        env = OffenseArena(512, rules, loop_detection=True)
        mirrored = OffenseArena(512, rules, loop_detection=True)
        env.ball[:, 0] = np.where(np.arange(512) % 2, 17, 343)
        env.ball[:, 1] = rng.integers(4, 157, 512)
        env.ball[:, 2] = np.where(np.arange(512) % 2, -5, 5)
        env.ball[:, 3] = rng.integers(-rules.vertical_max-rules.deflection, rules.vertical_max+rules.deflection+1, 512)
        env.paddles[:] = rng.integers(15, 146, (512, 2))
        mirrored.ball[:] = env.ball
        mirrored.ball[:, 1] = 160 - env.ball[:, 1]
        mirrored.ball[:, 3] = -env.ball[:, 3]
        mirrored.paddles[:] = 160 - env.paddles
        selected = rng.integers(0, 3, (512, 2))
        active = rng.random(512) > .25
        old_ball, old_paddles, old_ticks = env.ball.copy(), env.paddles.copy(), env.ticks.copy()
        out, hit = env.step(selected, active)
        out_m, hit_m = mirrored.step(2 - selected, active)
        np.testing.assert_array_equal(out, out_m)
        np.testing.assert_array_equal(hit, hit_m)
        np.testing.assert_array_equal(env.ball[:, [0, 2]], mirrored.ball[:, [0, 2]])
        np.testing.assert_array_equal(env.ball[:, 1], 160 - mirrored.ball[:, 1])
        np.testing.assert_array_equal(env.ball[:, 3], -mirrored.ball[:, 3])
        np.testing.assert_array_equal(env.paddles, 160 - mirrored.paddles)
        np.testing.assert_array_equal(env.ball[~active], old_ball[~active])
        np.testing.assert_array_equal(env.paddles[~active], old_paddles[~active])
        np.testing.assert_array_equal(env.ticks[~active], old_ticks[~active])
        assert (out[~active] == -1).all() and (hit[~active] == -1).all()
        checks += len(active)
    return checks


def main():
    contacts = exhaustive_contacts()
    cases = edge_cases()
    trajectory_count, traces = trajectories()
    mask_count = mask_and_symmetry()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ['training/scripts/offense_rules.py', 'training/scripts/check_offense_rules.py',
               'training/scripts/env.py', 'training/v3/reference.py', 'src/game.ts']
    result = dict(status='PASS', configs={k: asdict(v) for k, v in CONFIGS.items()},
                  exhaustive_contact_transitions_per_config=contacts,
                  random_trajectory_transitions=trajectory_count, masked_symmetry_transitions=mask_count,
                  baseline_frozen_scalar_transitions=trajectory_count // len(CONFIGS), edge_cases=len(cases),
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources})
    (OUT / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'fixtures.json').write_text(json.dumps(dict(format='offense-study-fixtures-1',
                                                     configs=result['configs'], cases=cases, trajectories=traces), separators=(',', ':')) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
