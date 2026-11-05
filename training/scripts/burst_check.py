"""Independent scalar/vector verification and reproducible prototype fixtures."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from burst_rules import BurstArena, DEFAULT, make_policy, travel_table

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('burst_reference', ROOT/'training/burst/reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def scores(actions):
    return [[int(action == j) for j in range(5)] for action in actions]


def check_frame(env, index, expected):
    assert env.ball[index].tolist() == [expected[k] for k in ('x', 'y', 'vx', 'vy')]
    assert env.paddles[index].tolist() == expected['paddles']
    assert env.energy[index].tolist() == expected['energy']
    assert int(env.returns[index]) == expected['returns']
    assert int(env.ticks[index]) == expected['rallyTick']
    for side in (0, 1):
        assert env.obs(side)[index].tolist() == ref.observation(expected, side)


def main():
    rng = np.random.default_rng(20261212)
    frames, choices = [], rng.integers(0, 5, (12000, 2))
    for i in range(len(choices)):
        f = ref.serve(i)
        f.update(x=float(rng.integers(30, 691)/2), y=int(rng.integers(4, 157)),
            vx=float(rng.integers(10, 27)/2*rng.choice([-1, 1])), vy=int(rng.integers(-9, 10)),
            paddles=rng.integers(15, 146, 2).tolist(), energy=rng.integers(0, 97, 2).tolist(),
            rallyTick=int(rng.choice([0, 100, 1599])))
        frames.append(f)
    env = BurstArena(len(frames))
    env.load(frames)
    outcome, hit = env.step(choices)
    boundaries = []
    for i, f in enumerate(frames):
        expected = ref.advance(f, scores(choices[i]))
        check_frame(env, i, expected)
        terminal = 2 if expected['event'] == 'draw' else expected['actor'] if expected['event'] == 'point' else -1
        assert int(outcome[i]) == terminal
        if i < 100:
            boundaries.append(dict(before=f, actions=choices[i].tolist(), after=expected))

    # Resource is part of the physical Markov state, including exact loops.
    loop = ref.LoopDetector()
    f = ref.serve(11)
    f.update(event='return', x=15, vx=13, vy=0)
    assert not loop.observe(f)
    g = dict(f, energy=[95, 96])
    assert not loop.observe(g)
    assert loop.observe(g)
    # Depleted requests move normally, but cannot secretly recharge.
    f = ref.serve(11)
    f['energy'] = [0, 7]
    g = ref.advance(f, scores([3, 4]))
    assert g['paddles'] == [78, 82] and g['energy'] == [0, 7]
    h = ref.advance(g, scores([0, 1]))
    assert h['energy'] == [1, 8]
    boundaries.append(dict(before=f, actions=[3, 4], after=g))
    boundaries.append(dict(before=g, actions=[0, 1], after=h))

    matches, transitions = [], 0
    teacher, defender = make_policy(DEFAULT, 'attack'), make_policy(DEFAULT, 'defend')
    for seed in (11, 2026, 4294967295):
        f = ref.serve(seed)
        env = BurstArena(1, loop_detection=True)
        env.load([f])
        loop = ref.LoopDetector()
        match = dict(seed=seed, steps=[])
        while not (f['rally'] == 5 and f['event'] in ('point', 'draw')):
            if f['event'] in ('point', 'draw'):
                f = ref.serve(seed, f)
                env.load([f])
                match['steps'].append(dict(actions=None, state=f))
            else:
                action = [int(teacher(env.obs())[0]), int(defender(env.obs(1))[0])]
                f = ref.advance(f, scores(action), loop)
                terminal, _ = env.step(np.array([action]))
                check_frame(env, 0, f)
                expected = 2 if f['event'] == 'draw' else f['actor'] if f['event'] == 'point' else -1
                assert int(terminal[0]) == expected
                match['steps'].append(dict(actions=action, state=f))
                transitions += 1
        matches.append(match)
    fixtures = dict(rules='pong-burst-v1', config=vars(DEFAULT), boundaries=boundaries, matches=matches)
    (ROOT/'training/burst/fixtures.json').write_text(json.dumps(fixtures,separators=(',',':'))+'\n')
    result = dict(random_transitions=len(frames), full_match_transitions=transitions,
        matches=len(matches), boundaries=len(boundaries), scalar_vector_parity=True,
        observations_both_sides=True, energy_loop_key=True, exhausted_burst_rule=True)
    (ROOT/'training/results/burst-study/rules-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':main()
