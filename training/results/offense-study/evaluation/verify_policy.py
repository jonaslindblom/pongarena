"""Reproduce the independent movement-plan and catchability checks."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'training/scripts'))
from offense_eval import CONFIGS, NEXT, OffenseArena, contact_paths, make_policy, np, reflect

started = time.monotonic()
first = contact_paths()
truth = np.zeros((67, 131, 131, 4), bool)
for p in range(131):
    for action in range(3):
        q = NEXT[p, action] - 15
        movement = int(np.sign(q - p)) + 1
        truth[1, p, q, movement] = True
        truth[1, p, q, 3] = True
for ticks in range(2, 67):
    for action in range(3):
        truth[ticks] |= truth[ticks - 1, NEXT[:, action] - 15]
assert np.array_equal(first[1:] >= 0, truth[1:])
checks = 0
for ticks in range(2, 67):
    p, q, movement = np.where(first[ticks] >= 0)
    action = first[ticks, p, q, movement]
    assert truth[ticks - 1, NEXT[p, action] - 15, q, movement].all()
    checks += len(p)
result = dict(all_contact_paths_verified=True, reachable_first_action_cases=checks,
              all_feasibility_entries=int(truth[1:].size), variants={},
              source_sha256=hashlib.sha256((ROOT / 'training/scripts/offense_eval.py').read_bytes()).hexdigest())
rng = np.random.default_rng(20261119)
for name, rules in CONFIGS.items():
    n = 12000
    y = rng.integers(4, 157, n)
    vy = rng.integers(-rules.vertical_max - rules.deflection, rules.vertical_max + rules.deflection + 1, n)
    vx = rng.integers(5, 14, n)
    ticks = rng.integers(1, 27, n)
    landing = reflect(y + ticks * vy).astype(int)
    own = rng.integers(15, 146, n)
    possible = np.abs(landing - own) <= 15 + 2 * ticks
    for kind in ['attack', 'defend']:
        ids = np.flatnonzero(possible)
        env = OffenseArena(len(ids), rules)
        env.ball[:] = np.column_stack([15 + ticks[ids] * vx[ids], y[ids], -vx[ids], vy[ids]])
        env.paddles[:, 0] = own[ids]
        policy = make_policy(rules, kind)
        done = np.zeros(len(ids), bool)
        misses, returns = 0, 0
        for _ in range(26):
            selected = np.column_stack((policy(env.obs()), np.ones(len(ids), int)))
            terminal, hit = env.step(selected, ~done)
            misses += int(((terminal == 1) & ~done).sum())
            returns += int(((hit == 0) & ~done).sum())
            done |= (terminal >= 0) | (hit >= 0)
        assert done.all() and misses == 0, (name, kind, misses)
        result['variants'][name + ':' + kind] = dict(catchable_incoming_cases=len(ids), caught=returns, misses=misses)
result['seconds'] = time.monotonic() - started
Path(__file__).with_name('policy-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
