"""Independent closure check using the evaluator's contact-action path tables.

Does not invoke the fixed-point solver. It consumes the saved proposed sets and
independently checks every universal incoming shot / existential own response.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'training/scripts'))
from offense_eval import CONFIGS, CENTERS, contact_paths, np, reflect, tables

started = time.monotonic()
source = ROOT / 'training/results/offense-study/verification/invariant.json'
sets = json.loads(source.read_text())
paths = contact_paths()
result = dict(method='independent universal closure check using exact first-action paths and separate shot tables',
              sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [source, ROOT / 'training/scripts/offense_eval.py', Path(__file__)]}, configs={})
for name in ['baseline', 'angle8']:
    rules = CONFIGS[name]
    tab = tables(rules)
    safe = np.zeros((157, 131), bool)
    for height, centers in sets['configs'][name]['safe_centers'].items():
        safe[int(height), np.array(centers) - 15] = True
    recoverable = np.zeros_like(tab['legal'])
    for y in range(4, 157):
        for option in np.flatnonzero(tab['legal'][y]):
            q = tab['centers'][y, option]
            future = tab['landing'][y, option]
            duration = tab['flight'][option]
            recoverable[y, option] = (safe[future] & (paths[duration, q - 15, :, 3] >= 0)).any()
    interception = np.zeros((67, 157, 131), bool)
    for duration in range(1, 67):
        for y in range(4, 157):
            opts = np.flatnonzero(recoverable[y])
            q = tab['centers'][y, opts] - 15
            m = tab['motion_index'][opts]
            if len(opts):
                interception[duration, y] = (paths[duration, np.arange(131)[:, None], q, m] >= 0).any(axis=1)
    checks, options = 0, 0
    for y in range(4, 157):
        for option in np.flatnonzero(tab['legal'][y]):
            flight, landing = tab['flight'][option], tab['landing'][y, option]
            retained = safe[y]
            assert interception[flight, landing, retained].all(), (name, y, option)
            checks += int(retained.sum())
            options += 1
    cases = 0
    for center in [32, 56, 80, 104, 128]:
        for y in range(center - 10, center + 11):
            for vy in [*range(-5, 0), *range(1, 6)]:
                landing = int(reflect(y + 33 * vy))
                assert interception[33, landing, 80 - 15], (name, y, vy, 'receiver')
                assert (safe[landing] & (paths[33, 80 - 15, :, 3] >= 0)).any(), (name, y, vy, 'nonreceiver')
                cases += 1
    result['configs'][name] = dict(closure_verified=True, retained_states=int(safe.sum()),
        legal_opponent_shots=options, safe_state_shot_checks=checks, standard_serve_cases_per_role=cases,
        guarantees='Every legal reply can be caught with a selected return permitting safe recovery; both standard serve roles enter the set.')
result['seconds'] = time.monotonic() - started
Path(__file__).with_name('invariant-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
