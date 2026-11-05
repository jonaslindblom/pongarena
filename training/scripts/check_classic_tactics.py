"""Independently replay Classic placement branches with the frozen scalar port.

Physics checks use training/v4/reference.py, not the vector training engine.
Diagnostic continuations retain the original rally clock and disable loops.
The displayed strict improvement is additionally checked for observation/action
consistency and normal stateless-loop termination from the original serve.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'training/v4/reference.py'
spec = importlib.util.spec_from_file_location('classic_scalar', REFERENCE)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
SCORES = [[int(i == a) for i in range(3)] for a in range(3)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actions(code):
    assert code in '012345678', f'Illegal joint action: {code}'
    value = int(code)
    return value//3, value%3


def state(frame):
    return dict(ball=[frame[k] for k in ('x', 'y', 'vx', 'vy')],
        paddles=frame['paddles'], tick=frame['rallyTick'], returns=frame['returns'])


def opening(match):
    frame = None
    for _ in range(match['rally']):
        frame = ref.serve(match['seed'], frame)
    # Probes record isolated rallies with a local clock, not whole-match ticks.
    frame['tick'] = 0
    assert state(frame) == match['opening']
    return frame


def advance(frame, code, loops=None):
    a, b = actions(code)
    assert not ref.ended_rally(frame), 'Recorded actions continue beyond termination'
    return ref.advance(frame, [SCORES[a], SCORES[b]], loops)


def terminal(frame, row):
    assert ref.ended_rally(frame), 'Continuation ended before a terminal event'
    assert state(frame) == row['terminal']
    outcome = 2 if frame['event'] == 'draw' else frame['actor']
    assert outcome == row['outcome']
    if outcome == 2:
        assert frame['drawReason'] == 'timeout' and frame['rallyTick'] == 1600
    assert frame['rallyTick'] <= 1600


def legal_targets(frame, side):
    """Independent movement BFS and scalar wall/flight stepping."""
    x, y, vx, vy = (frame[k] for k in ('x', 'y', 'vx', 'vy'))
    reachable = {frame['paddles'][side]}
    ticks = 0
    while True:
        ticks += 1
        reachable = {max(15, min(145, p+d)) for p in reachable for d in (-2, 0, 2)}
        x, y = x+vx, y+vy
        if y < 4 or y > 156:
            y, vy = (8-y if y < 4 else 312-y), -vy
        if (vx < 0 and x <= 15) or (vx > 0 and x >= 345):
            return ticks, sorted(p for p in reachable if abs(y-p) <= 15)


def observation_consistency(match, prefix, branch, side):
    frame, loops = opening(match), ref.LoopDetector()
    mapping = {}
    conflict = None
    early_loop = None
    encoded = prefix+branch['actions']
    for tick, code in enumerate(encoded):
        key = tuple(ref.observation(frame, side))
        action = actions(code)[side]
        if key in mapping and mapping[key] != action and conflict is None:
            conflict = dict(tick=tick, earlier_action=mapping[key], action=action)
        mapping[key] = action
        # Inspect normal loop behavior without allowing it to truncate the
        # independently recorded deadline-only physical trajectory.
        frame = advance(frame, code)
        if loops.observe(frame) and early_loop is None:
            early_loop = frame['rallyTick']
    return dict(unique_observations=len(mapping), action_conflict=conflict,
        first_standard_loop_tick=early_loop, full_prefix_ticks=len(prefix),
        total_ticks=len(encoded))


def verify(path, representative_only=False):
    started = time.monotonic()
    data = json.loads(path.read_text())
    assert data['rules'] == 'pong-v4'
    assert len(data['seeds']) == len(set(data['seeds']))
    matches = {m['id']:m for m in data['matches']}
    snapshots = {s['id']:s for s in data['snapshots']}
    groups = {key:[] for key in snapshots}
    for branch in data['candidates']:
        groups[branch['snapshot_id']].append(branch)
    snapshot_frames = {}
    prefix_steps = 0
    for match_id, match in matches.items():
        frame = opening(match)
        by_tick = {s['tick']:s for s in snapshots.values() if s['match_id'] == match_id}
        for code in match['actions']:
            if frame['rallyTick'] in by_tick:
                snapshot = by_tick[frame['rallyTick']]
                assert state(frame) == {k:snapshot[k] for k in ('ball', 'paddles', 'tick', 'returns')}
                assert snapshot['side'] == int(frame['vx'] > 0)
                snapshot_frames[snapshot['id']] = frame
            frame = advance(frame, code)
            prefix_steps += 1
        terminal(frame, match)
    assert len(snapshot_frames) == len(snapshots)
    checked, steps, selected, strict, consistent = 0, 0, 0, 0, 0
    representative = None
    for sid, snapshot in snapshots.items():
        branches = groups[sid]
        baselines = [b for b in branches if b['target'] is None]
        assert len(baselines) == 1
        baseline = baselines[0]
        t, targets = legal_targets(snapshot_frames[sid], snapshot['side'])
        assert sorted(b['target'] for b in branches if b['target'] is not None) == targets
        for branch in branches:
            assert branch['flight_ticks'] == t
            assert branch['value'] == (1 if branch['outcome'] == snapshot['side'] else 0 if branch['outcome'] == 2 else -1)
            improvement = baseline['value'] == -1 and branch['value'] == 1
            strict += improvement
            chosen = branch['target'] is not None and branch['target'] == snapshot.get('selected_target')
            selected += chosen
            if representative_only and not improvement:
                continue
            frame = snapshot_frames[sid]
            first_contact = -1
            for index, code in enumerate(branch['actions']):
                frame = advance(frame, code)
                steps += 1
                if index+1 == t:
                    first_contact = frame['paddles'][snapshot['side']]
                    if branch['target'] is not None:
                        assert frame['event'] == 'return' and frame['actor'] == snapshot['side']
                        assert first_contact == branch['target']
            assert first_contact == branch['contact_center']
            terminal(frame, branch)
            checked += 1
            if improvement:
                match = matches[snapshot['match_id']]
                prefix = match['actions'][:snapshot['tick']]
                check = observation_consistency(match, prefix, branch, snapshot['side'])
                good = check['action_conflict'] is None and check['first_standard_loop_tick'] is None
                consistent += good
                if good and snapshot['returns'] > 0 and representative is None:
                    representative = dict(phase=data['phase'], seed=snapshot['seed'],
                        rally=snapshot['rally'], snapshot_id=sid, target=branch['target'],
                        side=snapshot['side'], baseline_value=baseline['value'],
                        alternative_value=branch['value'], returns_at_start=snapshot['returns'],
                        baseline_contact_center=baseline['contact_center'],
                        baseline_terminal=baseline['terminal'], alternative_terminal=branch['terminal'],
                        selection='First retained strict loss-to-win branch after at least one return whose full trajectory is observation-consistent and never loops.',
                        **check)
                    print('Representative', json.dumps(representative), flush=True)
            if representative_only and representative is not None:
                break
        if representative_only and representative is not None:
            break
        if (sid+1)%100 == 0:
            print(path.name, 'checked snapshots', sid+1, '/', len(snapshots), flush=True)
    result = dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        sha256=digest(path), phase=data['phase'], seeds=data['seeds'],
        scope='representative search only' if representative_only else 'all retained branches',
        baseline_matches=len(matches), prefix_transitions=prefix_steps,
        snapshots=len(snapshots), branches_checked=checked, branch_transitions=steps,
        strict_loss_to_win_branches_checked=strict,
        observation_consistent_strict_improvements=consistent,
        selected_training_branches=selected, representative=representative,
        seconds=time.monotonic()-started)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--representative-only', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'training/results/classic-tactics/verification.json')
    args = parser.parse_args()
    records = []
    for path in args.paths:
        records.append(verify(path, args.representative_only))
    result = dict(rules='pong-v4', scalar_reference_sha256=digest(REFERENCE),
        verifier_sha256=digest(Path(__file__)), game_source_unchanged=
        (ROOT/'src/game.ts').read_bytes() == (ROOT/'training/v4/game.ts').read_bytes(),
        results=records,
        limitations=['A consistent finite trajectory establishes a possible stateless mapping on visited states, not a general winning policy.'])
    assert result['game_source_unchanged']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
