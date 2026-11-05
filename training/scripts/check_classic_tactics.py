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


def verify_labels(path, data, snapshots, groups, snapshot_frames):
    import numpy as np
    from pace_study import CONFIGS, make_policy

    dataset = path.with_suffix('.npz')
    with np.load(dataset, allow_pickle=False) as stored:
        arrays = {key:stored[key].copy() for key in stored.files}
    observations, labels, seeds, ids, kinds = [], [], [], [], []
    for sid, snapshot in snapshots.items():
        rows = groups[sid]
        baseline = next(row for row in rows if row['target'] is None)
        winners = [row for row in rows if row['target'] is not None and row['value'] == 1]
        if baseline['value'] != -1 or not winners:
            assert 'selected_target' not in snapshot
            continue
        side = snapshot['side']
        def tie(branch):
            prefix = zip(branch['actions'][:branch['flight_ticks']], baseline['actions'][:branch['flight_ticks']])
            return sum(actions(a)[side] != actions(b)[side] for a, b in prefix), branch['target']
        selected = min(winners, key=tie)
        assert selected['target'] == snapshot['selected_target']
        frame = snapshot_frames[sid]
        for code in selected['actions'][:selected['flight_ticks']]:
            observations.append(ref.observation(frame, side))
            labels.append(actions(code)[side])
            seeds.append(snapshot['seed'])
            ids.append(sid)
            kinds.append(snapshot['kinds'][side])
            frame = advance(frame, code)
    expected = dict(obs=np.asarray(observations, dtype=np.float32), labels=np.asarray(labels),
        source_seeds=np.asarray(seeds), branch_ids=np.asarray(ids))
    for key, values in expected.items():
        assert np.array_equal(arrays[key], values), f'{dataset}: incorrect {key}'
    original = np.empty(len(labels), dtype=int)
    kinds = np.asarray(kinds)
    for kind in ('attack', 'defend'):
        mask = kinds == kind
        if mask.any():
            original[mask] = make_policy(CONFIGS['pace13-v9-scaled'], kind)(expected['obs'][mask])
    assert np.array_equal(arrays['original_labels'], original)
    assert len(labels) == data['summary']['training_path_observations']
    return dict(path=str(dataset.relative_to(ROOT)) if dataset.is_relative_to(ROOT) else str(dataset),
        sha256=digest(dataset), examples=len(labels), selected_branches=len(set(ids)),
        observations_and_actions_match_scalar=True, selection_tie_rule_verified=True,
        original_script_labels_verified=True)


def verify_seeds():
    base = ROOT/'training/results/classic-tactics'
    protocol = json.loads((base/'probes/protocol.json').read_text())
    seed_sets = {'probe_'+key:values for key, values in protocol['seed_sets'].items()}
    for path in sorted((base/'learning').glob('*/config.json')):
        value = json.loads(path.read_text())
        if 'development_seeds' in value:
            seed_sets[path.parent.name+'_development'] = value['development_seeds']
    for path in sorted((base/'learning').glob('*/manifest.json')):
        value = json.loads(path.read_text())
        if 'final_seeds' in value:
            seed_sets[path.parent.name+'_final'] = value['final_seeds']
    proposed = set()
    for values in seed_sets.values():
        assert len(values) == len(set(values))
        proposed.update(values)
    historical_files, hits = 0, []
    for path in (ROOT/'training/results').rglob('*.json'):
        if 'classic-tactics' in path.parts:
            continue
        try:
            value = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        historical_files += 1
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, int) and not isinstance(item, bool) and item in proposed:
                hits.append(dict(path=str(path.relative_to(ROOT)), seed=item))
    assert not hits, hits
    train, dev, final = (set(protocol['seed_sets'][key]) for key in ('train', 'dev', 'final'))
    assert not train & dev and not train & final and not dev & final
    for key, values in seed_sets.items():
        if key.endswith('_development'):
            assert not set(values) & (train | final)
        if key.endswith('_final'):
            assert not set(values) & (train | dev)
            for name, development in seed_sets.items():
                if name.endswith('_development'):
                    assert not set(values) & set(development)
    return dict(seed_sets=seed_sets, distinct_seed_count=len(proposed),
        historical_json_files_checked=historical_files, historical_overlap=hits,
        train_development_final_separation=True,
        note='Repeated development lists across bounded follow-ups are intentional; fresh seeds do not guarantee unseen physical states.')


def verify_displayed_examples():
    base = ROOT/'training/results/classic-tactics/probes'
    data = json.loads((base/'train.json').read_text())
    output = []
    for path in sorted(base.glob('representative*.json')):
        record = json.loads(path.read_text())
        assert record['phase'] == 'train'
        snapshot = data['snapshots'][record['snapshot_id']]
        match = data['matches'][snapshot['match_id']]
        rows = [b for b in data['candidates'] if b['snapshot_id'] == snapshot['id']]
        baseline = next(b for b in rows if b['target'] is None)
        branch = next(b for b in rows if b['target'] == record['alternative_contact'])
        assert baseline['contact_center'] == record['baseline_contact']
        assert snapshot['tick'] == record['start_tick']
        assert record['contact_tick'] == snapshot['tick']+branch['flight_ticks']
        prefix = match['actions'][:snapshot['tick']]
        for key, row in [('baseline', baseline), ('alternative', branch)]:
            frame = opening(match)
            frames = [state(frame)]
            for code in prefix+row['actions']:
                frame = advance(frame, code)
                frames.append(state(frame))
            assert frames == record[key]['frames']
            assert row['outcome'] == record[key]['outcome']
            assert frame['rallyTick'] == record[key]['end_tick']
        consistency = observation_consistency(match, prefix, branch, snapshot['side'])
        assert consistency['action_conflict'] is None
        assert consistency['first_standard_loop_tick'] is None
        if 'legal_centers' in record:
            assert record['legal_centers'] == sorted(b['target'] for b in rows if b['target'] is not None)
            assert record['winning_centers'] == sorted(b['target'] for b in rows if b['target'] is not None and b['value'] == 1)
        output.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
            snapshot_id=snapshot['id'], seed=snapshot['seed'], rally=snapshot['rally'],
            baseline_contact=baseline['contact_center'], alternative_contact=branch['target'],
            baseline_value=baseline['value'], alternative_value=branch['value'], **consistency))
    return output


def canonical_observation_audit():
    """Numeric float32 equality intentionally treats +0 and -0 as the same."""
    import numpy as np

    base = ROOT/'training/results/classic-tactics'
    with np.load(base/'probes/train.npz', allow_pickle=False) as stored:
        training = stored['obs'].copy()
    keys = {tuple(row) for row in training}
    result = dict(equality='Numeric float32 values; signed zeros are equal',
        training_path_examples=len(training), training_unique_observations=len(keys),
        scope='Overlap with tactical-path supervision only; pooled game and synthetic training observations are not included.',
        phases={})
    for phase in ('dev', 'final'):
        data_path = base/'probes'/f'{phase}.json'
        if not data_path.exists():
            continue
        data = json.loads(data_path.read_text())
        with np.load(data_path.with_suffix('.npz'), allow_pickle=False) as stored:
            values = stored['obs'].copy()
        seen = np.array([tuple(row) in keys for row in values])
        unique = {tuple(row) for row in values}
        start_seen = {}
        for snapshot in data['snapshots']:
            frame = dict(zip(('x', 'y', 'vx', 'vy'), snapshot['ball']))
            frame['paddles'] = snapshot['paddles']
            start_seen[snapshot['id']] = tuple(ref.observation(frame, snapshot['side'])) in keys
        row = dict(path_examples=len(values), path_examples_seen=int(seen.sum()),
            unique_path_observations=len(unique), unique_path_observations_seen=len(unique & keys),
            flight_starts=len(start_seen), flight_starts_seen=sum(start_seen.values()))
        evaluation = base/'learning/evaluation/tactic-probes.json'
        if phase == 'final' and evaluation.exists():
            probes = json.loads(evaluation.read_text())
            assert probes['data_sha256'] == digest(data_path)
            row['model_partitions'] = {}
            for name, model in probes['models'].items():
                partitions = {}
                for value, label in [(False, 'unseen_in_training_paths'), (True, 'seen_in_training_paths')]:
                    subset = [r for r in model['rows'] if start_seen[r['snapshot_id']] == value]
                    opportunities = [r for r in subset if r['strict_opportunity']]
                    partitions[label] = dict(states=len(subset), strict_opportunities=len(opportunities),
                        strict_opportunities_won=sum(r['value'] == 1 for r in opportunities),
                        model_wins=sum(r['value'] == 1 for r in subset),
                        better=sum(r['value'] > r['baseline_value'] for r in subset),
                        worse=sum(r['value'] < r['baseline_value'] for r in subset))
                row['model_partitions'][name] = partitions
            row['frozen_probe_results_sha256'] = digest(evaluation)
        result['phases'][phase] = row
    return result


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
    if not representative_only:
        result['label_verification'] = verify_labels(path, data, snapshots, groups, snapshot_frames)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', type=Path)
    parser.add_argument('--representative-only', action='store_true')
    parser.add_argument('--supplement-only', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'training/results/classic-tactics/verification.json')
    args = parser.parse_args()
    if args.supplement_only:
        prior = json.loads(args.out.read_text())
        assert prior['verifier_sha256'] == digest(Path(__file__)), 'Replay verification belongs to a different checker revision'
        records = prior['results']
        for record in records:
            assert digest(ROOT/record['path']) == record['sha256']
    else:
        if not args.paths:
            parser.error('Supply probe JSON paths, or --supplement-only after replay verification.')
        records = [verify(path, args.representative_only) for path in args.paths]
    result = dict(rules='pong-v4', scalar_reference_sha256=digest(REFERENCE),
        verifier_sha256=digest(Path(__file__)), game_source_unchanged=
        (ROOT/'src/game.ts').read_bytes() == (ROOT/'training/v4/game.ts').read_bytes(),
        results=records, seed_audit=verify_seeds(), displayed_examples=verify_displayed_examples(),
        canonical_observation_overlap=canonical_observation_audit(),
        limitations=['A consistent finite trajectory establishes a possible stateless mapping on visited states, not a general winning policy.'])
    assert result['game_source_unchanged']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
