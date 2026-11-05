"""Run with Python 3.10+ from any directory. Replace reference with your port."""
import json
from pathlib import Path
from reference import advance, serve, observation, ended_match, ended_rally, LoopDetector


def check(actual, expected):
    assert actual == expected['state'], (actual, expected['state'])
    for side in (0, 1):
        assert observation(actual, side) == expected['observations'][side]


def main():
    data = json.loads(Path(__file__).with_name('fixtures.json').read_text())
    steps = 0
    for match in data['matches']:
        loops = LoopDetector() if match['loopDetection'] else None
        f = serve(match['seed'])
        check(f, match['initial'])
        for row in match['steps']:
            actions = row['actions']
            f = serve(match['seed'], f) if actions is None else advance(
                f, [[int(a == i) for i in range(3)] for a in actions], loops)
            check(f, row)
            steps += 1
        assert ended_match(f)
    trace_steps = 0
    for trace in data['traces']:
        loops = LoopDetector() if trace['loopDetection'] else None
        f = trace['initial']['state']
        check(f, trace['initial'])
        for row in trace['steps']:
            f = advance(f, [[int(a == i) for i in range(3)] for a in row['actions']], loops)
            check(f, row)
            trace_steps += 1
        assert ended_rally(f)
    for case in data['cases']:
        loops = LoopDetector() if case.get('loopDetection') else None
        for frame in case.get('seenContacts', []):
            if loops is not None:
                loops.observe(frame)
        check(advance(case['before'], case['scores'], loops), case['expected'])
    for case in data['detectorCases']:
        loops = LoopDetector()
        assert [loops.observe(f) for f in case['frames']] == case['expected'], case['name']
    # An explicit restart or controller change must drop old contact history.
    loops = LoopDetector()
    contact = data['detectorCases'][0]['frames'][0]
    assert not loops.observe(contact) and loops.observe(contact)
    loops.reset()
    assert not loops.observe(contact)
    print(f'PASS: {steps} exact match steps, {trace_steps} trace steps, {len(data["cases"])} boundary cases, '
          f'{len(data["detectorCases"])} detector histories, reset, and both float32 observations.')


if __name__ == '__main__':
    main()
