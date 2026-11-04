"""Run with Python 3.10+ from any directory. Replace reference with your port."""
import json
from pathlib import Path
from reference import advance, serve, observation, ended_match


def check(actual, expected):
    assert actual == expected['state'], (actual, expected['state'])
    for side in (0, 1):
        assert observation(actual, side) == expected['observations'][side]


def main():
    data = json.loads(Path(__file__).with_name('fixtures.json').read_text())
    steps = 0
    for match in data['matches']:
        f = serve(match['seed'])
        check(f, match['initial'])
        for row in match['steps']:
            actions = row['actions']
            f = serve(match['seed'], f) if actions is None else advance(
                f, [[int(a == i) for i in range(3)] for a in actions])
            check(f, row)
            steps += 1
        assert ended_match(f)
    for case in data['cases']:
        check(advance(case['before'], case['scores']), case['expected'])
    print(f'PASS: {steps} exact match steps, {len(data["cases"])} boundary cases, both float32 observations.')


if __name__ == '__main__':
    main()
