"""Verify a Python port with the committed burst fixtures; standard library only."""
import json
from pathlib import Path
from reference import advance, serve, ended_match, ended_rally, LoopDetector


def scores(actions):
    return [[int(action == i) for i in range(5)] for action in actions]


def main():
    data = json.loads(Path(__file__).with_name('fixtures.json').read_text())
    for case in data['boundaries']:
        assert advance(case['before'], scores(case['actions'])) == case['after']
    count = 0
    for match in data['matches']:
        frame, loop = serve(match['seed']), LoopDetector()
        for step in match['steps']:
            assert ended_rally(frame) == (step['actions'] is None)
            frame = serve(match['seed'], frame) if step['actions'] is None else advance(frame, scores(step['actions']), loop)
            assert frame == step['state'], (frame, step['state'])
            count += 1
        assert ended_match(frame)
    print(f"PASS: {len(data['boundaries'])} boundaries and {count} full-match frames.")


if __name__ == '__main__':main()
