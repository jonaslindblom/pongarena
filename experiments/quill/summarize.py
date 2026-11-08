"""Summarize a frozen WASM league, with paired whole-seed comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

COMMON_FIELD = ['large', 'small', 'counter97', 'attack75', 'predict75', 'follow']
CANDIDATES = ['quill', 'quill_speed', 'quill_geometry']


def summarize(run, plan):
    if run['status'] != 'complete' or run['protocol'] != plan['protocol']:
        raise ValueError('Require a complete league from the saved plan')
    seeds = sorted(plan['seeds'])
    if sorted(run['spec']['seeds']) != seeds:
        raise ValueError('League seeds differ from the frozen plan')
    identity = lambda rows: sorted((r['sha256'], r['params'], r['macs']) for r in rows)
    if identity(run['spec']['entrants']) != identity(plan['entrants']):
        raise ValueError('League model bytes differ from the frozen plan')
    entrants = {r['id']: r for r in run['spec']['entrants']}
    expected = len(entrants) * (len(entrants) - 1) * len(seeds) // 2
    if len(run['contests']) != expected:
        raise ValueError('The round robin is incomplete')
    contests = {(c['a'], c['b'], c['seed']): c for c in run['contests']}
    if len(contests) != expected:
        raise ValueError('Repeated contests')
    samples = np.random.default_rng(831701).integers(0, len(seeds), (5000, len(seeds)))

    def interval(values):
        values = np.asarray(values)
        return (np.quantile(values[samples].mean(axis=1), [.025, .975]).tolist()
                if len(values) > 1 and np.ptp(values) > 0 else None)

    report = {
        'plan': plan, 'benchmark': run['benchmark'],
        'runtime': 'ONNX Runtime Web CPU/WASM, one thread per independent worker',
        'games': expected * 2, 'rallies_per_opponent': len(seeds) * 10,
        'comparison_field': COMMON_FIELD,
        'interval_method': '95% percentile bootstrap, 5000 resamples of whole shared serve sets, '
                           'retaining both sides and all opponents; NumPy default_rng(831701). '
                           'No interval when no variation was observed.',
        'models': {}, 'paired_gains_over_quill': {},
    }
    for name in CANDIDATES:
        own_hash = entrants[name]['sha256']
        record = {'sha256': own_hash, 'parameters': entrants[name]['params'],
                  'macs': entrants[name]['macs'], 'opponents': {}}
        for opponent, entry in entrants.items():
            if opponent == name:
                continue
            other_hash = entry['sha256']
            a, b = sorted((own_hash, other_hash))
            rows = []
            for seed in seeds:
                contest = contests[a, b, seed]
                first, second = contest['matches']
                for match in (first, second):
                    points = match.get('points')
                    if (match.get('rallies') != 5 or not isinstance(points, list) or len(points) != 2 or
                            any(type(value) is not int or value < 0 for value in points) or
                            sum(points) > 5):
                        raise ValueError('Each paired game requires five completed rallies and valid points')
                a_points = first['points'][0] + second['points'][1]
                b_points = first['points'][1] + second['points'][0]
                if not 0 <= a_points + b_points <= 10:
                    raise ValueError('Invalid rally totals')
                wins, losses = (a_points, b_points) if own_hash == a else (b_points, a_points)
                rows.append((wins, 10 - wins - losses, losses))
            rows = np.asarray(rows)
            margins = rows[:, 0] - rows[:, 2]
            rally = .5 + margins / 20
            contest = .5 + np.sign(margins) / 2
            record['opponents'][opponent] = {
                'rally_wins': int(rows[:, 0].sum()), 'rally_draws': int(rows[:, 1].sum()),
                'rally_losses': int(rows[:, 2].sum()),
                'contest_wins': int((margins > 0).sum()),
                'contest_draws': int((margins == 0).sum()),
                'contest_losses': int((margins < 0).sum()),
                'rally_score': float(rally.mean()), 'rally_score_ci95': interval(rally),
                'contest_score': float(contest.mean()), 'contest_score_ci95': interval(contest),
                'paired_rally_score': rally.tolist(), 'paired_contest_score': contest.tolist(),
            }
        for metric in ('rally', 'contest'):
            values = np.mean([record['opponents'][opponent][f'paired_{metric}_score']
                              for opponent in COMMON_FIELD], axis=0)
            record[f'common_field_{metric}_score'] = float(values.mean())
            record[f'common_field_{metric}_score_ci95'] = interval(values)
            record[f'paired_common_field_{metric}_score'] = values.tolist()
        report['models'][name] = record
    for name in CANDIDATES[1:]:
        report['paired_gains_over_quill'][name] = {}
        for metric in ('rally', 'contest'):
            key = f'paired_common_field_{metric}_score'
            delta = np.asarray(report['models'][name][key]) - report['models']['quill'][key]
            report['paired_gains_over_quill'][name][metric] = {
                'mean': float(delta.mean()), 'ci95': interval(delta)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.league.read_text()), json.loads(args.plan.read_text()))
    with args.out.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
