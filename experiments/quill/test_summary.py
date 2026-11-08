"""Exercise paired league orientation, drawn rallies and common-field gains."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quill_league_summary', HERE / 'summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def synthetic_league():
    # Interleave candidate and opponent hashes to exercise both orientations.
    names = ['large', 'quill_speed', 'small', 'counter97', 'attack75',
             'quill_geometry', 'predict75', 'quill', 'follow']
    entrants = [{'id': name, 'name': name, 'sha256': f'{i:064x}', 'params': i, 'macs': i}
                for i, name in enumerate(names, 1)]
    seeds = [101, 202, 303]
    plan = {'protocol': 'paired-round-robin', 'seeds': seeds, 'entrants': entrants}
    run = {'status': 'complete', 'protocol': plan['protocol'], 'benchmark': 'synthetic',
           'spec': {'entrants': copy.deepcopy(entrants), 'seeds': list(reversed(seeds))},
           'contests': []}
    # Tuples are (own rally wins, opponent rally wins); the remainder draw.
    profiles = {
        'quill': [(6, 2), (2, 6), (3, 3)],
        'quill_speed': [(7, 1), (3, 5), (4, 2)],
        'quill_geometry': [(4, 4), (4, 4), (4, 4)],
    }
    for i, a in enumerate(entrants):
        for b in entrants[i + 1:]:
            for seed_index, seed in enumerate(seeds):
                candidate = (a if a['id'] in profiles else
                             b if b['id'] in profiles else None)
                other = b if candidate is a else a
                a_points = b_points = 0
                if candidate and other['id'] in summary.COMMON_FIELD:
                    wins, losses = profiles[candidate['id']][seed_index]
                    a_points, b_points = ((wins, losses) if candidate is a else (losses, wins))
                elif {a['id'], b['id']} == {'quill_speed', 'quill'}:
                    # A severe head-to-head loss must not leak into the common
                    # field, which excludes Quill and both proposed variants.
                    a_points, b_points = (0, 10) if a['id'] == 'quill_speed' else (10, 0)
                first_a = min(a_points, 5)
                first_b = min(b_points, 5 - first_a)
                second_a, second_b = a_points - first_a, b_points - first_b
                run['contests'].append({'a': a['sha256'], 'b': b['sha256'], 'seed': seed,
                    'matches': [{'points': [first_a, first_b], 'rallies': 5},
                                {'points': [second_b, second_a], 'rallies': 5}]})
    return run, plan


class LeagueSummaryTests(unittest.TestCase):
    def test_swapped_sides_and_hash_orientation_preserve_rally_and_contest_counts(self):
        run, plan = synthetic_league()
        result = summary.summarize(run, plan)
        self.assertEqual(result['games'], 216)
        self.assertEqual(result['rallies_per_opponent'], 30)
        for candidate, expected in [('quill', (11, 8, 11, 1, 1, 1)),
                                    ('quill_speed', (14, 8, 8, 2, 0, 1)),
                                    ('quill_geometry', (12, 6, 12, 0, 3, 0))]:
            for opponent in summary.COMMON_FIELD:
                pair = result['models'][candidate]['opponents'][opponent]
                self.assertEqual(tuple(pair[key] for key in [
                    'rally_wins', 'rally_draws', 'rally_losses',
                    'contest_wins', 'contest_draws', 'contest_losses']), expected)
        baseline = result['models']['quill']['opponents']['large']
        self.assertEqual(baseline['paired_rally_score'], [.7, .3, .5])
        self.assertEqual(baseline['paired_contest_score'], [1., 0., .5])
        self.assertEqual(baseline['rally_score'], .5)
        self.assertEqual(baseline['contest_score'], .5)

    def test_common_six_opponent_field_excludes_variant_head_to_heads(self):
        run, plan = synthetic_league()
        result = summary.summarize(run, plan)
        speed = result['models']['quill_speed']
        self.assertEqual(speed['opponents']['quill']['rally_losses'], 30)
        self.assertEqual(speed['opponents']['quill']['contest_losses'], 3)
        np.testing.assert_allclose(speed['paired_common_field_rally_score'], [.8, .4, .6])
        self.assertAlmostEqual(speed['common_field_rally_score'], .6)
        self.assertAlmostEqual(speed['common_field_contest_score'], 2 / 3)
        gains = result['paired_gains_over_quill']['quill_speed']
        self.assertAlmostEqual(gains['rally']['mean'], .1)
        self.assertAlmostEqual(gains['contest']['mean'], 1 / 6)

    def test_bootstrap_retains_all_common_opponents_within_whole_seed_samples(self):
        run, plan = synthetic_league()
        result = summary.summarize(run, plan)
        samples = np.random.default_rng(831701).integers(0, 3, (5000, 3))
        # All six opponent effects on a seed agree here. Resampling their 18
        # individual results would hide the seed sensitivity of these gains.
        rally_delta = np.array([-.2, .2, 0])
        contest_delta = np.array([-.5, .5, 0])
        for metric, delta in [('rally', rally_delta), ('contest', contest_delta)]:
            expected = np.quantile(delta[samples].mean(axis=1), [.025, .975])
            actual = result['paired_gains_over_quill']['quill_geometry'][metric]
            self.assertAlmostEqual(actual['mean'], 0)
            np.testing.assert_allclose(actual['ci95'], expected)
        geometry = result['models']['quill_geometry']
        self.assertIsNone(geometry['common_field_rally_score_ci95'])
        self.assertIsNone(geometry['common_field_contest_score_ci95'])

    def test_incomplete_and_changed_model_evaluations_are_rejected(self):
        for mutation, expected in [
            (lambda run: run.update(status='running'), 'complete league'),
            (lambda run: run['contests'].pop(), 'incomplete'),
            (lambda run: run['spec']['entrants'][0].update(sha256='changed'), 'model bytes'),
            (lambda run: run['spec'].update(seeds=[101]), 'seeds differ'),
        ]:
            run, plan = synthetic_league()
            mutation(run)
            with self.assertRaisesRegex(ValueError, expected):
                summary.summarize(run, plan)

    def test_impossible_individual_games_cannot_hide_inside_valid_aggregate_totals(self):
        for points in ([6, 0], [-1, 2], [1.5, 1]):
            run, plan = synthetic_league()
            target = next(contest for contest in run['contests']
                          if contest['a'] == run['spec']['entrants'][0]['sha256'] and
                          contest['b'] == run['spec']['entrants'][1]['sha256'])
            target['matches'][0]['points'] = points
            target['matches'][1]['points'] = [0, 0]
            with self.assertRaisesRegex(ValueError, 'five completed rallies and valid points'):
                summary.summarize(run, plan)


if __name__ == '__main__':
    unittest.main()
