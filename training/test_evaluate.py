"""Evaluation keeps paired serves together and holds test seeds out of selection."""
import unittest
from unittest.mock import patch
import numpy as np
from evaluate import DEVELOPMENT_SEEDS, FINAL_SEEDS, development_field, summarize


class EvaluationTests(unittest.TestCase):
    def test_seed_sets_are_distinct_from_training_and_each_other(self):
        self.assertEqual(len(set(DEVELOPMENT_SEEDS)), 32)
        self.assertEqual(len(set(FINAL_SEEDS)), 128)
        self.assertTrue(all(1_000_000_000 <= seed < 2_000_000_000 for seed in DEVELOPMENT_SEEDS))
        self.assertTrue(all(2_510_000_000 <= seed < 2_600_000_000 for seed in FINAL_SEEDS))

    def test_selection_rejects_test_seeds_before_playing(self):
        with patch('evaluate.field') as run:
            for seeds in (FINAL_SEEDS, [2_100_000_000], [1], []):
                with self.assertRaisesRegex(ValueError, 'development seeds'):
                    development_field(None, seeds)
            run.assert_not_called()
            development_field(None)
            run.assert_called_once_with(None, DEVELOPMENT_SEEDS)

    def test_confidence_interval_resamples_whole_paired_serve_sets(self):
        # One seed always wins, the other always loses, on both paddle sides.
        # Twenty independent-rally samples would hide this extreme seed sensitivity.
        sides = np.repeat([0, 1], 10)
        outcomes = [0] * 5 + [1] * 5 + [1] * 5 + [0] * 5
        result = summarize(outcomes, sides, 2)
        self.assertEqual(result['paired_rally_score'], [1., 0.])
        self.assertEqual(result['paired_point_margin'], [10, -10])
        self.assertEqual(result['rally_score'], .5)
        self.assertEqual(result['rally_score_ci95'], [0., 1.])
        self.assertEqual(result, summarize(outcomes, sides, 2))

    def test_draws_and_swapped_sides_have_the_same_scoring(self):
        sides = np.repeat([0, 1], 5)
        outcomes = [0, 0, 1, 2, 2, 1, 1, 0, 2, 2]
        result = summarize(outcomes, sides, 1)
        self.assertEqual((result['wins'], result['draws'], result['losses']), (4, 4, 2))
        self.assertAlmostEqual(result['rally_score'], .6)
        self.assertEqual(result['rally_score_ci95'], [.6, .6])
        self.assertEqual(result, summarize([1 - x if x < 2 else x for x in outcomes], 1 - sides, 1))

    def test_incomplete_rallies_cannot_be_reported_as_losses(self):
        with self.assertRaisesRegex(ValueError, 'completed rallies'):
            summarize([-1] * 10, np.repeat([0, 1], 5), 1)


if __name__ == '__main__':
    unittest.main()
