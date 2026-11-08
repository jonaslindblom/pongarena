"""Check batched experiments against fixed-batch ONNX and scalar Pong physics.

Run with: .venv/bin/python -m unittest discover -s experiments/quill -p 'test_*.py'
Requires NumPy, ONNX and ONNX Runtime, without PyTorch.
"""
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quill_experiment_evaluator', HERE / 'evaluate.py')
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
from game import LoopDetector, advance, ended_rally, observation, serve  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_batched_outcomes_match_original_onnx_and_scalar_physics(self):
        paths = [evaluate.ROOT / 'public/models/quill.onnx',
                 evaluate.ROOT / 'public/models/small.onnx']
        loaded = [evaluate.load_policy(path) for path in paths]
        for path, (_, record) in zip(paths, loaded):
            self.assertEqual(record['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        # Published development serves are appropriate for a reproducible parity
        # test; these results are never used as independent strength evidence.
        seeds = [1153541941, 1229251971]
        vector = evaluate.paired_outcomes(loaded[0][0], loaded[1][0], seeds)
        self.assertEqual(vector.shape, (2, 2, 5))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        sessions = [ort.InferenceSession(path.read_bytes(), options,
                                        providers=['CPUExecutionProvider']) for path in paths]
        scalar = np.full_like(vector, -1)
        for side in (0, 1):
            for seed_index, seed in enumerate(seeds):
                frame = None
                for rally in range(5):
                    frame = serve(seed, frame)
                    loops = LoopDetector()
                    while not ended_rally(frame):
                        scores = []
                        for player in (0, 1):
                            controller = 0 if player == side else 1
                            obs = np.array([observation(frame, player)], np.float32)
                            scores.append(sessions[controller].run(
                                ['action_scores'], {'observation': obs})[0][0].tolist())
                        frame = advance(frame, scores, loops)
                    scalar[side, seed_index, rally] = (frame['actor']
                        if frame['event'] == 'point' else 2)
        np.testing.assert_array_equal(vector, scalar)

    def test_scores_keep_both_sides_and_all_five_rallies_together(self):
        # Seed zero has 4 wins, 2 losses, 4 draws. Seed one has 3 wins,
        # 3 losses, 4 draws. These are one won contest and one drawn contest.
        outcomes = np.array([
            [[0, 0, 1, 2, 2], [1, 1, 0, 2, 2]],
            [[1, 1, 0, 2, 2], [1, 1, 0, 2, 2]],
        ])
        resamples = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (1024, 1))
        result = evaluate.summarize(outcomes, resamples)
        self.assertEqual((result['rally_wins'], result['rally_draws'],
                          result['rally_losses']), (7, 8, 5))
        self.assertEqual((result['contest_wins'], result['contest_draws'],
                          result['contest_losses']), (1, 1, 0))
        self.assertEqual(result['paired_point_margin'], [2, 0])
        self.assertEqual(result['paired_rally_score'], [.6, .5])
        self.assertEqual(result['paired_contest_score'], [1., .5])
        self.assertAlmostEqual(result['rally_score'], .55)
        self.assertEqual(result['contest_score'], .75)
        self.assertEqual(result['rally_score_ci95'], [.5, .6])
        self.assertEqual(result['contest_score_ci95'], [.5, 1.])
        # Mirroring the court and swapping which player is evaluated preserves
        # scores, including first-index tie semantics inside each ONNX policy.
        mirrored = np.where(outcomes[::-1] < 2, 1 - outcomes[::-1], 2)
        self.assertEqual(result, evaluate.summarize(mirrored, resamples))

    def test_incomplete_rallies_and_invalid_shapes_are_rejected(self):
        resamples = np.zeros((4096, 1), dtype=int)
        for outcomes in (np.full((2, 1, 5), -1), np.full((2, 1, 5), 3),
                         np.zeros((1, 1, 5)), np.zeros((2, 5))):
            with self.assertRaisesRegex(ValueError, 'completed paired rallies'):
                evaluate.summarize(outcomes, resamples)

    def test_no_variation_does_not_create_a_false_precision_interval(self):
        resamples = np.zeros((4096, 1), dtype=int)
        result = evaluate.summarize(np.full((2, 1, 5), 2), resamples)
        self.assertEqual(result['rally_score'], .5)
        self.assertEqual(result['contest_score'], .5)
        self.assertIsNone(result['rally_score_ci95'])
        self.assertIsNone(result['contest_score_ci95'])

    def test_model_names_are_unambiguous(self):
        self.assertEqual(evaluate.named_paths(['Quill=public/models/quill.onnx']),
                         {'Quill': 'public/models/quill.onnx'})
        for entries in (['Quill'], ['=model.onnx'], ['Quill='], ['A=a.onnx', 'A=b.onnx']):
            with self.assertRaisesRegex(ValueError, 'NAME=PATH'):
                evaluate.named_paths(entries)


if __name__ == '__main__':
    unittest.main()
