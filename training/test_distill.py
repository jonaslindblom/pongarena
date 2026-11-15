"""The student stays tiny and independent; labels and selection retain their meaning."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from check import contract
from distill import WIDTHS, development, distillation_loss, learned_teacher
from export import export_model
from models import network
from evaluate import pair
from search import population_scores
from teachers import teacher

ROOT = Path(__file__).resolve().parent.parent


class DistillationTests(unittest.TestCase):
    def test_teacher_is_only_the_recorded_independent_learned_model(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / 'teacher.onnx'
            fake.write_bytes(b'unrecorded model')
            with self.assertRaisesRegex(ValueError, 'independently trained'):
                learned_teacher(fake)

    def test_teacher_conversion_preserves_soft_labels(self):
        torch.set_num_threads(1)
        path = ROOT / 'public/models/large.onnx'
        teacher = learned_teacher(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        probes = np.random.default_rng(17).uniform(
            [-1, -.95, -13/6, -1.5, -.8125, -.8125],
            [1, .95, 13/6, 1.5, .8125, .8125], (32, 6)).astype(np.float32)
        with torch.no_grad():
            expected = teacher(torch.from_numpy(probes)).numpy()
        actual = np.concatenate([session.run(None, {'observation': row[None]})[0] for row in probes])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))

    def test_export_has_exactly_quills_architecture_and_budget(self):
        policy = network(WIDTHS, 'Relu')
        self.assertEqual(sum(p.numel() for p in policy.parameters()), 163)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'student.onnx'
            export_model(policy, path, 'Relu')
            self.assertEqual(contract(path, 256), (163, 141))

    def test_matching_soft_preferences_is_optimal_without_inventing_certainty(self):
        logits = torch.tensor([[.3, -.2, .1]], requires_grad=True)
        targets = (logits.detach() / 2).softmax(-1)
        loss = distillation_loss(logits, targets, torch.ones(1), 2.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)
        confident = torch.tensor([[10., -10., -10.]])
        self.assertGreater(float(distillation_loss(confident, targets, torch.ones(1), 2.)), 1.)

    def test_selection_uses_paired_contests_and_rejects_final_seeds(self):
        with patch('distill.pair') as run:
            with self.assertRaisesRegex(ValueError, 'development seeds'):
                development(None, {'opponent': None}, [2_100_000_000])
            run.assert_not_called()
            run.return_value = dict(paired_point_margin=[2, -1, 0], rally_score=.52)
            score, rows = development(None, {'opponent': None}, [1_100_000_001])
            self.assertEqual(score, (.5, .52))
            self.assertEqual(rows['opponent']['league_score'], .5)

    def test_search_candidates_replay_the_same_paired_games_as_individual_policies(self):
        torch.set_num_threads(1)
        torch.manual_seed(48)
        policies = [network(WIDTHS, 'Relu') for _ in range(2)]
        # A simple follower fixture ensures parity covers successful returns too.
        with torch.no_grad():
            for p in policies[0].parameters():
                p.zero_()
            layers = [l for l in policies[0] if isinstance(l, torch.nn.Linear)]
            layers[0].weight[0, 1], layers[0].weight[0, 4] = 1., -1.
            layers[0].weight[1, 1], layers[0].weight[1, 4] = -1., 1.
            for layer in layers[1:-1]:
                layer.weight[0, 0] = layer.weight[1, 1] = 1.
            layers[-1].weight[0, 1] = layers[-1].weight[2, 0] = 1.
            layers[-1].bias[1] = .01
        vectors = [torch.nn.utils.parameters_to_vector(p.parameters()).detach().numpy() for p in policies]
        opponents = [teacher('interceptor'), teacher('attack')]
        seeds = [71, 183]
        actual = population_scores(vectors, seeds, opponents)
        for i, policy in enumerate(policies):
            expected = []
            for opponent in opponents:
                row = pair(policy, opponent, seeds)
                margins = np.asarray(row['paired_point_margin'])
                expected.append([np.mean((margins > 0) + .5*(margins == 0)), row['rally_score']])
            np.testing.assert_array_equal(actual[i], np.mean(expected, axis=0))


if __name__ == '__main__':
    unittest.main()
