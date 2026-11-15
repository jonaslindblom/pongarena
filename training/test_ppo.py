"""Checks for reward credit, terminal draws, and a real PPO/export update."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import onnxruntime as ort
import torch
from models import network
from export import export_model
from ppo import advantages, finetune


class PPOChecks(unittest.TestCase):
    def test_invalid_settings_cannot_silently_remove_the_opponent_pool_limit(self):
        for config in ({'snapshot_limit': 0}, {'snapshot_every': 0}, {'temperature': 0},
                       {'script_fraction': 1.1}, {'snapshots_limit': 4}):
            with self.assertRaises(ValueError):
                finetune(None, config, seed=52)

    def test_rally_reset_does_not_leak_future_rewards_or_values(self):
        # Win, terminal draw, and a live rollout cut, all with a next-state
        # value of 100. Only the live rally is allowed to bootstrap that value.
        reward = torch.tensor([[1., 0., 0.]])
        value = torch.tensor([[.25, .25, .25]])
        ended = torch.tensor([[True, True, False]])
        advantage, returns = advantages(reward, value, ended, torch.full((3,), 100.), .9, .95)
        torch.testing.assert_close(returns, torch.tensor([[1., 0., 90.]]))
        torch.testing.assert_close(advantage, returns - value)

        reward = torch.tensor([[1.], [-1.]])
        advantage, returns = advantages(reward, torch.zeros_like(reward),
            torch.ones_like(reward, dtype=torch.bool), torch.tensor([99.]), 1., 1.)
        torch.testing.assert_close(returns, reward)

    def test_delayed_rally_result_reaches_earlier_decisions(self):
        reward = torch.tensor([[0.], [0.], [1.]])
        advantage, returns = advantages(reward, torch.zeros_like(reward),
            torch.tensor([[False], [False], [True]]), torch.tensor([99.]), .9, 1.)
        torch.testing.assert_close(returns, torch.tensor([[.81], [.9], [1.]]))

    def test_seeded_outcome_updates_preserve_the_exported_actor_contract(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)
        policy = network([6, 16, 6, 3], 'Relu')
        before = [value.detach().clone() for value in policy.parameters()]
        cfg = dict(rounds=2, envs=8, horizon=64, epochs=2, batch_size=128,
                   critic_warmup=0, learning_rate=.001, snapshot_every=1, snapshot_limit=2)
        result = finetune(policy, cfg, seed=52)
        self.assertEqual(result['environment_steps'], 1024)
        self.assertGreater(result['wins'] + result['losses'] + result['draws'], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())))
        torch.manual_seed(51)
        repeated = network([6, 16, 6, 3], 'Relu')
        self.assertEqual(finetune(repeated, cfg, seed=52), result)
        for a, b in zip(policy.parameters(), repeated.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.onnx'
            export_model(policy, path, 'Relu')
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            probes = np.random.default_rng(53).uniform(-1, 1, (32, 6)).astype(np.float32)
            for observation in probes:
                with torch.no_grad():
                    expected = policy(torch.from_numpy(observation[None])).numpy()
                actual = session.run(None, {'observation': observation[None]})[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)
                self.assertEqual(actual.argmax(), expected.argmax())


if __name__ == '__main__':
    unittest.main()
