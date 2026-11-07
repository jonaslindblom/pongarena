"""Centered openings remain human-catchable and scalar/vector physics agree."""
import math
import unittest
import numpy as np
from game import Arena, advance, serve


class ServeTests(unittest.TestCase):
    def test_openings_allow_reaction_time_without_a_wall_bounce(self):
        arena = Arena(1)
        for seed in [0, 1, 123456789, 4294967295]:
            prior = None
            for _ in range(5):
                start = serve(seed, prior)
                self.assertEqual((start['x'], start['y']), (180, 80))
                angle = math.degrees(math.atan2(abs(start['vy']), abs(start['vx'])))
                self.assertTrue(10 <= angle <= 15)
                receiver = 0 if start['vx'] < 0 else 1
                landing = start['y'] + 33 * start['vy']
                frame = start
                arena.load([start])
                for tick in range(1, 34):
                    actions = [1, 1]
                    if tick > 18:
                        error = landing - frame['paddles'][receiver]
                        actions[receiver] = 1 if abs(error) <= 1 else 0 if error < 0 else 2
                    frame = advance(frame, np.eye(3, dtype=int)[actions].tolist())
                    outcome, _ = arena.step(np.array([actions]))
                    np.testing.assert_array_equal(arena.ball[0], [frame[k] for k in ('x', 'y', 'vx', 'vy')])
                    np.testing.assert_array_equal(arena.paddles[0], frame['paddles'])
                    self.assertEqual(outcome[0], -1)
                    if tick < 33:
                        self.assertEqual(frame['vy'], start['vy'])
                self.assertEqual(frame['event'], 'return')
                self.assertEqual(frame['actor'], receiver)
                prior = frame


if __name__ == '__main__':
    unittest.main()
