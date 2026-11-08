"""Independent physics checks for analytical Quill flight/recovery networks."""
import ast
from pathlib import Path
import unittest

import numpy as np

import precision_variants
import recovery_variants


def activations(observations, layers):
    """Evaluate ordinary float32 dense layers and retain hidden activations."""
    current = np.asarray(observations, dtype=np.float32)
    result = []
    for i, (weight, bias) in enumerate(layers):
        current = current @ weight.T + bias
        if i < len(layers) - 1:
            current = np.maximum(current, 0)
        result.append(current)
    return result


def observations_for_flight(direction):
    """Legal horizontal lattices, including every pre-contact decision.

    Speed 5 is a center serve; every later speed starts at the other contact
    plane. Vertical states cover both walls, flat flights and steep returns.
    Expected forecasts use discrete ticks, before applying wall reflection.
    """
    inputs, expected, descriptions = [], [], []
    for speed in range(5, 14):
        origin = 180 if speed == 5 else (345 if direction < 0 else 15)
        distance = origin - 15 if direction < 0 else 345 - origin
        for elapsed in range(int(np.ceil(distance / speed))):
            x = origin + direction * speed * elapsed
            remaining = (x - 15) if direction < 0 else (345 - x)
            ticks = int(np.ceil(remaining / speed))
            for y in (4, 9, 40, 79, 80, 81, 120, 151, 156):
                for vy in (-9, -6, -3, -1, 0, 1, 3, 6, 9):
                    inputs.append([(x - 180) / 180, (y - 80) / 80,
                                   direction * speed / 6, vy / 6, 0, 0])
                    expected.append(y + ticks * vy)
                    descriptions.append((speed, x, y, vy, ticks))
    return np.asarray(inputs, np.float32), np.asarray(expected), descriptions


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.precision = precision_variants.weights(step=.25, flight_bias=0, exact_ticks=True)
        cls.recovery = recovery_variants.weights(step=.25, exact_ticks=True)

    def assert_flight_forecast(self, layers, direction, index):
        observations, expected, descriptions = observations_for_flight(direction)
        # Keep the largest matrices bounded when running on a small CPU.
        largest_error, largest_case = 0, None
        for start in range(0, len(observations), 512):
            hidden = activations(observations[start:start + 512], layers)[2]
            actual = (hidden[:, index] - 16) * 80 + 80
            errors = np.abs(actual - expected[start:start + 512])
            worst = int(np.argmax(errors))
            if errors[worst] > largest_error:
                largest_error = float(errors[worst])
                largest_case = descriptions[start + worst]
        # Linear square interpolation at step .25 has <=step²/4 error.
        # The difference of two interpolation errors is bounded by the same
        # value: each error lies in [0,step²/4]. At speed 5 that contributes
        # at most (270/5)*(.25²/4)=.84375 court units, plus float32 roundoff.
        self.assertLessEqual(largest_error, 1.0,
                             f'worst speed/x/y/vy/ticks={largest_case}')

    def test_exact_tick_incoming_forecast_on_every_legal_horizontal_lattice(self):
        self.assert_flight_forecast(self.precision, -1, 0)

    def test_bidirectional_incoming_forecast_keeps_exact_tick_geometry(self):
        self.assert_flight_forecast(self.recovery, -1, 0)

    def test_exact_tick_outgoing_forecast_on_every_legal_horizontal_lattice(self):
        self.assert_flight_forecast(self.recovery, 1, 5)

    def test_direction_gates_zero_the_inactive_forecast(self):
        for direction, inactive in ((-1, 5), (1, 0)):
            observations, _, _ = observations_for_flight(direction)
            for start in range(0, len(observations), 512):
                hidden = activations(observations[start:start + 512], self.recovery)[2]
                np.testing.assert_allclose(hidden[:, inactive] - 16, 0, atol=2e-5, rtol=0)

    def test_compiled_recovery_targets_match_independent_discrete_return_range(self):
        heights = (4, 5, 9, 10, 14, 15, 40, 80, 120, 145, 146, 150, 155, 156)
        inputs = np.array([[0, (height - 80) / 80, 13 / 6, 0, 0, 0]
                           for height in heights], np.float32)
        target = 80 + 80 * activations(inputs, self.recovery)[-1][:, 2]
        expected = []
        for height in heights:
            low, high = max(-15, height - 145), min(15, height - 15)
            speeds = range(int(np.floor(.6 * low + .5)), int(np.floor(.6 * high + .5)) + 1)
            # Repeated geometric reflection, independently of the compiled
            # triangle wave and the recovery() helper under test.
            destinations = []
            for vy in speeds:
                y = height + 26 * vy
                while y < 4 or y > 156:
                    y = 8 - y if y < 4 else 312 - y
                destinations.append(y)
            expected.append((min(destinations) + max(destinations)) / 2)
        np.testing.assert_allclose(target, expected, atol=.03, rtol=0)
        self.assertGreater(target[0], 80, 'Near the upper wall, the legal return range shifts down')
        self.assertLess(target[-1], 80, 'Near the lower wall, the legal return range shifts up')
        self.assertAlmostEqual(float(target[7]), 80, delta=.03)

    def test_recovery_branch_adds_no_offset_during_incoming_flights(self):
        inputs, _, _ = observations_for_flight(-1)
        # Representative samples span all speeds, wall extremes and directions.
        inputs = inputs[::17]
        original = activations(inputs, self.precision)[-1]
        recovery = activations(inputs, self.recovery)[-1]
        np.testing.assert_allclose(recovery, original, atol=4e-4, rtol=0)

    def test_constructors_remain_independent_of_training_modules(self):
        for module in (precision_variants, recovery_variants):
            source = Path(module.__file__).read_text()
            parsed = ast.parse(source)
            imports = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(any(name.split('.')[0] in {'torch', 'training', 'teachers', 'ppo'}
                                 for name in imports), imports)


if __name__ == '__main__':
    unittest.main()
