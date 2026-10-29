"""Regression checks for the standalone local-estimate prototype."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "pyton_code/sobolev_task/main.py"
SPEC = importlib.util.spec_from_file_location("sobolev_demo", SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class IlluminanceTests(unittest.TestCase):
    def test_constant_weight_preserves_original_formula(self):
        result = demo.calculate_illuminance(1, 20, 5, 0.5, 2)
        self.assertAlmostEqual(result, 6 * 0.5 / (np.pi**2 * 2))

    def test_collision_count_does_not_depend_on_ray_count(self):
        result = demo.calculate_illuminance(1, 1, 5, 0.5, 2)
        self.assertAlmostEqual(result, 6 * 0.5 / (np.pi**2 * 2))

    def test_per_collision_weights(self):
        result = demo.calculate_illuminance(2, 3, 2, [1, 0.5, 0.25], 4)
        self.assertAlmostEqual(result, 2 * 1.75 / (np.pi**2 * 4))

    def test_original_weight_array_remains_supported(self):
        result = demo.calculate_illuminance(1, 20, 5, np.full(21, 0.5), 2)
        self.assertAlmostEqual(result, 6 * 0.5 / (np.pi**2 * 2))

    def test_rejects_singular_and_non_finite_distances(self):
        for distance in (0, -1, np.nan, np.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                demo.calculate_illuminance(1, 20, 5, 0.5, distance)

    def test_rejects_invalid_counts_and_weights(self):
        cases = [
            (1, 0, 5, 0.5, 1),
            (1, 2.5, 5, 0.5, 1),
            (1, 20, -1, 0.5, 1),
            (1, 20, 2, [0.5], 1),
            (1, 20, 2, np.nan, 1),
            (1, 20, 2, 1.1, 1),
            (-1, 20, 2, 0.5, 1),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                demo.calculate_illuminance(*arguments)


if __name__ == "__main__":
    unittest.main()
