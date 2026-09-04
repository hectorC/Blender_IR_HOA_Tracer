"""Independent frequency, timing, and boundary checks for arrival placement."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import (  # noqa: E402
    FRACTIONAL_DELAY_HALF_WIDTH,
    add_fractional_impulse,
    fractional_delay_kernel,
)


class FractionalDelayTests(unittest.TestCase):
    def test_integers_remain_exact_impulses(self):
        for delay in (-5, 0, 100):
            start, weights = fractional_delay_kernel(delay)
            self.assertEqual(start, delay)
            np.testing.assert_array_equal(weights, np.ones(1))

    def test_fractional_delay_is_flat_and_time_aligned_in_the_passband(self):
        frequencies = np.linspace(0.0, 0.45, 2048)
        for fraction in (0.01, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99):
            with self.subTest(fraction=fraction):
                delay = 100 + fraction
                start, weights = fractional_delay_kernel(delay)
                samples = start + np.arange(weights.size)
                response = np.exp(-2j * np.pi * frequencies[:, None] * samples) @ weights
                expected = np.exp(-2j * np.pi * frequencies * delay)
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
                self.assertLess(float(np.max(abs(20 * np.log10(abs(response))))), 0.002)
                self.assertLess(float(np.max(abs(np.angle(response / expected)))), 0.0001)

    def test_half_sample_delay_does_not_lose_three_db_at_quarter_sample_rate(self):
        start, weights = fractional_delay_kernel(100.5)
        response = np.exp(-2j * np.pi * 0.25 * (start + np.arange(weights.size))) @ weights
        self.assertLess(abs(float(20 * np.log10(abs(response)))), 0.002)

    def test_output_clipping_matches_full_kernel_without_wrap_or_renormalization(self):
        values = np.array((1.0, -0.4), dtype=np.float32)
        for delay in (-100.25, 0.25, 63.75, 164.25):
            with self.subTest(delay=delay):
                ir = np.zeros((2, 64), dtype=np.float32)
                add_fractional_impulse(ir, values, delay)
                expected = np.zeros_like(ir)
                start, weights = fractional_delay_kernel(delay)
                for index, weight in enumerate(weights):
                    sample = start + index
                    if 0 <= sample < ir.shape[-1]:
                        expected[:, sample] = values * weight
                np.testing.assert_array_equal(ir, expected)

    def test_kernel_support_is_bounded(self):
        start, weights = fractional_delay_kernel(100.25)
        self.assertEqual(start, 100 - FRACTIONAL_DELAY_HALF_WIDTH)
        self.assertEqual(weights.size, 2 * FRACTIONAL_DELAY_HALF_WIDTH + 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
