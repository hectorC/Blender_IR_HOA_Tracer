"""Regression tests for octave-band impulse reconstruction."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import (  # noqa: E402
    BAND_CENTERS_HZ,
    NUM_BANDS,
    add_filtered_impulse,
    design_band_kernel,
)


class BandFilterTests(unittest.TestCase):
    profile = np.array(
        (1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05), dtype=np.float32
    )

    def test_default_filter_scales_with_sample_rate(self):
        self.assertEqual(len(design_band_kernel(self.profile, 48000)), 512)
        self.assertEqual(len(design_band_kernel(self.profile, 96000)), 1024)

    def test_response_matches_requested_values_at_band_centres(self):
        sample_rate = 48000
        kernel = design_band_kernel(self.profile, sample_rate)
        fft_size = 65536
        response = np.abs(np.fft.rfft(kernel, fft_size))
        frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        measured = np.array([
            response[np.argmin(np.abs(frequencies - centre))]
            for centre in BAND_CENTERS_HZ
        ])
        error_db = 20.0 * np.log10(measured / self.profile)
        self.assertLess(float(np.max(np.abs(error_db))), 0.25)

    def test_flat_profile_remains_a_time_aligned_impulse(self):
        kernel = design_band_kernel(np.ones(NUM_BANDS), 48000)
        self.assertEqual(int(np.argmax(np.abs(kernel))), 0)
        self.assertAlmostEqual(float(kernel[0]), 1.0, places=6)
        np.testing.assert_allclose(kernel[1:], 0.0, atol=1e-7)

    def test_filtered_arrival_is_causal_and_preserves_dc_level(self):
        ir = np.zeros((16, 1200), dtype=np.float32)
        ambisonic_vector = np.zeros(16, dtype=np.float32)
        ambisonic_vector[0] = 1.0

        wrote = add_filtered_impulse(
            ir,
            ambisonic_vector,
            delay_samples=100.25,
            amplitude=0.5,
            band_profile=self.profile,
            sr=48000,
        )

        self.assertTrue(wrote)
        np.testing.assert_array_equal(ir[:, :100], np.zeros_like(ir[:, :100]))
        self.assertAlmostEqual(float(np.sum(ir[0])), 0.5, places=5)
        np.testing.assert_array_equal(ir[1:], np.zeros_like(ir[1:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
