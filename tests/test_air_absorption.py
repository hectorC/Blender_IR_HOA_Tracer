"""ISO 9613-1 atmospheric absorption regression tests."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import (  # noqa: E402
    air_attenuation_bands,
    iso9613_alpha_dbpm,
)


class AtmosphericAbsorptionTests(unittest.TestCase):
    def test_matches_iso_table_at_minus_20_c_and_50_percent_humidity(self):
        # ISO 9613-1:1993 Table 1 values are stated in dB/km. These three
        # frequencies exercise the relaxation and classical-loss regions.
        reference_db_per_m = {
            1000.0: 9.14e-3,
            4000.0: 13.2e-3,
            8000.0: 20.3e-3,
        }
        for frequency_hz, expected in reference_db_per_m.items():
            with self.subTest(frequency_hz=frequency_hz):
                actual = iso9613_alpha_dbpm(
                    frequency_hz, -20.0, 50.0, 101.325
                )
                self.assertAlmostEqual(actual / expected, 1.0, delta=0.025)

    def test_zero_distance_has_unity_gain(self):
        np.testing.assert_array_equal(
            air_attenuation_bands(0.0),
            np.ones(7, dtype=np.float32),
        )

    def test_distance_attenuation_composes_multiplicatively(self):
        gain_10m = air_attenuation_bands(10.0, 20.0, 50.0, 101.325)
        gain_20m = air_attenuation_bands(20.0, 20.0, 50.0, 101.325)
        np.testing.assert_allclose(gain_20m, gain_10m ** 2, rtol=2e-6, atol=2e-6)

    def test_high_frequencies_lose_more_level_over_room_scale_distance(self):
        gains = air_attenuation_bands(30.0, 20.0, 50.0, 101.325)
        self.assertGreater(float(gains[0]), float(gains[-1]))
        self.assertTrue(np.all(gains > 0.0))
        self.assertTrue(np.all(gains <= 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
