"""Tests for energy-to-pressure ambisonic synthesis."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import (  # noqa: E402
    NUM_BANDS,
    design_complementary_filter_bank,
)
from ir_raytracer.core.ambisonic import AmbisonicEncoder  # noqa: E402
from ir_raytracer.core.synthesis import (  # noqa: E402
    AcousticEvent,
    synthesize_ambisonic_ir,
)


class SynthesisTests(unittest.TestCase):
    def test_filter_bank_recombines_to_a_delayed_impulse(self):
        kernels, delay = design_complementary_filter_bank(48000)
        combined = np.sum(kernels, axis=0)
        expected = np.zeros_like(combined)
        expected[delay] = 1.0
        np.testing.assert_allclose(combined, expected, atol=2e-5)

    def test_flat_deterministic_event_keeps_delay_and_pressure(self):
        event = AcousticEvent(
            delay_seconds=0.01,
            arrival_direction=Vector((1.0, 0.0, 0.0)),
            energy_bands=np.full(NUM_BANDS, 0.25, dtype=np.float32),
            kind='DIRECT',
        )
        ir, stats = synthesize_ambisonic_ir(
            [event], 48000, 0.05, AmbisonicEncoder()
        )
        self.assertEqual(stats.direct_events, 1)
        self.assertEqual(int(np.argmax(np.abs(ir[0]))), 480)
        self.assertAlmostEqual(float(ir[0, 480]), 0.5, places=5)

    def test_diffuse_phase_is_repeatable_for_a_seed(self):
        events = [
            AcousticEvent(
                delay_seconds=0.01 + index * 0.0007,
                arrival_direction=Vector((1.0, index * 0.1, 0.2)).normalized(),
                energy_bands=np.full(NUM_BANDS, 0.01, dtype=np.float32),
                kind='DIFFUSE',
                order=2,
            )
            for index in range(12)
        ]
        first, _ = synthesize_ambisonic_ir(
            events, 48000, 0.1, AmbisonicEncoder(), seed=4
        )
        second, _ = synthesize_ambisonic_ir(
            events, 48000, 0.1, AmbisonicEncoder(), seed=4
        )
        different, _ = synthesize_ambisonic_ir(
            events, 48000, 0.1, AmbisonicEncoder(), seed=5
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertGreater(float(np.max(np.abs(first))), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
