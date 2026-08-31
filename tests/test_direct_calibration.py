"""Regression tests for direct-path level calibration."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.ui.operators import calibrate_direct_1_over_r  # noqa: E402
from ir_raytracer.utils.scene_utils import speed_of_sound_bu  # noqa: E402


def _context(unit_scale: float = 1.0):
    scene = SimpleNamespace(
        airt_sr=48000,
        airt_air_temp_c=20.0,
        airt_air_humidity=50.0,
        unit_settings=SimpleNamespace(scale_length=unit_scale),
    )
    return SimpleNamespace(scene=scene)


class DirectCalibrationTests(unittest.TestCase):
    source = Vector((0.0, 0.0, 0.0))
    receiver = Vector((1.0, 0.0, 0.0))

    def test_blocked_direct_does_not_calibrate_from_a_reflection(self):
        context = _context()
        ir = np.zeros((16, 12000), dtype=np.float32)
        ir[0, 5000] = 0.75
        original = ir.copy()

        calibrated, message, applied = calibrate_direct_1_over_r(
            ir, context, self.source, self.receiver
        )

        self.assertFalse(applied)
        self.assertIn("no direct arrival", message)
        np.testing.assert_array_equal(calibrated, original)

    def test_calibration_uses_scene_scale_and_preserves_relative_levels(self):
        context = _context(unit_scale=2.0)
        ir = np.zeros((16, 12000), dtype=np.float32)
        direct_sample = int(round(
            (self.receiver - self.source).length
            / speed_of_sound_bu(context)
            * context.scene.airt_sr
        ))
        ir[0, direct_sample] = 0.25
        ir[0, 5000] = 0.1

        calibrated, _message, applied = calibrate_direct_1_over_r(
            ir, context, self.source, self.receiver
        )

        self.assertTrue(applied)
        self.assertAlmostEqual(float(calibrated[0, direct_sample]), 0.5, places=6)
        self.assertAlmostEqual(float(calibrated[0, 5000]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
