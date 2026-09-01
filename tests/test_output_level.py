"""Output-level policy tests."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.ui.operators import prepare_ir_for_export  # noqa: E402


class OutputLevelTests(unittest.TestCase):
    def test_preserve_mode_does_not_change_relative_or_absolute_level(self):
        ir = np.array(((0.25, -0.5, 0.1),), dtype=np.float32)
        output, gain = prepare_ir_for_export(ir, 'PRESERVE', -1.0)
        self.assertEqual(gain, 1.0)
        np.testing.assert_array_equal(output, ir)
        self.assertIsNot(output, ir)

    def test_peak_mode_hits_selected_headroom(self):
        ir = np.array(((0.25, -0.5, 0.1),), dtype=np.float32)
        output, gain = prepare_ir_for_export(ir, 'PEAK', -6.0)
        target = 10.0 ** (-6.0 / 20.0)
        self.assertAlmostEqual(float(np.max(np.abs(output))), target, places=6)
        self.assertAlmostEqual(gain, target / 0.5, places=6)
        self.assertAlmostEqual(float(output[0, 0] / output[0, 1]), -0.5, places=6)

    def test_silent_ir_is_not_amplified(self):
        ir = np.zeros((16, 128), dtype=np.float32)
        output, gain = prepare_ir_for_export(ir, 'PEAK', -1.0)
        self.assertEqual(gain, 1.0)
        np.testing.assert_array_equal(output, ir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
