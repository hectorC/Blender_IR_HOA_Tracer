"""Source-local, frequency-dependent radiation pattern tests."""
from __future__ import annotations

import os
import sys
import unittest
from math import pi, sin, cos

import numpy as np
from mathutils import Quaternion, Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import NUM_BANDS  # noqa: E402
from ir_raytracer.core.directivity import (  # noqa: E402
    DEFAULT_CUSTOM_SH,
    DIRECTIVITY_STRENGTH_PRESETS,
    SourceDirectivity,
)


class SourceDirectivityTests(unittest.TestCase):
    front = Vector((0.0, -1.0, 0.0))
    rear = Vector((0.0, 1.0, 0.0))
    side = Vector((1.0, 0.0, 0.0))

    def test_omnidirectional_default_is_unity_everywhere(self):
        directivity = SourceDirectivity()
        for direction in (self.front, self.rear, self.side, Vector((1, 2, 3))):
            np.testing.assert_array_equal(
                directivity.pressure_gain(direction),
                np.ones(NUM_BANDS, dtype=np.float32),
            )

    def test_cardioid_has_full_front_half_side_and_quiet_rear(self):
        directivity = SourceDirectivity('CARDIOID')
        np.testing.assert_allclose(directivity.pressure_gain(self.front), 1.0)
        np.testing.assert_allclose(directivity.pressure_gain(self.side), 0.5)
        np.testing.assert_allclose(directivity.pressure_gain(self.rear), 0.0)

    def test_dipole_preserves_opposite_rear_polarity(self):
        directivity = SourceDirectivity('DIPOLE')
        np.testing.assert_allclose(directivity.pressure_gain(self.front), 1.0)
        np.testing.assert_allclose(directivity.pressure_gain(self.side), 0.0)
        np.testing.assert_allclose(directivity.pressure_gain(self.rear), -1.0)
        energy, polarity = directivity.energy_gain_and_polarity(self.rear)
        np.testing.assert_allclose(energy, 1.0)
        np.testing.assert_allclose(polarity, -1.0)

    def test_evaluated_source_rotation_aims_local_minus_y(self):
        rotation = Quaternion(Vector((0.0, 0.0, 1.0)), pi / 2.0)
        directivity = SourceDirectivity('CARDIOID', source_rotation=rotation)
        np.testing.assert_allclose(
            directivity.pressure_gain(Vector((1.0, 0.0, 0.0))), 1.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            directivity.pressure_gain(Vector((-1.0, 0.0, 0.0))), 0.0,
            atol=1e-6,
        )

    def test_loudspeaker_spreads_bass_more_broadly_than_treble(self):
        directivity = SourceDirectivity('LOUDSPEAKER')
        expected_rear = 1.0 - np.asarray(
            DIRECTIVITY_STRENGTH_PRESETS['LOUDSPEAKER']
        )
        np.testing.assert_allclose(
            directivity.pressure_gain(self.rear), expected_rear, rtol=1e-6
        )
        self.assertGreater(expected_rear[0], expected_rear[-1])
        np.testing.assert_allclose(directivity.pressure_gain(self.front), 1.0)

    def test_cone_width_is_six_db_pressure_width(self):
        directivity = SourceDirectivity('FORWARD_CONE', cone_width_deg=90.0)
        half_angle = pi / 4.0
        edge = Vector((sin(half_angle), -cos(half_angle), 0.0))
        np.testing.assert_allclose(
            directivity.pressure_gain(edge), 0.5, atol=1e-6
        )
        np.testing.assert_allclose(directivity.pressure_gain(self.front), 1.0)
        np.testing.assert_allclose(directivity.pressure_gain(self.rear), 0.0)

    def test_frequency_strength_blends_even_and_shaped_response(self):
        strengths = np.linspace(0.0, 1.0, NUM_BANDS)
        directivity = SourceDirectivity(
            'CARDIOID', strength_bands=strengths
        )
        np.testing.assert_allclose(
            directivity.pressure_gain(self.side),
            1.0 - 0.5 * strengths,
            atol=1e-7,
        )

    def test_default_custom_shape_is_peak_normalized_forward_focus(self):
        directivity = SourceDirectivity(
            'CUSTOM_SH', custom_sh=DEFAULT_CUSTOM_SH
        )
        np.testing.assert_allclose(
            directivity.pressure_gain(self.front), 1.0, atol=1e-6
        )
        np.testing.assert_allclose(
            directivity.pressure_gain(self.rear), 0.0, atol=1e-6
        )
        metadata = directivity.metadata()
        self.assertEqual(metadata['forward_axis'], '-Y')
        self.assertEqual(metadata['normalization'], 'unity_peak_pressure')
        self.assertEqual(
            len(metadata['custom_acn_sn3d_pressure_coefficients']), 16
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
