"""Direct-path and output-content tests for the unified engine."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.ambisonic import AmbisonicEncoder  # noqa: E402
from ir_raytracer.core.ray_tracer import (  # noqa: E402
    AcousticRenderConfig,
    AmbisonicIREngine,
)
from ir_raytracer.utils.scene_utils import AcousticScene  # noqa: E402


def _config(content: str) -> AcousticRenderConfig:
    return AcousticRenderConfig(
        ray_count=1,
        max_bounces=0,
        sample_rate=48000,
        duration_seconds=0.1,
        output_content=content,
        early_reflections=True,
        seed=1,
        min_energy=1e-8,
        rr_enabled=False,
        rr_start=20,
        rr_survival=0.97,
        specular_roughness_rad=0.1,
        unit_scale=1.0,
        speed_of_sound_bu=343.0,
        air_enabled=False,
        air_temperature_c=20.0,
        air_humidity_pct=50.0,
        air_pressure_kpa=101.325,
        diffraction_enabled=False,
        diffraction_paths=1,
        diffraction_max_angle_rad=0.8,
        early_gain_db=0.0,
        diffuse_gain_db=0.0,
        encoder=AmbisonicEncoder(),
    )


class DirectPathTests(unittest.TestCase):
    source = Vector((0.0, -1.0, 0.0))
    receiver = Vector((0.0, 0.0, 0.0))

    def _render(self, content):
        engine = AmbisonicIREngine(
            None,
            _config(content),
            AcousticScene(bvh=None, faces=[]),
        )
        return engine.render(self.source, self.receiver)

    def test_full_ir_contains_one_unit_direct_arrival_at_one_metre(self):
        result = self._render('FULL')
        self.assertEqual(result.synthesis.direct_events, 1)
        self.assertEqual(result.synthesis.early_events, 0)
        self.assertEqual(result.synthesis.diffuse_events, 0)
        self.assertAlmostEqual(float(np.sum(result.ir[0])), 1.0, places=5)
        expected_sample = int(round(48000.0 / 343.0))
        self.assertLessEqual(abs(int(np.argmax(np.abs(result.ir[0]))) - expected_sample), 1)

    def test_reflections_and_diffuse_modes_exclude_direct_sound(self):
        for content in ('REFLECTIONS', 'DIFFUSE'):
            with self.subTest(content=content):
                result = self._render(content)
                self.assertEqual(result.synthesis.direct_events, 0)
                np.testing.assert_array_equal(result.ir, np.zeros_like(result.ir))

    def test_direct_pressure_obeys_inverse_distance(self):
        engine = AmbisonicIREngine(None, _config('FULL'), AcousticScene(None, []))
        one_metre = engine.render(Vector((0.0, -1.0, 0.0)), self.receiver).ir
        engine = AmbisonicIREngine(None, _config('FULL'), AcousticScene(None, []))
        two_metres = engine.render(Vector((0.0, -2.0, 0.0)), self.receiver).ir
        self.assertAlmostEqual(
            float(np.sum(two_metres[0]))
            / float(np.sum(one_metre[0])),
            0.5,
            places=5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
