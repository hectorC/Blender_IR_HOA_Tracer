"""Regression tests for Full IR and Reverb Only direct-path handling.

Run with Blender's Python environment, for example:

    blender --background --factory-startup --python-exit-code 1 \
        --python tests/test_direct_path.py
"""
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
    ForwardRayTracer,
    ReverseRayTracer,
)


class EmptyBVH:
    """A scene with unobstructed line of sight and no reflecting geometry."""

    def ray_cast(self, _origin, _direction):
        return None, None, None, None


class DirectPathConfig:
    """Minimal deterministic config accepted by both tracer implementations."""

    def __init__(self, include_direct: bool, segment_capture: bool = True):
        self.num_rays = 1
        self.max_bounces = 0
        self.sample_rate = 48000
        self.ir_length_samples = 1000
        self.speed_of_sound = 343.0
        self.unit_scale = 1.0
        self.receiver_radius_m = 0.25
        self.receiver_radius = 0.25
        self.angle_tolerance_rad = np.deg2rad(8.0)
        self.specular_roughness_rad = 0.0
        self.segment_capture = segment_capture
        self.rr_enable = False
        self.rr_start_bounce = 40
        self.rr_survive_prob = 0.99
        self.enable_diffraction = False
        self.diffraction_samples = 0
        self.diffraction_max_angle = 0.0
        self.air_enable = False
        self.air_temp_c = 20.0
        self.air_humidity = 50.0
        self.air_pressure_kpa = 101.325
        self.quick_broadband = True
        self.min_throughput = 1e-6
        self.include_direct = include_direct
        self.ambisonic_encoder = AmbisonicEncoder()
        self.eps = 1e-4


class DirectPathTests(unittest.TestCase):
    source = Vector((0.0, 0.0, 0.0))
    receiver = Vector((0.0, -1.0, 0.0))
    direct_direction = [(0.0, -1.0, 0.0)]

    def _trace(self, tracer_type, include_direct: bool) -> np.ndarray:
        tracer = tracer_type(DirectPathConfig(include_direct))
        return tracer.trace_rays(
            self.source,
            self.receiver,
            EmptyBVH(),
            [],
            self.direct_direction,
        )

    def test_forward_full_ir_contains_direct_arrival_exactly_once(self):
        ir = self._trace(ForwardRayTracer, include_direct=True)

        # At one metre, broadband W amplitude is 1/r = 1. Segment capture is
        # deliberately enabled: bounce zero must still not duplicate it.
        self.assertAlmostEqual(float(np.sum(ir[0])), 1.0, places=6)
        self.assertGreater(np.count_nonzero(ir[0]), 0)

    def test_forward_reverb_only_suppresses_zero_bounce_capture(self):
        ir = self._trace(ForwardRayTracer, include_direct=False)
        np.testing.assert_array_equal(ir, np.zeros_like(ir))

    def test_reverse_full_ir_contains_direct_arrival_exactly_once(self):
        ir = self._trace(ReverseRayTracer, include_direct=True)
        self.assertAlmostEqual(float(np.sum(ir[0])), 1.0, places=6)

    def test_reverse_reverb_only_has_no_direct_arrival(self):
        ir = self._trace(ReverseRayTracer, include_direct=False)
        np.testing.assert_array_equal(ir, np.zeros_like(ir))


if __name__ == "__main__":
    unittest.main(verbosity=2)
