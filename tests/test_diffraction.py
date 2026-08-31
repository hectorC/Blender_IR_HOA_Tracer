"""Regression tests for bounded single-edge diffraction."""
from __future__ import annotations

import os
import sys
import unittest
from math import pi
from types import SimpleNamespace

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import NUM_BANDS  # noqa: E402
from ir_raytracer.core.diffraction import (  # noqa: E402
    DiffractionEdge,
    DiffractionEdgeIndex,
    extract_diffraction_edges,
    find_diffraction_paths,
    maekawa_diffraction_gains,
)
from ir_raytracer.core.ray_tracer import ForwardRayTracer  # noqa: E402


class DiffractionProbe(ForwardRayTracer):
    def __init__(self, config):
        super().__init__(config)
        self.emissions = []

    def emit_impulse(
        self, band_amplitude, distance_bu, incoming_direction, amplitude_scalar
    ):
        self.emissions.append((
            band_amplitude.copy(),
            float(distance_bu),
            incoming_direction.copy(),
            float(amplitude_scalar),
        ))
        return True


class DiffractionTests(unittest.TestCase):
    origin = Vector((-1.0, 0.0, 0.0))
    receiver = Vector((1.0, 0.0, 0.0))
    edge = DiffractionEdge(
        Vector((0.0, 0.2, -1.0)),
        Vector((0.0, 0.2, 1.0)),
    )

    def test_coplanar_triangulation_edge_is_not_a_diffraction_edge(self):
        vertices = [
            Vector((-1.0, -1.0, 0.0)),
            Vector((1.0, -1.0, 0.0)),
            Vector((1.0, 1.0, 0.0)),
            Vector((-1.0, 1.0, 0.0)),
        ]
        edges = extract_diffraction_edges(
            vertices,
            ((0, 1, 2), (0, 2, 3)),
        )
        self.assertEqual(len(edges), 4)

    def test_cube_has_twelve_sharp_edges(self):
        vertices = [Vector(coordinate) for coordinate in (
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
        )]
        polygons = (
            (0, 1, 2, 3), (4, 7, 6, 5),
            (0, 4, 5, 1), (1, 5, 6, 2),
            (2, 6, 7, 3), (4, 0, 3, 7),
        )
        self.assertEqual(len(extract_diffraction_edges(vertices, polygons)), 12)

    def test_visible_edge_produces_a_longer_bounded_path(self):
        paths = find_diffraction_paths(
            self.origin,
            self.receiver,
            DiffractionEdgeIndex([self.edge]),
            bvh=None,
            unit_scale=1.0,
            max_angle_rad=pi / 4.0,
            max_paths=1,
        )
        self.assertEqual(len(paths), 1)
        self.assertGreater(paths[0].distance_bu, 2.0)
        self.assertGreater(paths[0].path_difference_m, 0.0)
        self.assertLess(paths[0].bend_angle_rad, pi / 4.0)

    def test_diffraction_is_low_frequency_weighted_and_capped(self):
        gains = maekawa_diffraction_gains(0.1, 343.0)
        self.assertGreater(float(gains[0]), float(gains[-1]))
        self.assertTrue(np.all(gains <= 1.0))
        minimum_gain = 10.0 ** (-25.0 / 20.0)
        heavily_shadowed = maekawa_diffraction_gains(100.0, 343.0)
        np.testing.assert_allclose(
            heavily_shadowed,
            np.full(NUM_BANDS, minimum_gain),
            rtol=1e-6,
        )

    def test_primary_diffraction_emits_one_frequency_shaped_path(self):
        config = SimpleNamespace(
            ir_length_samples=2048,
            enable_diffraction=True,
            diffraction_samples=1,
            diffraction_max_angle=pi / 4.0,
            diffraction_edge_index=DiffractionEdgeIndex([self.edge]),
            unit_scale=1.0,
            speed_of_sound=343.0,
            receiver_radius_m=0.1,
            eps=1e-4,
        )
        tracer = DiffractionProbe(config)

        wrote = tracer._emit_primary_diffraction(
            self.origin,
            self.receiver,
            bvh=None,
            throughput=np.ones(NUM_BANDS, dtype=np.float32),
        )

        self.assertTrue(wrote)
        self.assertEqual(len(tracer.emissions), 1)
        bands, distance, _incoming, spreading = tracer.emissions[0]
        self.assertGreater(float(bands[0]), float(bands[-1]))
        self.assertGreater(distance, 2.0)
        self.assertAlmostEqual(spreading, 1.0 / distance, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
