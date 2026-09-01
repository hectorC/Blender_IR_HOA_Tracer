"""Regression tests for tessellation-invariant deterministic reflections."""
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

from ir_raytracer.core.acoustics import NUM_BANDS  # noqa: E402
from ir_raytracer.core.ray_tracer import (  # noqa: E402
    _EarlyPathCandidate,
    _build_specular_surfaces,
    _cluster_unresolved_early_paths,
    _surface_sequence_count,
)
from ir_raytracer.core.synthesis import AcousticEvent  # noqa: E402
from ir_raytracer.utils.scene_utils import AcousticFace, AcousticScene  # noqa: E402


def _candidate(delay, direction, energy, surface_id=1, order=1):
    return _EarlyPathCandidate(
        AcousticEvent(
            delay_seconds=delay,
            arrival_direction=Vector(direction).normalized(),
            energy_bands=np.full(NUM_BANDS, energy, dtype=np.float32),
            kind='EARLY',
            order=order,
        ),
        surface_id,
    )


class EarlyPathClusteringTests(unittest.TestCase):
    def test_unresolved_paths_from_one_surface_are_consolidated(self):
        candidates = [
            _candidate(0.40000, (0.00, 0.00, -1.0), 0.30),
            _candidate(0.40005, (0.05, 0.00, -1.0), 0.25),
            _candidate(0.40010, (-0.05, 0.00, -1.0), 0.20),
        ]
        events = _cluster_unresolved_early_paths(candidates)

        self.assertEqual(len(events), 1)
        np.testing.assert_allclose(events[0].energy_bands, 0.30)
        self.assertLess(abs(events[0].arrival_direction.x), 0.02)
        self.assertLess(events[0].arrival_direction.z, -0.99)

    def test_resolvable_or_separate_surface_paths_remain_distinct(self):
        candidates = [
            _candidate(0.4000, (0.0, 0.0, -1.0), 0.30, surface_id=1),
            _candidate(0.4010, (0.0, 0.0, -1.0), 0.30, surface_id=1),
            _candidate(0.4000, (0.5, 0.0, -1.0), 0.30, surface_id=1),
            _candidate(0.4000, (0.0, 0.0, -1.0), 0.30, surface_id=2),
        ]
        events = _cluster_unresolved_early_paths(candidates)
        self.assertEqual(len(events), 4)

    def test_reflection_orders_are_never_clustered_together(self):
        candidates = [
            _candidate(0.4, (0.0, 0.0, -1.0), 0.3, order=1),
            _candidate(0.4, (0.0, 0.0, -1.0), 0.3, order=2),
        ]
        events = _cluster_unresolved_early_paths(candidates)
        self.assertEqual(len(events), 2)

    def test_coplanar_triangles_share_one_sequence_surface(self):
        obj = SimpleNamespace(name="Triangulated Wall")
        material_a = SimpleNamespace(
            name="Left Finish", airt_acoustic_enabled=True
        )
        material_b = SimpleNamespace(
            name="Right Finish", airt_acoustic_enabled=True
        )
        faces = [
            AcousticFace(
                vertices=(
                    Vector((0.0, 0.0, 0.0)),
                    Vector((1.0, 0.0, 0.0)),
                    Vector((1.0, 1.0, 0.0)),
                ),
                normal=Vector((0.0, 0.0, 1.0)),
                object_ref=obj,
                material_ref=material_a,
            ),
            AcousticFace(
                vertices=(
                    Vector((0.0, 0.0, 0.0)),
                    Vector((0.0, 1.0, 0.0)),
                    Vector((1.0, 1.0, 0.0)),
                ),
                normal=Vector((0.0, 0.0, -1.0)),
                object_ref=obj,
                material_ref=material_b,
            ),
        ]
        surfaces = _build_specular_surfaces(AcousticScene(None, faces))
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surfaces[0].face_indices, [0, 1])

    def test_sequence_budget_count_rejects_immediate_plane_repeats(self):
        self.assertEqual(_surface_sequence_count(6, 1), 6)
        self.assertEqual(_surface_sequence_count(6, 2), 30)
        self.assertEqual(_surface_sequence_count(6, 3), 150)


if __name__ == "__main__":
    unittest.main(verbosity=2)
