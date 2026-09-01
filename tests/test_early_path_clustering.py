"""Regression tests for tessellation-invariant deterministic reflections."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import NUM_BANDS  # noqa: E402
from ir_raytracer.core.ray_tracer import (  # noqa: E402
    _EarlyPathCandidate,
    _cluster_unresolved_early_paths,
)
from ir_raytracer.core.synthesis import AcousticEvent  # noqa: E402


def _candidate(delay, direction, energy, surface_id=1):
    return _EarlyPathCandidate(
        AcousticEvent(
            delay_seconds=delay,
            arrival_direction=Vector(direction).normalized(),
            energy_bands=np.full(NUM_BANDS, energy, dtype=np.float32),
            kind='EARLY',
            order=1,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
