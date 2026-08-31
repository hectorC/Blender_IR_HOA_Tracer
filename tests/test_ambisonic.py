"""Regression tests for the built-in ACN/SN3D encoder.

Run with Blender's Python environment, for example:

    blender --background --factory-startup --python-exit-code 1 \
        --python tests/test_ambisonic.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.ambisonic import (  # noqa: E402
    AmbisonicEncoder,
    apply_orientation_transform,
    encode_ambisonics_3rd_order,
)


def _associated_legendre_no_cs(order: int, degree: int, z: float) -> float:
    """Evaluate P_order^degree(z) without the Condon-Shortley phase."""
    p_mm = 1.0
    if degree > 0:
        root = math.sqrt(max(0.0, 1.0 - z * z))
        for factor in range(1, 2 * degree, 2):
            p_mm *= factor * root

    if order == degree:
        return p_mm

    p_m1_m = z * (2 * degree + 1) * p_mm
    if order == degree + 1:
        return p_m1_m

    p_lm2 = p_mm
    p_lm1 = p_m1_m
    for current_order in range(degree + 2, order + 1):
        p_lm = (
            (2 * current_order - 1) * z * p_lm1
            - (current_order + degree - 1) * p_lm2
        ) / (current_order - degree)
        p_lm2, p_lm1 = p_lm1, p_lm
    return p_lm1


def _reference_acn_sn3d(direction: Vector) -> np.ndarray:
    """Independent recurrence-based real SH reference implementation."""
    x, y, z = (float(component) for component in direction)
    radius = math.sqrt(x * x + y * y + z * z)
    if radius <= 1e-12:
        result = np.zeros(16, dtype=np.float64)
        result[0] = 1.0
        return result

    x /= radius
    y /= radius
    z /= radius
    azimuth = math.atan2(y, x)
    result = []

    for order in range(4):
        for degree in range(-order, order + 1):
            abs_degree = abs(degree)
            legendre = _associated_legendre_no_cs(order, abs_degree, z)
            normalization = math.sqrt(
                math.factorial(order - abs_degree)
                / math.factorial(order + abs_degree)
            )

            if degree < 0:
                value = (
                    math.sqrt(2.0)
                    * normalization
                    * legendre
                    * math.sin(abs_degree * azimuth)
                )
            elif degree == 0:
                value = normalization * legendre
            else:
                value = (
                    math.sqrt(2.0)
                    * normalization
                    * legendre
                    * math.cos(degree * azimuth)
                )
            result.append(value)

    return np.asarray(result, dtype=np.float64)


class AmbisonicEncodingTests(unittest.TestCase):
    def assert_vector_close(self, actual, expected, tolerance=1e-6):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=0.0, atol=tolerance
        )

    def test_first_order_cardinal_directions_have_ambix_polarity(self):
        # ACN channels 1, 2, 3 are AmbiX Y (left), Z (up), X (front).
        cases = (
            (Vector((1.0, 0.0, 0.0)), (0.0, 0.0, 1.0)),
            (Vector((-1.0, 0.0, 0.0)), (0.0, 0.0, -1.0)),
            (Vector((0.0, 1.0, 0.0)), (1.0, 0.0, 0.0)),
            (Vector((0.0, -1.0, 0.0)), (-1.0, 0.0, 0.0)),
            (Vector((0.0, 0.0, 1.0)), (0.0, 1.0, 0.0)),
            (Vector((0.0, 0.0, -1.0)), (0.0, -1.0, 0.0)),
        )
        for direction, expected_first_order in cases:
            with self.subTest(direction=tuple(direction)):
                encoded = encode_ambisonics_3rd_order(direction)
                self.assertEqual(float(encoded[0]), 1.0)
                self.assert_vector_close(encoded[1:4], expected_first_order)

    def test_explicit_equations_match_independent_recurrence(self):
        directions = (
            Vector((1.0, 2.0, 3.0)),
            Vector((-0.3, 0.7, 0.2)),
            Vector((0.4, -0.1, -0.9)),
            Vector((-2.0, -3.0, 0.5)),
        )
        for direction in directions:
            with self.subTest(direction=tuple(direction)):
                actual = encode_ambisonics_3rd_order(direction)
                expected = _reference_acn_sn3d(direction)
                self.assert_vector_close(actual, expected)

    def test_each_sn3d_order_has_unit_point_source_energy(self):
        encoded = encode_ambisonics_3rd_order(Vector((0.31, -0.47, 0.82)))
        for order in range(4):
            start = order * order
            end = (order + 1) * (order + 1)
            order_energy = float(np.dot(encoded[start:end], encoded[start:end]))
            with self.subTest(order=order):
                self.assertAlmostEqual(order_energy, 1.0, places=6)

    def test_scaled_and_zero_directions_are_safe(self):
        unit = encode_ambisonics_3rd_order(Vector((1.0, 2.0, 3.0)))
        scaled = encode_ambisonics_3rd_order(Vector((10.0, 20.0, 30.0)))
        self.assert_vector_close(unit, scaled)

        zero = encode_ambisonics_3rd_order(Vector((0.0, 0.0, 0.0)))
        expected = np.zeros(16, dtype=np.float32)
        expected[0] = 1.0
        self.assert_vector_close(zero, expected)


class OrientationTransformTests(unittest.TestCase):
    def assert_vector_close(self, actual, expected, tolerance=1e-6):
        np.testing.assert_allclose(
            np.asarray(tuple(actual)), np.asarray(expected), rtol=0.0, atol=tolerance
        )

    def test_blender_reference_axes_map_to_ambix_axes(self):
        cases = (
            (Vector((0.0, -1.0, 0.0)), (1.0, 0.0, 0.0)),  # front
            (Vector((1.0, 0.0, 0.0)), (0.0, 1.0, 0.0)),   # left
            (Vector((0.0, 0.0, 1.0)), (0.0, 0.0, 1.0)),   # up
        )
        for direction, expected in cases:
            with self.subTest(direction=tuple(direction)):
                self.assert_vector_close(apply_orientation_transform(direction), expected)

    def test_yaw_and_z_flip_are_applied_in_ambix_space(self):
        front = Vector((0.0, -1.0, 0.0))
        self.assert_vector_close(
            apply_orientation_transform(front, yaw_offset_deg=90.0),
            (0.0, 1.0, 0.0),
        )
        self.assert_vector_close(
            apply_orientation_transform(Vector((0.0, 0.0, 1.0)), invert_z=True),
            (0.0, 0.0, -1.0),
        )

    def test_encoder_composes_orientation_and_spherical_harmonics(self):
        encoded = AmbisonicEncoder().encode(Vector((0.0, -1.0, 0.0)))
        self.assertAlmostEqual(float(encoded[3]), 1.0, places=6)
        self.assertAlmostEqual(float(encoded[1]), 0.0, places=6)
        self.assertAlmostEqual(float(encoded[2]), 0.0, places=6)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Ambisonic regression tests failed")
