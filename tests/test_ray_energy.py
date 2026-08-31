"""Regression tests for ray-budget and path-throughput estimation."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import MaterialProperties, NUM_BANDS  # noqa: E402
from ir_raytracer.core.ambisonic import AmbisonicEncoder  # noqa: E402
from ir_raytracer.core.ray_tracer import (  # noqa: E402
    ForwardRayTracer,
    ReverseRayTracer,
)


def _config(**overrides):
    values = dict(
        num_rays=1,
        max_bounces=1,
        sample_rate=48000,
        ir_length_samples=1024,
        speed_of_sound=343.0,
        unit_scale=1.0,
        receiver_radius_m=0.25,
        receiver_radius=0.25,
        angle_tolerance_rad=np.deg2rad(8.0),
        specular_roughness_rad=0.0,
        segment_capture=False,
        rr_enable=False,
        rr_start_bounce=40,
        rr_survive_prob=0.99,
        enable_diffraction=False,
        diffraction_samples=0,
        diffraction_max_angle=0.0,
        air_enable=False,
        air_temp_c=20.0,
        air_humidity=50.0,
        air_pressure_kpa=101.325,
        quick_broadband=True,
        min_throughput=1e-8,
        include_direct=False,
        ambisonic_encoder=AmbisonicEncoder(),
        eps=1e-4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ThroughputProbe(ReverseRayTracer):
    """Replace geometry traversal with one fixed contribution per ray."""

    def _trace_single_ray(
        self,
        _direction,
        _start_pos,
        _target,
        _bvh,
        _obj_map,
        initial_throughput,
        _arrival_direction,
    ):
        self.ir[0, 0] += initial_throughput[0]


class ConnectionProbe(ReverseRayTracer):
    def __init__(self, config):
        super().__init__(config)
        self.emission = None

    def emit_impulse(
        self, band_amplitude, distance_bu, incoming_direction, amplitude_scalar
    ):
        self.emission = (
            band_amplitude.copy(),
            distance_bu,
            incoming_direction.copy(),
            amplitude_scalar,
        )
        return True


class RayEnergyTests(unittest.TestCase):
    def test_reverse_estimate_is_invariant_to_ray_budget(self):
        results = []
        for count in (1, 8, 128):
            tracer = ThroughputProbe(_config(num_rays=count))
            directions = [(1.0, 0.0, 0.0)] * count
            ir = tracer.trace_rays(
                Vector((0.0, 0.0, 0.0)),
                Vector((1.0, 0.0, 0.0)),
                object(),
                [],
                directions,
            )
            results.append(float(ir[0, 0]))

        np.testing.assert_allclose(results, (1.0, 1.0, 1.0), atol=1e-6)

    def test_surface_branch_sampling_is_pressure_unbiased(self):
        material_object = SimpleNamespace(
            absorption_bands=(0.2,) * NUM_BANDS,
            scatter_bands=(0.3,) * NUM_BANDS,
            transmission=0.1,
        )
        material = MaterialProperties(material_object)
        tracer = ForwardRayTracer(_config())
        direction = Vector((0.0, 0.0, -1.0))
        normal = Vector((0.0, 0.0, 1.0))
        throughput = np.ones(NUM_BANDS, dtype=np.float32)

        energy = np.array((0.1, 0.7 * 0.3, 0.7 * 0.7))
        probabilities = energy / np.sum(energy)
        samples = []
        branch_random_values = (
            probabilities[0] * 0.5,
            probabilities[0] + probabilities[1] * 0.5,
            probabilities[0] + probabilities[1] + probabilities[2] * 0.5,
        )
        for random_value in branch_random_values:
            with patch(
                "ir_raytracer.core.ray_tracer.random.random",
                return_value=float(random_value),
            ):
                _outgoing, sampled = tracer._sample_surface_scatter(
                    direction, normal, material, throughput
                )
            samples.append(sampled)

        expected = (
            material.transmission_amplitude
            + material.diffuse_amplitude
            + material.specular_amplitude
        )
        estimate = sum(
            probability * sample
            for probability, sample in zip(probabilities, samples)
        )
        np.testing.assert_allclose(estimate, expected, rtol=1e-6, atol=1e-6)

    def test_geometric_spreading_uses_blender_unit_scale(self):
        tracer = ForwardRayTracer(
            _config(unit_scale=0.1, receiver_radius_m=0.01, receiver_radius=0.1)
        )
        self.assertAlmostEqual(tracer._geometric_spreading(10.0), 1.0)

    def test_reverse_connection_keeps_receiver_arrival_direction(self):
        tracer = ConnectionProbe(_config())
        material_object = SimpleNamespace(
            absorption_bands=(0.0,) * NUM_BANDS,
            scatter_bands=(0.0,) * NUM_BANDS,
            transmission=0.0,
        )
        tracer._check_source_connection(
            hit_point=Vector((0.0, 0.0, 0.0)),
            normal=Vector((0.0, 1.0, 0.0)),
            source=Vector((0.0, 1.0, 0.0)),
            throughput=np.ones(NUM_BANDS, dtype=np.float32),
            material=MaterialProperties(material_object),
            path_length=1.0,
            bvh=None,
            ray_direction=Vector((0.0, -1.0, 0.0)),
            arrival_direction=Vector((1.0, 0.0, 0.0)),
            bounce=0,
        )

        self.assertIsNotNone(tracer.emission)
        np.testing.assert_allclose(tuple(tracer.emission[2]), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(tracer.emission[3], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
