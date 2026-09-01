"""Energy-domain transport estimator regression tests."""
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

from ir_raytracer.core.acoustics import MaterialProperties, NUM_BANDS  # noqa: E402
from ir_raytracer.core.ambisonic import AmbisonicEncoder  # noqa: E402
from ir_raytracer.core.ray_tracer import (  # noqa: E402
    AcousticRenderConfig,
    ReceiverPathTracer,
)
from ir_raytracer.utils.scene_utils import AcousticScene  # noqa: E402


def _config(ray_count=1):
    return AcousticRenderConfig(
        ray_count=ray_count,
        max_bounces=1,
        sample_rate=48000,
        duration_seconds=1.0,
        output_content='DIFFUSE',
        early_reflections=False,
        seed=1,
        min_energy=1e-8,
        rr_enabled=False,
        rr_start=20,
        rr_survival=0.97,
        specular_roughness_rad=0.15,
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


class SequenceRNG:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


class RayEnergyTests(unittest.TestCase):
    def test_surface_component_sampling_is_unbiased_in_energy(self):
        material = MaterialProperties(SimpleNamespace(
            absorption_bands=(0.2,) * NUM_BANDS,
            scatter_bands=(0.3,) * NUM_BANDS,
            transmission=0.1,
            transmission_bands=(0.1,) * NUM_BANDS,
        ))
        tracer = ReceiverPathTracer(_config(), AcousticScene(None, []))
        direction = Vector((0.0, 0.0, -1.0))
        normal = Vector((0.0, 0.0, 1.0))
        throughput = np.ones(NUM_BANDS)
        probabilities = np.array((0.1, 0.7 * 0.3, 0.7 * 0.7))

        tracer.rng = SequenceRNG((0.05,))
        _direction, transmission_sample, _transmitted = tracer._sample_surface(
            direction, normal, material, throughput
        )
        tracer.rng = SequenceRNG((0.15, 0.25, 0.5))
        _direction, diffuse_sample, _transmitted = tracer._sample_surface(
            direction, normal, material, throughput
        )
        tracer.rng = SequenceRNG((0.5, 0.25, 0.5))
        _direction, specular_sample, _transmitted = tracer._sample_surface(
            direction, normal, material, throughput
        )

        estimate = (
            probabilities[0] * transmission_sample
            + probabilities[1] * diffuse_sample
            + probabilities[2] * specular_sample
        )
        expected = (
            material.transmission_spectrum
            + material.reflection_spectrum * material.diffuse_fraction
            + material.reflection_spectrum * material.specular_fraction
        )
        np.testing.assert_allclose(estimate, expected, rtol=1e-6, atol=1e-6)

    def test_source_connection_weight_scales_inverse_with_ray_count(self):
        material = MaterialProperties(SimpleNamespace(
            absorption_bands=(0.0,) * NUM_BANDS,
            scatter_bands=(1.0,) * NUM_BANDS,
            transmission=0.0,
            transmission_bands=(0.0,) * NUM_BANDS,
        ))
        energies = []
        for ray_count in (128, 1024):
            tracer = ReceiverPathTracer(_config(ray_count), AcousticScene(None, []))
            event = tracer._source_connection(
                hit_point=Vector((0.0, 0.0, 0.0)),
                normal=Vector((0.0, 0.0, 1.0)),
                reverse_direction=Vector((0.0, 0.0, -1.0)),
                source=Vector((0.0, 0.0, 1.0)),
                path_distance_bu=1.0,
                throughput=np.ones(NUM_BANDS),
                material=material,
                first_direction=Vector((1.0, 0.0, 0.0)),
                bounce=0,
            )
            self.assertIsNotNone(event)
            energies.append(float(event.energy_bands[0]))
            np.testing.assert_allclose(tuple(event.arrival_direction), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(energies[0] / energies[1], 8.0, places=5)

    def test_material_supports_frequency_dependent_transmission(self):
        transmission = tuple(np.linspace(0.05, 0.65, NUM_BANDS))
        material = MaterialProperties(SimpleNamespace(
            absorption_bands=(0.1,) * NUM_BANDS,
            scatter_bands=(0.3,) * NUM_BANDS,
            transmission=0.0,
            transmission_bands=transmission,
        ))
        np.testing.assert_allclose(material.transmission_spectrum, transmission)
        np.testing.assert_allclose(
            material.reflection_spectrum,
            1.0 - np.array(transmission) - 0.1,
            atol=1e-7,
        )

    def test_surface_sampling_preserves_a_reflective_low_frequency_band(self):
        material = MaterialProperties(SimpleNamespace(
            absorption_bands=(0.08, 0.12, 0.30, 0.55, 0.65, 0.70, 0.70),
            scatter_bands=(0.55,) * NUM_BANDS,
            transmission=0.0,
            transmission_bands=(0.0,) * NUM_BANDS,
        ))
        tracer = ReceiverPathTracer(_config(), AcousticScene(None, []))
        direction = Vector((0.0, 0.0, -1.0))
        normal = Vector((0.0, 0.0, 1.0))

        # A draw above the mean reflection (0.56) but below the strongest
        # surviving band (0.92) must continue, avoiding a sparse bass tail.
        tracer.rng = SequenceRNG((0.75, 0.25, 0.5))
        _direction, sampled, _transmitted = tracer._sample_surface(
            direction, normal, material, np.ones(NUM_BANDS)
        )
        self.assertIsNotNone(sampled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
