"""Regression tests for calibrated construction-specific material presets."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import (  # noqa: E402
    DEFAULT_SCATTER_SPECTRUM,
    MATERIAL_PRESET_DATA,
    MATERIAL_PRESETS,
    MaterialProperties,
    NUM_BANDS,
)


class MaterialPresetTests(unittest.TestCase):
    def test_coherent_reflection_matches_normal_energy_and_changes_with_angle(self):
        material = MaterialProperties(SimpleNamespace(
            absorption_bands=(0.75,) * NUM_BANDS,
            scatter_bands=(0.0,) * NUM_BANDS,
            transmission=0.0,
            transmission_bands=(0.0,) * NUM_BANDS,
        ))
        normal = material.specular_pressure_spectrum(1.0)
        grazing = material.specular_pressure_spectrum(0.01)

        np.testing.assert_allclose(np.square(normal), 0.25, atol=1e-6)
        self.assertTrue(np.all(grazing < 0.0))
        self.assertTrue(np.all(np.abs(grazing) > np.abs(normal)))

    def test_presets_have_unique_valid_seven_band_spectra(self):
        identifiers = []
        for identifier, _label, description, absorption, scatter in (
            MATERIAL_PRESET_DATA
        ):
            identifiers.append(identifier)
            self.assertTrue(description)
            self.assertEqual(len(absorption), NUM_BANDS)
            self.assertEqual(len(scatter), NUM_BANDS)
            self.assertTrue(np.all(np.asarray(absorption) >= 0.0))
            self.assertTrue(np.all(np.asarray(absorption) <= 1.0))
            self.assertTrue(np.all(np.asarray(scatter) >= 0.0))
            self.assertTrue(np.all(np.asarray(scatter) <= 1.0))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), set(MATERIAL_PRESETS))

    def test_legacy_identifiers_remain_available(self):
        for identifier in (
            'WOOD', 'CONCRETE', 'PLASTER', 'CARPET', 'TILE', 'BRICK'
        ):
            self.assertIn(identifier, MATERIAL_PRESETS)

    def test_smooth_hard_surfaces_use_low_unmodeled_scattering(self):
        for identifier in ('CONCRETE_SMOOTH', 'TILE', 'GLASS', 'METAL'):
            spectrum = MATERIAL_PRESETS[identifier]['scatter_spectrum']
            self.assertLess(spectrum[2], 0.02)
            self.assertLess(spectrum[3], 0.03)

    def test_rough_cave_rock_differs_mainly_in_scattering(self):
        dense = MATERIAL_PRESETS['ROCK_DENSE']
        cave = MATERIAL_PRESETS['ROCK_CAVE']

        self.assertLess(max(cave['absorption_spectrum']), 0.11)
        self.assertGreater(
            np.mean(cave['scatter_spectrum']),
            np.mean(dense['scatter_spectrum']) * 3.0,
        )
        self.assertGreater(
            cave['scatter_spectrum'][-1],
            cave['scatter_spectrum'][0] * 5.0,
        )

    def test_custom_default_scattering_is_gentle_and_frequency_dependent(self):
        self.assertEqual(len(DEFAULT_SCATTER_SPECTRUM), NUM_BANDS)
        self.assertLess(np.mean(DEFAULT_SCATTER_SPECTRUM), 0.1)
        self.assertGreater(
            DEFAULT_SCATTER_SPECTRUM[-1],
            DEFAULT_SCATTER_SPECTRUM[0],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
