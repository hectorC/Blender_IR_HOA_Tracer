"""Tests for energy-to-pressure ambisonic synthesis."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ir_raytracer.core.acoustics import (  # noqa: E402
    BAND_CENTERS_HZ,
    NUM_BANDS,
    design_power_complementary_filter_bank,
)
from ir_raytracer.core.ambisonic import AmbisonicEncoder  # noqa: E402
from ir_raytracer.core.synthesis import (  # noqa: E402
    AcousticEvent,
    synthesize_ambisonic_ir,
)


class SynthesisTests(unittest.TestCase):
    def test_filter_bank_preserves_independent_band_power_across_sample_rates(self):
        for sample_rate in (8000, 16000, 44100, 48000, 96000, 192000):
            with self.subTest(sample_rate=sample_rate):
                kernels, delay = design_power_complementary_filter_bank(sample_rate)
                self.assertEqual(kernels.shape, (NUM_BANDS, 2 * delay + 1))
                np.testing.assert_allclose(kernels, kernels[:, ::-1], atol=1e-7)
                response = np.fft.rfft(kernels, n=262144, axis=1)
                power = np.sum(np.abs(response) ** 2, axis=0)
                self.assertLess(float(np.max(np.abs(10 * np.log10(power)))), 0.15)

    def test_diffuse_synthesis_preserves_requested_energy_spectrum(self):
        sample_rate = 48000
        length = 8192
        energies = np.array((0.2, 0.5, 0.8, 0.3, 0.9, 0.6, 0.4))
        power = np.zeros(length // 2 + 1)
        # Isolating each independent band measures ensemble power exactly,
        # without relying on a noisy finite random-noise realization.
        for band in range(NUM_BANDS):
            energy = np.zeros(NUM_BANDS)
            energy[band] = energies[band]
            event = AcousticEvent(
                delay_seconds=4096 / sample_rate,
                arrival_direction=Vector((0, 1, 0)),
                energy_bands=energy, kind='DIFFUSE',
            )
            ir, _ = synthesize_ambisonic_ir(
                [event], sample_rate, length / sample_rate, AmbisonicEncoder()
            )
            power += np.abs(np.fft.rfft(ir[0])) ** 2
        frequencies = np.fft.rfftfreq(length, 1 / sample_rate)
        expected = np.full(frequencies.shape, energies[-1])
        expected[frequencies <= BAND_CENTERS_HZ[0]] = energies[0]
        for band in range(NUM_BANDS - 1):
            low, high = BAND_CENTERS_HZ[band:band + 2]
            mask = (frequencies > low) & (frequencies <= high)
            angle = 0.5 * np.pi * np.log2(frequencies[mask] / low)
            expected[mask] = (
                energies[band] * np.cos(angle) ** 2
                + energies[band + 1] * np.sin(angle) ** 2
            )
        self.assertLess(float(np.max(np.abs(10 * np.log10(power / expected)))), 0.2)

    def test_fractional_arrival_preserves_hoa_relationships_for_every_event_kind(self):
        encoder = AmbisonicEncoder()
        direction = Vector((0.4, -0.3, 0.8)).normalized()
        encoded = encoder.encode(direction)
        for kind in ('DIRECT', 'EARLY', 'DIFFUSE'):
            with self.subTest(kind=kind):
                event = AcousticEvent(
                    delay_seconds=4096.375 / 48000,
                    arrival_direction=direction,
                    energy_bands=np.linspace(0.1, 0.7, NUM_BANDS), kind=kind,
                )
                ir, _ = synthesize_ambisonic_ir(
                    [event], 48000, 8192 / 48000, encoder
                )
                np.testing.assert_allclose(
                    ir, encoded[:, None] * ir[0], atol=2e-7, rtol=2e-5
                )

    def test_diffuse_fractional_delay_preserves_treble_and_phase(self):
        outputs = []
        energy = np.zeros(NUM_BANDS)
        energy[-1] = 1.0
        for delay in (4096.0, 4096.5):
            event = AcousticEvent(
                delay_seconds=delay / 48000,
                arrival_direction=Vector((0, 1, 0)),
                energy_bands=energy, kind='DIFFUSE',
            )
            ir, _ = synthesize_ambisonic_ir(
                [event], 48000, 8192 / 48000, AmbisonicEncoder()
            )
            outputs.append(np.fft.rfft(ir[0]))
        frequencies = np.fft.rfftfreq(8192, 1 / 48000)
        mask = (frequencies >= 10000) & (frequencies <= 21600)
        ratio = outputs[1][mask] / outputs[0][mask]
        self.assertLess(float(np.max(np.abs(20 * np.log10(np.abs(ratio))))), 0.002)
        expected = np.exp(-2j * np.pi * frequencies[mask] * 0.5 / 48000)
        self.assertLess(float(np.max(np.abs(np.angle(ratio / expected)))), 0.0001)

    def test_diffuse_output_boundaries_match_a_longer_render_crop(self):
        sample_rate = 48000
        length = 4096
        offset = 4096
        encoder = AmbisonicEncoder()
        for delay in (0.25, length - 0.25):
            with self.subTest(delay=delay):
                kwargs = dict(
                    arrival_direction=Vector((0.2, 0.8, -0.3)),
                    energy_bands=np.linspace(0.1, 0.7, NUM_BANDS),
                    kind='DIFFUSE',
                )
                short, _ = synthesize_ambisonic_ir(
                    [AcousticEvent(delay / sample_rate, **kwargs)],
                    sample_rate, length / sample_rate, encoder,
                )
                long, _ = synthesize_ambisonic_ir(
                    [AcousticEvent((delay + offset) / sample_rate, **kwargs)],
                    sample_rate, (length + 2 * offset) / sample_rate, encoder,
                )
                np.testing.assert_allclose(
                    short, long[:, offset:offset + length], atol=1e-7
                )

    def test_diffuse_filter_delay_is_compensated(self):
        energy = np.zeros(NUM_BANDS)
        energy[-1] = 1.0
        event = AcousticEvent(
            delay_seconds=4096 / 48000,
            arrival_direction=Vector((0, 1, 0)),
            energy_bands=energy, kind='DIFFUSE',
        )
        ir, _ = synthesize_ambisonic_ir(
            [event], 48000, 8192 / 48000, AmbisonicEncoder()
        )
        self.assertEqual(int(np.argmax(np.abs(ir[0]))), 4096)

    def test_flat_deterministic_event_keeps_delay_and_pressure(self):
        event = AcousticEvent(
            delay_seconds=0.01,
            arrival_direction=Vector((1.0, 0.0, 0.0)),
            energy_bands=np.full(NUM_BANDS, 0.25, dtype=np.float32),
            kind='DIRECT',
        )
        ir, stats = synthesize_ambisonic_ir(
            [event], 48000, 0.05, AmbisonicEncoder()
        )
        self.assertEqual(stats.direct_events, 1)
        self.assertEqual(int(np.argmax(np.abs(ir[0]))), 480)
        self.assertAlmostEqual(float(ir[0, 480]), 0.5, places=5)

    def test_signed_directivity_reverses_deterministic_pressure(self):
        positive = AcousticEvent(
            delay_seconds=0.01,
            arrival_direction=Vector((1.0, 0.0, 0.0)),
            energy_bands=np.full(NUM_BANDS, 0.25, dtype=np.float32),
            kind='DIRECT',
            pressure_sign_bands=np.ones(NUM_BANDS, dtype=np.float32),
        )
        negative = AcousticEvent(
            delay_seconds=positive.delay_seconds,
            arrival_direction=positive.arrival_direction,
            energy_bands=positive.energy_bands,
            kind='DIRECT',
            pressure_sign_bands=-np.ones(NUM_BANDS, dtype=np.float32),
        )
        positive_ir, _ = synthesize_ambisonic_ir(
            [positive], 48000, 0.05, AmbisonicEncoder()
        )
        negative_ir, _ = synthesize_ambisonic_ir(
            [negative], 48000, 0.05, AmbisonicEncoder()
        )
        np.testing.assert_allclose(negative_ir, -positive_ir, atol=1e-7)

    def test_coherent_pressure_transfer_is_authoritative(self):
        event = AcousticEvent(
            delay_seconds=0.01,
            arrival_direction=Vector((1.0, 0.0, 0.0)),
            energy_bands=np.full(NUM_BANDS, 0.81, dtype=np.float32),
            kind='EARLY',
            coherent_pressure_bands=np.full(
                NUM_BANDS, -0.2, dtype=np.float32
            ),
        )
        ir, stats = synthesize_ambisonic_ir(
            [event], 48000, 0.05, AmbisonicEncoder()
        )

        self.assertEqual(stats.coherent_events, 1)
        self.assertAlmostEqual(float(ir[0, 480]), -0.2, places=5)

    def test_diffuse_phase_is_repeatable_for_a_seed(self):
        events = [
            AcousticEvent(
                delay_seconds=0.01 + index * 0.0007,
                arrival_direction=Vector((1.0, index * 0.1, 0.2)).normalized(),
                energy_bands=np.full(NUM_BANDS, 0.01, dtype=np.float32),
                kind='DIFFUSE',
                order=2,
            )
            for index in range(12)
        ]
        first, _ = synthesize_ambisonic_ir(
            events, 48000, 0.1, AmbisonicEncoder(), seed=4
        )
        second, _ = synthesize_ambisonic_ir(
            events, 48000, 0.1, AmbisonicEncoder(), seed=4
        )
        different, _ = synthesize_ambisonic_ir(
            events, 48000, 0.1, AmbisonicEncoder(), seed=5
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertGreater(float(np.max(np.abs(first))), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
