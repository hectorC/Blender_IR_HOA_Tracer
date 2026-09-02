"""Convert directional acoustic-energy events into a third-order HOA IR."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Iterable, List

import mathutils
import numpy as np

from .acoustics import (
    NUM_BANDS,
    add_filtered_impulse,
    design_complementary_filter_bank,
)
from .ambisonic import AmbisonicEncoder


@dataclass
class AcousticEvent:
    """A time-, direction-, and frequency-resolved energy contribution."""

    delay_seconds: float
    arrival_direction: mathutils.Vector
    energy_bands: np.ndarray
    kind: str
    order: int = 0
    pressure_sign_bands: np.ndarray | None = None


@dataclass
class SynthesisStats:
    direct_events: int = 0
    early_events: int = 0
    diffuse_events: int = 0


def _gain_for_event(kind: str, early_gain: float, diffuse_gain: float) -> float:
    if kind == 'DIFFUSE':
        return diffuse_gain
    if kind == 'EARLY':
        return early_gain
    return 1.0


def _add_signed_filtered_impulse(
    result: np.ndarray,
    ambisonic: np.ndarray,
    delay_samples: float,
    pressure_bands: np.ndarray,
    sample_rate: int,
) -> None:
    """Reconstruct a causal spectrum that may change pressure polarity."""
    positive = np.maximum(pressure_bands, 0.0)
    negative = np.maximum(-pressure_bands, 0.0)
    if np.any(positive > 1e-12):
        add_filtered_impulse(
            result,
            ambisonic,
            delay_samples,
            1.0,
            positive,
            sample_rate,
        )
    if np.any(negative > 1e-12):
        add_filtered_impulse(
            result,
            ambisonic,
            delay_samples,
            -1.0,
            negative,
            sample_rate,
        )


def synthesize_ambisonic_ir(
    events: Iterable[AcousticEvent],
    sample_rate: int,
    duration_seconds: float,
    encoder: AmbisonicEncoder,
    seed: int = 1,
    early_gain_db: float = 0.0,
    diffuse_gain_db: float = 0.0,
) -> tuple[np.ndarray, SynthesisStats]:
    """Synthesize pressure from deterministic and stochastic energy events.

    Direct and deterministic early events retain a coherent, causal pressure
    impulse. Monte Carlo events receive repeatable random polarity before a
    complementary filter bank converts their seven energy bands to pressure.
    This prevents unrelated diffuse paths from adding as if phase-coherent.
    """
    sample_rate = max(1000, int(sample_rate))
    sample_count = max(1, int(round(float(duration_seconds) * sample_rate)))
    result = np.zeros((16, sample_count), dtype=np.float32)
    diffuse_trains = np.zeros((NUM_BANDS, 16, sample_count), dtype=np.float32)
    early_gain = 10.0 ** (float(early_gain_db) / 20.0)
    diffuse_gain = 10.0 ** (float(diffuse_gain_db) / 20.0)
    rng = np.random.default_rng(int(seed) if int(seed) != 0 else None)
    stats = SynthesisStats()

    for event in events:
        delay_samples = float(event.delay_seconds) * sample_rate
        if delay_samples < 0.0 or delay_samples >= sample_count:
            continue
        energy = np.maximum(np.asarray(event.energy_bands, dtype=np.float64), 0.0)
        if energy.shape != (NUM_BANDS,) or not np.any(energy > 1e-20):
            continue
        pressure = np.sqrt(energy).astype(np.float32)
        if event.pressure_sign_bands is not None:
            signs = np.asarray(
                event.pressure_sign_bands, dtype=np.float32
            )
            if signs.shape == (NUM_BANDS,):
                pressure *= np.where(signs < 0.0, -1.0, 1.0)
        pressure *= _gain_for_event(event.kind, early_gain, diffuse_gain)
        ambisonic = encoder.encode(event.arrival_direction)

        if event.kind != 'DIFFUSE':
            _add_signed_filtered_impulse(
                result,
                ambisonic,
                delay_samples,
                pressure,
                sample_rate,
            )
            if event.kind == 'DIRECT':
                stats.direct_events += 1
            else:
                stats.early_events += 1
            continue

        stats.diffuse_events += 1
        signs = rng.choice((-1.0, 1.0), size=NUM_BANDS).astype(np.float32)
        band_pressure = pressure * signs
        base = int(np.floor(delay_samples))
        fraction = float(delay_samples - base)
        values = band_pressure[:, None] * ambisonic[None, :]
        if 0 <= base < sample_count:
            diffuse_trains[:, :, base] += values * (1.0 - fraction)
        if fraction > 0.0 and 0 <= base + 1 < sample_count:
            diffuse_trains[:, :, base + 1] += values * fraction

    if stats.diffuse_events:
        kernels, group_delay = design_complementary_filter_bank(sample_rate)
        convolution_size = sample_count + kernels.shape[1] - 1
        fft_size = 1 << int(ceil(log2(max(2, convolution_size))))
        for band in range(NUM_BANDS):
            kernel_spectrum = np.fft.rfft(kernels[band], n=fft_size)
            train_spectrum = np.fft.rfft(diffuse_trains[band], n=fft_size, axis=1)
            filtered = np.fft.irfft(
                train_spectrum * kernel_spectrum[None, :],
                n=fft_size,
                axis=1,
            )
            result += filtered[:, group_delay:group_delay + sample_count].astype(
                np.float32
            )

    return result, stats
