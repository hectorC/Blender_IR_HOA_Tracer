# -*- coding: utf-8 -*-
"""
Acoustic modeling and material properties for ray tracing.
"""
import numpy as np
from math import pi, sqrt, exp
from functools import lru_cache
from typing import Tuple, Union, Optional, Any


# Frequency band definitions
BAND_CENTERS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
NUM_BANDS = len(BAND_CENTERS_HZ)

# Default material properties
DEFAULT_ABSORPTION_SPECTRUM = tuple(0.2 for _ in BAND_CENTERS_HZ)
DEFAULT_SCATTER_SPECTRUM = (0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.18)

# Material presets. Absorption is an energy coefficient for the complete named
# surface construction. Scattering represents unresolved surface detail only;
# relief already present in evaluated Blender geometry must not be counted a
# second time. The values are practical room-acoustic starting points rather
# than specifications for an individual commercial product.
MATERIAL_PRESET_DATA = [
    (
        'WOOD',
        'Wood Panel on Framing',
        'Cavity-backed wood panel; seams and fine relief are not modeled',
        (0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06),
        (0.02, 0.03, 0.04, 0.06, 0.10, 0.16, 0.24),
    ),
    (
        'WOOD_SOLID',
        'Wood - Solid/Hard-Backed',
        'Parquet or solid wood fixed to a hard backing',
        (0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.08),
        (0.01, 0.02, 0.03, 0.04, 0.06, 0.10, 0.16),
    ),
    (
        'CONCRETE',
        'Concrete - Rough/Unsealed',
        'Ordinary unfinished concrete with shallow unresolved texture',
        (0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07),
        (0.03, 0.04, 0.06, 0.09, 0.14, 0.22, 0.32),
    ),
    (
        'CONCRETE_SMOOTH',
        'Concrete - Smooth/Painted',
        'Sealed, painted, or glazed concrete',
        (0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02),
        (0.005, 0.007, 0.01, 0.015, 0.025, 0.04, 0.07),
    ),
    (
        'PLASTER',
        'Plaster on Lath',
        'Smooth plaster on a lightweight lath or framed construction',
        (0.14, 0.10, 0.06, 0.04, 0.04, 0.03, 0.02),
        (0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12),
    ),
    (
        'PLASTER_SOLID',
        'Plaster - Smooth on Masonry',
        'Lime or cement plaster applied to a solid backing',
        (0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05),
        (0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12),
    ),
    (
        'CARPET',
        'Carpet - Heavy with Underlay',
        'Heavy carpet over felt or open-cell foam underlay',
        (0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.75),
        (0.02, 0.03, 0.04, 0.06, 0.10, 0.16, 0.24),
    ),
    (
        'CARPET_HARD',
        'Carpet - Bonded to Concrete',
        'Heavy carpet directly bonded to a hard floor',
        (0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.70),
        (0.02, 0.03, 0.04, 0.06, 0.10, 0.16, 0.24),
    ),
    (
        'TILE',
        'Glazed Tile / Polished Stone',
        'Smooth glazed ceramic, marble, or polished dense stone',
        (0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02),
        (0.005, 0.008, 0.01, 0.015, 0.025, 0.04, 0.07),
    ),
    (
        'BRICK',
        'Brick - Flush Joints',
        'Unglazed brickwork with filled or flush mortar joints',
        (0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.09),
        (0.03, 0.04, 0.06, 0.09, 0.15, 0.24, 0.35),
    ),
    (
        'BRICK_OPEN',
        'Brick - Open/Deep Joints',
        'Brickwork with recessed or open joints absent from the mesh',
        (0.08, 0.09, 0.12, 0.16, 0.22, 0.24, 0.26),
        (0.06, 0.08, 0.11, 0.15, 0.24, 0.36, 0.50),
    ),
    (
        'GLASS',
        'Glass - Large Pane',
        'Large smooth glazing; low-frequency loss includes panel motion',
        (0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02),
        (0.005, 0.007, 0.01, 0.015, 0.025, 0.04, 0.07),
    ),
    (
        'METAL',
        'Metal - Smooth Solid',
        'Massive smooth metal without perforation or a resonant cavity',
        (0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02),
        (0.005, 0.007, 0.01, 0.015, 0.025, 0.04, 0.07),
    ),
    (
        'ROCK_DENSE',
        'Rock - Dense/Smooth',
        'Dense solid rock with only shallow unmodeled texture',
        (0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05),
        (0.01, 0.02, 0.03, 0.05, 0.08, 0.14, 0.22),
    ),
    (
        'ROCK_CAVE',
        'Rock - Rough Cave',
        'Irregular cave wall whose smaller fractures are absent from the mesh',
        (0.03, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10),
        (0.08, 0.12, 0.20, 0.32, 0.48, 0.65, 0.78),
    ),
    (
        'ROCK_POROUS',
        'Rock - Porous/Weathered',
        'Fractured or porous exposed rock with lossy small-scale structure',
        (0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.28),
        (0.10, 0.16, 0.25, 0.38, 0.52, 0.68, 0.80),
    ),
    (
        'GRAVEL',
        'Gravel / Crushed Rock Bed',
        'Loose crushed stone bed approximately 150 mm deep',
        (0.19, 0.23, 0.43, 0.37, 0.58, 0.62, 0.66),
        (0.12, 0.20, 0.32, 0.48, 0.62, 0.72, 0.80),
    ),
    (
        'SAND',
        'Sand / Loose Soil',
        'Loose granular ground approximately 100 mm deep',
        (0.15, 0.35, 0.40, 0.50, 0.55, 0.80, 1.00),
        (0.05, 0.08, 0.14, 0.22, 0.34, 0.48, 0.62),
    ),
    (
        'CURTAIN_HEAVY',
        'Curtain - Heavy Folded',
        'Heavy velour with folds represented by material scattering',
        (0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.60),
        (0.08, 0.12, 0.18, 0.28, 0.40, 0.50, 0.58),
    ),
    (
        'MINERAL_WOOL',
        'Mineral Wool - 50 mm',
        'Exposed 50 mm porous absorber mounted against a wall',
        (0.15, 0.70, 0.60, 0.60, 0.85, 0.90, 0.95),
        (0.02, 0.03, 0.04, 0.06, 0.10, 0.15, 0.20),
    ),
    (
        'WATER',
        'Water - Calm Surface',
        'Calm open water with no modeled waves',
        (0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02),
        (0.005, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05),
    ),
    (
        'AUDIENCE',
        'Audience - Upholstered Seating',
        'Area approximation for people and medium upholstered seating',
        (0.56, 0.64, 0.70, 0.72, 0.68, 0.62, 0.56),
        (0.45, 0.55, 0.62, 0.67, 0.70, 0.72, 0.74),
    ),
]


def _avg(values) -> float:
    """Calculate average of values."""
    return float(sum(values)) / max(len(values), 1)


def _band_label(freq_hz: float) -> str:
    """Generate human-readable label for frequency band."""
    if freq_hz >= 1000.0:
        return f"{int(freq_hz / 1000.0)} kHz"
    return f"{int(freq_hz)} Hz"


BAND_LABELS = tuple(_band_label(f) for f in BAND_CENTERS_HZ)

# Process material presets
MATERIAL_PRESETS = {
    identifier: {
        'absorption_spectrum': tuple(float(max(0.0, min(1.0, v))) for v in absorption),
        'scatter_spectrum': tuple(float(max(0.0, min(1.0, v))) for v in scatter),
        'absorption': _avg(absorption),
        'scatter': _avg(scatter)
    }
    for identifier, _, _, absorption, scatter in MATERIAL_PRESET_DATA
}


def get_object_spectrum(obj: Any, vec_attr: str, scalar_attr: str, 
                       default_vec: Tuple[float, ...]) -> np.ndarray:
    """Extract a frequency spectrum from an acoustic assignment owner."""
    if obj is None:
        return np.array(default_vec, dtype=np.float32)
    
    if hasattr(obj, vec_attr):
        values = getattr(obj, vec_attr)
        if values is not None and len(values) == NUM_BANDS:
            return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)
    
    scalar = float(getattr(obj, scalar_attr, default_vec[0])) if obj else default_vec[0]
    return np.clip(np.full(NUM_BANDS, scalar, dtype=np.float32), 0.0, 1.0)


def get_absorption_spectrum(obj: Any) -> np.ndarray:
    """Get absorption spectrum for an object."""
    return get_object_spectrum(obj, 'absorption_bands', 'absorption', DEFAULT_ABSORPTION_SPECTRUM)


def get_scatter_spectrum(obj: Any) -> np.ndarray:
    """Get scatter spectrum for an object."""
    return get_object_spectrum(obj, 'scatter_bands', 'scatter', DEFAULT_SCATTER_SPECTRUM)


def get_transmission_coeff(obj: Any) -> float:
    """Get transmission coefficient for an object."""
    return float(np.clip(getattr(obj, 'transmission', 0.0) if obj else 0.0, 0.0, 1.0))


def get_transmission_spectrum(obj: Any) -> np.ndarray:
    """Get per-band transmitted-energy fractions for an object."""
    scalar = get_transmission_coeff(obj)
    if obj is not None and hasattr(obj, 'transmission_bands'):
        values = getattr(obj, 'transmission_bands')
        if values is not None and len(values) == NUM_BANDS:
            return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)
    return np.full(NUM_BANDS, scalar, dtype=np.float32)


def iso9613_alpha_dbpm(f_hz: float, T_c: float, rh_pct: float, p_kpa: float) -> float:
    """Calculate ISO 9613-1 atmospheric absorption coefficient.
    
    Args:
        f_hz: Frequency in Hz
        T_c: Temperature in degrees C
        rh_pct: Relative humidity in %
        p_kpa: Pressure in kPa
        
    Returns:
        Absorption coefficient in dB/m
    """
    temperature_k = max(1.0, 273.15 + float(T_c))
    reference_temperature_k = 293.15
    triple_point_temperature_k = 273.16
    pressure_kpa = max(1e-3, float(p_kpa))
    reference_pressure_kpa = 101.325
    pressure_ratio = pressure_kpa / reference_pressure_kpa
    temperature_ratio = temperature_k / reference_temperature_k
    relative_humidity_pct = np.clip(float(rh_pct), 0.0, 100.0)

    # ISO 9613-1 Annex B converts relative humidity to the molar
    # concentration of water vapour. saturation_pressure_ratio is p_sat/p_ref.
    saturation_pressure_ratio = 10.0 ** (
        -6.8346 * (triple_point_temperature_k / temperature_k) ** 1.261
        + 4.6151
    )
    water_vapour_concentration = (
        relative_humidity_pct * saturation_pressure_ratio / pressure_ratio
    )

    oxygen_relaxation_hz = pressure_ratio * (
        24.0
        + 4.04e4
        * water_vapour_concentration
        * (0.02 + water_vapour_concentration)
        / (0.391 + water_vapour_concentration)
    )
    nitrogen_relaxation_hz = (
        pressure_ratio
        * temperature_ratio ** -0.5
        * (
            9.0
            + 280.0
            * water_vapour_concentration
            * exp(-4.17 * (temperature_ratio ** (-1.0 / 3.0) - 1.0))
        )
    )

    frequency_hz = max(0.0, float(f_hz))
    frequency_squared = frequency_hz * frequency_hz
    classical_term = (
        1.84e-11 * pressure_ratio ** -1.0 * sqrt(temperature_ratio)
    )
    vibrational_term = temperature_ratio ** -2.5 * (
        0.01275
        * exp(-2239.1 / temperature_k)
        * oxygen_relaxation_hz
        / (frequency_squared + oxygen_relaxation_hz ** 2)
        + 0.1068
        * exp(-3352.0 / temperature_k)
        * nitrogen_relaxation_hz
        / (frequency_squared + nitrogen_relaxation_hz ** 2)
    )

    alpha = 8.686 * frequency_squared * (classical_term + vibrational_term)
    return float(max(0.0, alpha))


def air_attenuation_bands(distance_m: float, temp_c: float = 20.0, 
                         rh_pct: float = 50.0, pressure_kpa: float = 101.325) -> np.ndarray:
    """Calculate frequency-dependent air absorption."""
    if distance_m <= 0.0:
        return np.ones(NUM_BANDS, dtype=np.float32)
    
    gains = []
    for f in BAND_CENTERS_HZ:
        alpha_dbpm = iso9613_alpha_dbpm(f, temp_c, rh_pct, pressure_kpa)
        gains.append(10.0 ** (-(alpha_dbpm * distance_m) / 20.0))
    
    return np.clip(np.array(gains, dtype=np.float32), 1e-4, 1.0)


@lru_cache(maxsize=4096)
def _band_kernel_cache(band_key: Tuple[float, ...], sr: int, kernel_len: int) -> np.ndarray:
    """Generate a cached causal minimum-phase octave-band reconstruction FIR."""
    band_profile = np.maximum(np.array(band_key, dtype=np.float64), 0.0)
    kernel_len = max(16, int(kernel_len))
    sr = max(1000, int(sr))
    if not np.any(band_profile > 1e-12):
        return np.zeros(kernel_len, dtype=np.float32)

    n_fft = max(4096, 1 << int(np.ceil(np.log2(kernel_len * 8))))
    frequency_axis = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    magnitudes = np.empty_like(frequency_axis)
    magnitudes[0] = band_profile[0]
    log_bands = np.log10(np.array(BAND_CENTERS_HZ, dtype=np.float64))
    magnitudes[1:] = np.interp(
        np.log10(np.maximum(frequency_axis[1:], 1.0)),
        log_bands,
        band_profile,
        left=band_profile[0],
        right=band_profile[-1],
    )

    # Real-cepstrum construction produces a causal minimum-phase filter with
    # the requested magnitude response and no linear-phase pre-ringing.
    log_magnitude = np.log(np.maximum(magnitudes, 1e-7))
    cepstrum = np.fft.irfft(log_magnitude, n=n_fft)
    minimum_phase_cepstrum = np.zeros(n_fft, dtype=np.float64)
    minimum_phase_cepstrum[0] = cepstrum[0]
    minimum_phase_cepstrum[1:n_fft // 2] = 2.0 * cepstrum[1:n_fft // 2]
    minimum_phase_cepstrum[n_fft // 2] = cepstrum[n_fft // 2]
    minimum_phase_spectrum = np.exp(
        np.fft.rfft(minimum_phase_cepstrum, n=n_fft)
    )
    kernel = np.fft.irfft(minimum_phase_spectrum, n=n_fft)[:kernel_len]

    # Fade only the final quarter to suppress truncation ripple while keeping
    # the early minimum-phase energy and acoustic arrival time intact.
    fade_start = (kernel_len * 3) // 4
    fade_size = kernel_len - fade_start
    if fade_size > 1:
        kernel[fade_start:] *= 0.5 * (
            1.0 + np.cos(np.linspace(0.0, pi, fade_size))
        )

    dc_sum = float(np.sum(kernel))
    if abs(dc_sum) > 1e-12:
        kernel *= float(magnitudes[0]) / dc_sum

    return kernel.astype(np.float32)


def _default_kernel_length(sr: int) -> int:
    """Use roughly 10.7 ms of FIR at common production sample rates."""
    return max(128, min(2048, int(round(max(1000, int(sr)) * 0.0106667))))


@lru_cache(maxsize=16)
def design_complementary_filter_bank(
    sr: int, kernel_len: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Return a common-delay, complementary seven-band FIR bank.

    Adjacent bands crossfade linearly on a log-frequency axis. Because every
    frequency-bin weight sums to one and each band receives the same window,
    summing all seven impulse responses reconstructs a delayed unit impulse.
    """
    sr = max(1000, int(sr))
    if kernel_len is None:
        kernel_len = _default_kernel_length(sr) | 1
    kernel_len = max(33, int(kernel_len) | 1)
    half = kernel_len // 2
    n_fft = max(4096, 1 << int(np.ceil(np.log2(kernel_len * 8))))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    weights = np.zeros((NUM_BANDS, frequencies.size), dtype=np.float64)
    centers = np.array(BAND_CENTERS_HZ, dtype=np.float64)

    weights[0, frequencies <= centers[0]] = 1.0
    weights[-1, frequencies >= centers[-1]] = 1.0
    for band in range(NUM_BANDS - 1):
        mask = (frequencies > centers[band]) & (frequencies < centers[band + 1])
        if not np.any(mask):
            continue
        fraction = (
            np.log2(frequencies[mask] / centers[band])
            / np.log2(centers[band + 1] / centers[band])
        )
        weights[band, mask] = 1.0 - fraction
        weights[band + 1, mask] = fraction
    for band, center in enumerate(centers):
        nearest = int(np.argmin(np.abs(frequencies - center)))
        weights[band, nearest] = 1.0
        if band > 0:
            weights[band - 1, nearest] = 0.0

    window = np.hanning(kernel_len)
    kernels = np.empty((NUM_BANDS, kernel_len), dtype=np.float64)
    for band in range(NUM_BANDS):
        zero_phase = np.fft.irfft(weights[band], n=n_fft)
        centered = np.concatenate((zero_phase[-half:], zero_phase[:half + 1]))
        kernels[band] = centered * window
    return kernels.astype(np.float32), half


def design_band_kernel(
    band_profile: np.ndarray, sr: int, kernel_len: Optional[int] = None
) -> np.ndarray:
    """Design a causal minimum-phase frequency reconstruction filter."""
    if kernel_len is None:
        kernel_len = _default_kernel_length(sr)
    key = tuple(float(round(float(v), 5)) for v in band_profile)
    return _band_kernel_cache(key, int(sr), int(kernel_len))


def add_filtered_impulse(ir: np.ndarray, ambi_vec: np.ndarray, delay_samples: float, 
                        amplitude: float, band_profile: np.ndarray, sr: int) -> bool:
    """Add a frequency-filtered impulse to the impulse response."""
    kernel = design_band_kernel(band_profile, sr)
    base = int(np.floor(delay_samples))
    frac = float(delay_samples - base)
    
    weights = ((base, 1.0 - frac), (base + 1, frac))
    wrote = False
    
    for start, weight in weights:
        if weight <= 0.0:
            continue
        source_start = max(0, -start)
        destination_start = max(0, start)
        count = min(
            kernel.shape[0] - source_start,
            ir.shape[1] - destination_start,
        )
        if count <= 0:
            continue
        destination = slice(destination_start, destination_start + count)
        source = slice(source_start, source_start + count)
        ir[:, destination] += (
            ambi_vec[:, None]
            * (amplitude * weight)
            * kernel[None, source]
        )
        wrote = True
    
    return wrote


class MaterialProperties:
    """Container for acoustic material properties."""
    
    def __init__(self, obj: Any = None):
        """Initialize from a Blender material, object fallback, or defaults."""
        if obj is None:
            self.absorption_spectrum = np.array(DEFAULT_ABSORPTION_SPECTRUM, dtype=np.float32)
            self.scatter_spectrum = np.array(DEFAULT_SCATTER_SPECTRUM, dtype=np.float32)
        else:
            self.absorption_spectrum = get_absorption_spectrum(obj)
            self.scatter_spectrum = get_scatter_spectrum(obj)
        
        # Calculate derived properties
        self.transmission_spectrum = get_transmission_spectrum(obj)
        self.reflection_spectrum = np.clip(
            1.0 - self.absorption_spectrum - self.transmission_spectrum, 0.0, 1.0
        )
        
        # Scattering fractions
        self.specular_fraction = np.clip(1.0 - self.scatter_spectrum, 0.0, 1.0)
        self.diffuse_fraction = np.clip(self.scatter_spectrum, 0.0, 1.0)
