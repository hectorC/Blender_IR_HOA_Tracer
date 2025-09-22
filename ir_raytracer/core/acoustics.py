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
DEFAULT_SCATTER_SPECTRUM = tuple(0.35 for _ in BAND_CENTERS_HZ)

# Material presets
MATERIAL_PRESET_DATA = [
    (
        'WOOD',
        'Wood (panel)',
        (0.18, 0.17, 0.16, 0.14, 0.12, 0.10, 0.10),
        (0.30, 0.32, 0.34, 0.36, 0.38, 0.38, 0.38)
    ),
    (
        'CONCRETE',
        'Concrete',
        (0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06),
        (0.10, 0.12, 0.14, 0.16, 0.18, 0.18, 0.18)
    ),
    (
        'CARPET',
        'Carpet',
        (0.08, 0.12, 0.30, 0.55, 0.65, 0.70, 0.70),
        (0.55, 0.57, 0.60, 0.62, 0.62, 0.62, 0.62)
    ),
    (
        'TILE',
        'Tile',
        (0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05),
        (0.15, 0.17, 0.18, 0.20, 0.22, 0.22, 0.22)
    ),
    (
        'BRICK',
        'Brick',
        (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09),
        (0.35, 0.37, 0.40, 0.43, 0.45, 0.45, 0.45)
    )
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
    for identifier, _, absorption, scatter in MATERIAL_PRESET_DATA
}


def get_object_spectrum(obj: Any, vec_attr: str, scalar_attr: str, 
                       default_vec: Tuple[float, ...]) -> np.ndarray:
    """Extract frequency spectrum from object properties."""
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
    T = 273.15 + float(T_c)
    T0 = 293.15
    fr_h = max(0.0, min(100.0, float(rh_pct))) / 100.0
    P = max(1e-3, float(p_kpa))
    P0 = 101.325
    
    # Relaxation frequencies (approximate forms used in practice)
    frO = 24.0 + 4.04e4 * fr_h * (0.02 + fr_h) / (0.391 + fr_h)
    frN = (T / T0)**(-0.5) * (9.0 + 280.0 * fr_h * exp(-4.17 * ((T / T0)**(-1.0/3.0) - 1.0)))
    
    fk = (float(f_hz) / 1000.0)
    fk2 = fk * fk
    
    # Classical + rotational/translational + vibrational contributions
    term_class = 1.84e-11 * (P0 / P) * sqrt(T / T0)
    term_O = (T / T0)**(-2.5) * (0.01275 * exp(-2239.1 / T)) * (frO / (frO*frO + fk2))
    term_N = (T / T0)**(-2.5) * (0.1068  * exp(-3352.0 / T)) * (frN / (frN*frN + fk2))
    
    alpha = 8.686 * fk2 * (term_class + term_O + term_N)  # dB/m
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
    """Cached frequency band kernel generation."""
    band_profile = np.array(band_key, dtype=np.float32)
    kernel_len = int(kernel_len)
    sr = max(1000, int(sr))
    
    n_fft = max(64, 1 << int(np.ceil(np.log2(kernel_len * 4))))
    freq_axis = np.linspace(0.0, sr * 0.5, n_fft // 2 + 1, dtype=np.float32)
    
    mags = np.empty_like(freq_axis)
    mags[0] = float(band_profile[0])
    
    if freq_axis.shape[0] > 1:
        log_freq = np.log10(np.maximum(freq_axis[1:], 1.0))
        log_bands = np.log10(np.array(BAND_CENTERS_HZ, dtype=np.float32))
        interp = np.interp(log_freq, log_bands, band_profile, 
                          left=band_profile[0], right=band_profile[-1])
        mags[1:] = interp
    
    mag_full = np.concatenate([mags, mags[-2:0:-1]])
    ir = np.fft.irfft(mag_full, n=n_fft).real[:kernel_len]
    
    if kernel_len >= 4:
        window = np.hanning(kernel_len * 2)[kernel_len:kernel_len * 2]
        ir *= window
    
    s = float(np.sum(ir))
    target_dc = mags[0]
    if abs(s) > 1e-12:
        ir *= target_dc / s
    
    return ir.astype(np.float32)


def design_band_kernel(band_profile: np.ndarray, sr: int, kernel_len: int = 16) -> np.ndarray:
    """Design frequency-dependent filter kernel."""
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
    
    for start, w in weights:
        if w <= 0.0:
            continue
        for k, kv in enumerate(kernel):
            idx = start + k
            if 0 <= idx < ir.shape[1]:
                ir[:, idx] += ambi_vec * (amplitude * w * kv)
                wrote = True
    
    return wrote


class MaterialProperties:
    """Container for acoustic material properties."""
    
    def __init__(self, obj: Any = None):
        """Initialize from Blender object or defaults."""
        if obj is None:
            self.absorption_spectrum = np.array(DEFAULT_ABSORPTION_SPECTRUM, dtype=np.float32)
            self.scatter_spectrum = np.array(DEFAULT_SCATTER_SPECTRUM, dtype=np.float32)
            self.transmission = 0.0
        else:
            self.absorption_spectrum = get_absorption_spectrum(obj)
            self.scatter_spectrum = get_scatter_spectrum(obj)
            self.transmission = get_transmission_coeff(obj)
        
        # Calculate derived properties
        self.transmission_spectrum = np.clip(
            np.full(NUM_BANDS, self.transmission, dtype=np.float32), 0.0, 1.0
        )
        self.reflection_spectrum = np.clip(
            1.0 - self.absorption_spectrum - self.transmission_spectrum, 0.0, 1.0
        )
        
        # Scattering fractions
        self.specular_fraction = np.clip(1.0 - self.scatter_spectrum, 0.0, 1.0)
        self.diffuse_fraction = np.clip(self.scatter_spectrum, 0.0, 1.0)
        
        # Amplitude coefficients (square root for energy conservation)
        self.reflection_amplitude = np.sqrt(np.maximum(self.reflection_spectrum, 1e-6))
        self.transmission_amplitude = np.sqrt(self.transmission_spectrum)
        self.specular_amplitude = self.reflection_amplitude * np.sqrt(np.maximum(self.specular_fraction, 1e-6))
        self.diffuse_amplitude = self.reflection_amplitude * np.sqrt(np.maximum(self.diffuse_fraction, 1e-6))