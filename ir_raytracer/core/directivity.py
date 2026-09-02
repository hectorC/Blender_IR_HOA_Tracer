"""Frequency-dependent, source-local radiation patterns."""
from __future__ import annotations

from functools import lru_cache
from math import cos, log, pi, sin, sqrt
from typing import Any, Sequence

import mathutils
import numpy as np

from .acoustics import NUM_BANDS
from .ambisonic import (
    apply_orientation_transform,
    encode_ambisonics_3rd_order,
)


DIRECTIVITY_PATTERNS = {
    'OMNI',
    'CARDIOID',
    'DIPOLE',
    'FORWARD_CONE',
    'LOUDSPEAKER',
    'CUSTOM_SH',
}

DIRECTIVITY_STRENGTH_PRESETS = {
    'OMNI': (0.0,) * NUM_BANDS,
    'CARDIOID': (1.0,) * NUM_BANDS,
    'DIPOLE': (1.0,) * NUM_BANDS,
    'FORWARD_CONE': (1.0,) * NUM_BANDS,
    # A compact loudspeaker is nearly omnidirectional in the bass and becomes
    # progressively more forward-radiating as wavelength decreases.
    'LOUDSPEAKER': (0.05, 0.12, 0.25, 0.45, 0.65, 0.82, 0.92),
    'CUSTOM_SH': (1.0,) * NUM_BANDS,
}

DEFAULT_CUSTOM_SH = (
    0.5,  # W
    0.0,  # Y
    0.0,  # Z
    0.5,  # X: source-local -Y/front after Blender-to-AmbiX mapping
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


def _safe_vector(values: Sequence[float], size: int, fallback) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        return np.asarray(fallback, dtype=np.float64)
    return result


@lru_cache(maxsize=256)
def _custom_pattern_peak(coefficients: tuple[float, ...]) -> float:
    """Estimate the peak absolute SH pressure response on a dense sphere."""
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    peak = 0.0
    directions = [
        mathutils.Vector(direction)
        for direction in (
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
        )
    ]
    sample_count = 2048
    golden_angle = pi * (3.0 - sqrt(5.0))
    for index in range(sample_count):
        z = 1.0 - 2.0 * (index + 0.5) / sample_count
        radius = sqrt(max(0.0, 1.0 - z * z))
        angle = golden_angle * index
        directions.append(mathutils.Vector((
            radius * cos(angle), radius * sin(angle), z
        )))
    for local_direction in directions:
        ambix_direction = apply_orientation_transform(local_direction)
        response = float(np.dot(
            coefficient_array,
            encode_ambisonics_3rd_order(ambix_direction),
        ))
        peak = max(peak, abs(response))
    return max(peak, 1e-12)


class SourceDirectivity:
    """Immutable snapshot of one source's pressure radiation pattern."""

    def __init__(
        self,
        pattern: str = 'OMNI',
        source_rotation: mathutils.Quaternion | None = None,
        strength_bands: Sequence[float] | None = None,
        cone_width_deg: float = 90.0,
        custom_sh: Sequence[float] = DEFAULT_CUSTOM_SH,
    ):
        self.pattern = (
            pattern if pattern in DIRECTIVITY_PATTERNS else 'OMNI'
        )
        rotation = (
            source_rotation.copy()
            if source_rotation is not None
            else mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        )
        rotation.normalize()
        self.source_rotation = rotation
        self._world_to_source = rotation.inverted()
        preset = DIRECTIVITY_STRENGTH_PRESETS[self.pattern]
        strengths = preset if strength_bands is None else strength_bands
        self.strength_bands = np.clip(
            _safe_vector(strengths, NUM_BANDS, preset), 0.0, 1.0
        ).astype(np.float32)
        self.cone_width_deg = float(np.clip(cone_width_deg, 10.0, 170.0))
        self.custom_sh = _safe_vector(
            custom_sh, 16, DEFAULT_CUSTOM_SH
        ).astype(np.float32)
        if self.pattern == 'CUSTOM_SH':
            coefficient_key = tuple(
                round(float(value), 7) for value in self.custom_sh
            )
            self._custom_peak = _custom_pattern_peak(coefficient_key)
        else:
            self._custom_peak = 1.0

    def _pattern_pressure(self, local_direction: mathutils.Vector) -> float:
        """Return peak-normalized signed pressure for a local direction."""
        forward_cosine = float(np.clip(-local_direction.y, -1.0, 1.0))
        if self.pattern == 'OMNI':
            return 1.0
        if self.pattern == 'CARDIOID':
            return 0.5 * (1.0 + forward_cosine)
        if self.pattern == 'DIPOLE':
            return forward_cosine
        if self.pattern == 'LOUDSPEAKER':
            return max(0.0, forward_cosine)
        if self.pattern == 'FORWARD_CONE':
            cosine = max(0.0, forward_cosine)
            half_angle = self.cone_width_deg * pi / 360.0
            half_power_cosine = max(cos(half_angle), 1e-4)
            exponent = max(0.05, log(0.5) / log(half_power_cosine))
            return cosine ** exponent

        ambix_direction = apply_orientation_transform(local_direction)
        pressure = float(np.dot(
            self.custom_sh,
            encode_ambisonics_3rd_order(ambix_direction),
        ))
        return float(np.clip(
            pressure / self._custom_peak, -1.0, 1.0
        ))

    def pressure_gain(self, emitted_world_direction) -> np.ndarray:
        """Return signed per-band pressure gain for an emitted direction."""
        direction = mathutils.Vector(emitted_world_direction)
        if direction.length_squared <= 1e-20:
            return np.ones(NUM_BANDS, dtype=np.float32)
        if self.pattern == 'OMNI':
            return np.ones(NUM_BANDS, dtype=np.float32)
        local_direction = self._world_to_source @ direction.normalized()
        patterned = self._pattern_pressure(local_direction)
        pressure = (
            1.0 - self.strength_bands
            + self.strength_bands * patterned
        )
        return np.clip(pressure, -1.0, 1.0).astype(np.float32)

    def energy_gain_and_polarity(
        self, emitted_world_direction
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return energy gain plus pressure polarity for one emitted ray."""
        pressure = self.pressure_gain(emitted_world_direction)
        return (
            np.square(pressure, dtype=np.float32),
            np.where(pressure < 0.0, -1.0, 1.0).astype(np.float32),
        )

    def metadata(self) -> dict:
        result = {
            "pattern": self.pattern,
            "reference": "source_local",
            "forward_axis": "-Y",
            "normalization": "unity_peak_pressure",
            "strength_bands": [
                float(value) for value in self.strength_bands
            ],
            "source_world_quaternion_wxyz": [
                float(value) for value in self.source_rotation
            ],
        }
        if self.pattern == 'FORWARD_CONE':
            result["minus_6_db_beam_width_degrees"] = self.cone_width_deg
        if self.pattern == 'CUSTOM_SH':
            result["custom_acn_sn3d_pressure_coefficients"] = [
                float(value) for value in self.custom_sh
            ]
        return result


def source_directivity_from_object(
    obj: Any,
    source_rotation: mathutils.Quaternion | None = None,
) -> SourceDirectivity:
    """Create a detached directivity snapshot from a Blender source object."""
    if obj is None:
        return SourceDirectivity(source_rotation=source_rotation)
    pattern = getattr(obj, 'airt_source_directivity', 'OMNI')
    preset = DIRECTIVITY_STRENGTH_PRESETS.get(
        pattern, DIRECTIVITY_STRENGTH_PRESETS['OMNI']
    )
    return SourceDirectivity(
        pattern=pattern,
        source_rotation=source_rotation,
        strength_bands=getattr(
            obj, 'airt_source_directivity_bands', preset
        ),
        cone_width_deg=getattr(
            obj, 'airt_source_cone_width_deg', 90.0
        ),
        custom_sh=getattr(
            obj, 'airt_source_directivity_sh', DEFAULT_CUSTOM_SH
        ),
    )
