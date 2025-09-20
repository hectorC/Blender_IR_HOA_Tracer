# -*- coding: utf-8 -*-
"""
Ambisonic encoding and spatial audio utilities.
"""
import mathutils
import numpy as np
from math import pi, sqrt, sin, cos, atan2, asin
from typing import Optional

# Try to import scipy for spherical harmonics
try:
    from scipy.special import lpmv
    HAVE_SCIPY = True
except ImportError:
    lpmv = None
    HAVE_SCIPY = False


# ACN order mapping for 3rd order (16 channels)
ACN_ORDERS = (0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3)


def apply_orientation_transform(direction: mathutils.Vector, yaw_offset_deg: float = 0.0, 
                               invert_z: bool = False) -> mathutils.Vector:
    """Apply orientation transforms to map Blender coordinates to AmbiX.
    
    Maps Blender world axes (X right, Y forward, Z up) to AmbiX axes (X front, Y left, Z up),
    then applies yaw rotation and optional Z flip.
    
    Args:
        direction: Direction vector in Blender coordinates
        yaw_offset_deg: Yaw rotation around Z axis in degrees
        invert_z: Whether to invert the Z axis
        
    Returns:
        Transformed direction vector
    """
    # Blender -> AmbiX basis mapping: X_a = -Y_b, Y_a = +X_b, Z_a = +Z_b
    xb, yb, zb = float(direction.x), float(direction.y), float(direction.z)
    xa = -yb
    ya = xb
    za = zb
    
    # Yaw around +Z (AmbiX frame)
    yaw = float(yaw_offset_deg) * pi / 180.0
    cz, sz = cos(yaw), sin(yaw)
    xr = xa * cz - ya * sz
    yr = xa * sz + ya * cz
    zr = za
    
    # Optional Z flip
    if invert_z:
        zr = -zr
    
    return mathutils.Vector((xr, yr, zr)).normalized()


def encode_ambisonics_3rd_order(direction: mathutils.Vector) -> np.ndarray:
    """Encode direction as 3rd-order ambisonic coefficients (ACN/SN3D).
    
    Args:
        direction: Unit direction vector
        
    Returns:
        16-channel ACN/SN3D encoded ambisonic coefficients
    """
    if not HAVE_SCIPY:
        # Fallback: return W channel only
        result = np.zeros(16, dtype=np.float32)
        result[0] = 1.0
        return result
    
    x, y, z = float(direction.x), float(direction.y), float(direction.z)
    r = sqrt(x*x + y*y + z*z)
    
    if r == 0:
        result = np.zeros(16, dtype=np.float32)
        result[0] = 1.0
        return result
    
    # Spherical coordinates
    az = atan2(y, x)  # Azimuth
    el = asin(z / r)  # Elevation
    s = sin(el)
    
    def N_sn3d(l: int, m: int) -> float:
        """SN3D normalization factor."""
        m = abs(m)
        from math import factorial
        return sqrt(float(factorial(l - m)) / float(factorial(l + m)))
    
    sh = np.zeros(16, dtype=np.float32)
    idx = 0
    
    # Calculate spherical harmonics up to 3rd order
    for l in range(0, 4):
        for m in range(-l, l + 1):
            P = float(lpmv(abs(m), l, s))
            
            if m == 0:
                ylm = N_sn3d(l, 0) * P
            elif m > 0:
                ylm = sqrt(2.0) * N_sn3d(l, m) * cos(m * az) * P
            else:
                ylm = sqrt(2.0) * N_sn3d(l, -m) * sin(-m * az) * P
            
            sh[idx] = ylm
            idx += 1
    
    return sh


def apply_near_field_compensation(ambi_vec: np.ndarray, distance_m: float, 
                                 reference_m: float = 1.0) -> np.ndarray:
    """Apply near-field compensation to ambisonic coefficients.
    
    Boosts higher orders for close distances to maintain spatial impression.
    
    Args:
        ambi_vec: Ambisonic coefficient vector
        distance_m: Distance to source in meters
        reference_m: Reference distance in meters
        
    Returns:
        Compensated ambisonic coefficients
    """
    if reference_m <= 0.0:
        return ambi_vec
    
    ref = max(reference_m, 1e-3)
    dist = max(float(distance_m), 1e-3)
    
    if dist >= ref:
        return ambi_vec
    
    scale = ref / dist
    gains = np.ones_like(ambi_vec, dtype=np.float32)
    
    # Apply frequency-dependent boost to higher orders
    for idx, order in enumerate(ACN_ORDERS):
        if order <= 0:
            continue
        # Limit boost to prevent instability
        gains[idx] = min(scale ** (0.5 * order), 8.0)
    
    return ambi_vec * gains


def get_ambi_channel_names() -> list:
    """Get standard ACN channel names for 3rd order ambisonics."""
    names = []
    for l in range(4):
        for m in range(-l, l + 1):
            if l == 0:
                names.append("W")
            elif l == 1:
                if m == -1:
                    names.append("Y")
                elif m == 0:
                    names.append("Z")
                elif m == 1:
                    names.append("X")
            elif l == 2:
                if m == -2:
                    names.append("V")
                elif m == -1:
                    names.append("T")
                elif m == 0:
                    names.append("R")
                elif m == 1:
                    names.append("S")
                elif m == 2:
                    names.append("U")
            elif l == 3:
                if m == -3:
                    names.append("Q")
                elif m == -2:
                    names.append("O")
                elif m == -1:
                    names.append("M")
                elif m == 0:
                    names.append("K")
                elif m == 1:
                    names.append("L")
                elif m == 2:
                    names.append("N")
                elif m == 3:
                    names.append("P")
    return names


class AmbisonicEncoder:
    """Third-order ambisonic encoder with orientation control."""
    
    def __init__(self, yaw_offset_deg: float = 0.0, invert_z: bool = False):
        """Initialize encoder with orientation settings."""
        self.yaw_offset_deg = yaw_offset_deg
        self.invert_z = invert_z
    
    def encode(self, direction: mathutils.Vector) -> np.ndarray:
        """Encode a direction vector to ambisonic coefficients."""
        oriented_dir = apply_orientation_transform(
            direction, self.yaw_offset_deg, self.invert_z
        )
        return encode_ambisonics_3rd_order(oriented_dir)
    
    def encode_with_nf_compensation(self, direction: mathutils.Vector, 
                                   distance_m: float, reference_m: float = 1.0) -> np.ndarray:
        """Encode with near-field compensation."""
        ambi_vec = self.encode(direction)
        return apply_near_field_compensation(ambi_vec, distance_m, reference_m)