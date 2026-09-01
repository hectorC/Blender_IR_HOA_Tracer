# -*- coding: utf-8 -*-
"""
Ambisonic encoding and spatial audio utilities.
"""
import mathutils
import numpy as np
from math import pi, sqrt, sin, cos


def apply_orientation_transform(direction: mathutils.Vector, yaw_offset_deg: float = 0.0, 
                               invert_z: bool = False) -> mathutils.Vector:
    """Apply orientation transforms to map Blender coordinates to AmbiX.
    
    Blender has no universal world-space "forward" direction, so the add-on uses
    the same reference frame as Blender's Front view: -Y is acoustic front, +X
    is acoustic left, and +Z is up. These axes map directly to AmbiX +X, +Y,
    and +Z respectively. A yaw rotation and optional Z flip are applied after
    that basis conversion.
    
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

    The expressions are the real spherical harmonics without the
    Condon-Shortley phase, as required by the ambiX convention. Keeping the
    equations explicit makes channel ordering and polarity auditable and avoids
    a SciPy dependency for a fixed third-order encoder.
    
    Args:
        direction: Unit direction vector
        
    Returns:
        16-channel ACN/SN3D encoded ambisonic coefficients
    """
    x, y, z = float(direction.x), float(direction.y), float(direction.z)
    r = sqrt(x*x + y*y + z*z)
    
    if r <= 1e-12:
        result = np.zeros(16, dtype=np.float32)
        result[0] = 1.0
        return result

    # Work on the unit sphere even if callers provide a scaled direction.
    x /= r
    y /= r
    z /= r

    sqrt3 = sqrt(3.0)
    sqrt15 = sqrt(15.0)
    sqrt_3_over_8 = sqrt(3.0 / 8.0)
    sqrt_5_over_8 = sqrt(5.0 / 8.0)

    # ACN index n(n + 1) + m, ordered by n=0..3 and m=-n..n.
    return np.array((
        # Order 0
        1.0,
        # Order 1: Y, Z, X
        y,
        z,
        x,
        # Order 2: V, T, R, S, U
        sqrt3 * x * y,
        sqrt3 * y * z,
        0.5 * (3.0 * z * z - 1.0),
        sqrt3 * x * z,
        0.5 * sqrt3 * (x * x - y * y),
        # Order 3: Q, O, M, K, L, N, P
        sqrt_5_over_8 * y * (3.0 * x * x - y * y),
        sqrt15 * x * y * z,
        sqrt_3_over_8 * y * (5.0 * z * z - 1.0),
        0.5 * z * (5.0 * z * z - 3.0),
        sqrt_3_over_8 * x * (5.0 * z * z - 1.0),
        0.5 * sqrt15 * z * (x * x - y * y),
        sqrt_5_over_8 * x * (x * x - 3.0 * y * y),
    ), dtype=np.float32)


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
    """Third-order ambisonic encoder with listener orientation control."""
    
    def __init__(
        self,
        yaw_offset_deg: float = 0.0,
        invert_z: bool = False,
        receiver_rotation: mathutils.Quaternion | None = None,
    ):
        """Initialize encoder with orientation settings."""
        self.yaw_offset_deg = yaw_offset_deg
        self.invert_z = invert_z
        self.use_receiver_orientation = receiver_rotation is not None
        if receiver_rotation is None:
            rotation = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            rotation = receiver_rotation.copy()
            rotation.normalize()
        self.receiver_rotation = rotation
        self._world_to_receiver = rotation.inverted()
    
    def encode(self, direction: mathutils.Vector) -> np.ndarray:
        """Encode a direction vector to ambisonic coefficients."""
        receiver_relative = self._world_to_receiver @ direction
        oriented_dir = apply_orientation_transform(
            receiver_relative, self.yaw_offset_deg, self.invert_z
        )
        return encode_ambisonics_3rd_order(oriented_dir)
