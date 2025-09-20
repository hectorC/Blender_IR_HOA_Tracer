# -*- coding: utf-8 -*-
"""
Mathematical utilities for acoustic ray tracing.
"""
import mathutils
import numpy as np
from math import pi, sqrt, sin, cos, atan2, asin, acos
import random
from typing import Tuple, List, Optional


def safe_divide(a: float, b: float, epsilon: float = 1e-9) -> float:
    """Safely divide a by b, avoiding division by zero."""
    return a / max(abs(b), epsilon) if abs(b) > epsilon else 0.0


def stable_acos(x: float) -> float:
    """Numerically stable arccosine function."""
    return acos(np.clip(x, -1.0, 1.0))


def stable_asin(x: float) -> float:
    """Numerically stable arcsine function."""
    return asin(np.clip(x, -1.0, 1.0))


def reflect(vec: mathutils.Vector, normal: mathutils.Vector) -> mathutils.Vector:
    """Reflect a vector about a normal."""
    return (vec - 2.0 * vec.dot(normal) * normal).normalized()


def orthonormal_basis(n: mathutils.Vector) -> Tuple[mathutils.Vector, mathutils.Vector, mathutils.Vector]:
    """Generate an orthonormal basis from a normal vector.
    
    Returns:
        Tuple of (tangent, bitangent, normal) vectors
    """
    n = n.normalized()
    if abs(n.x) < 0.5:
        helper = mathutils.Vector((1.0, 0.0, 0.0))
    else:
        helper = mathutils.Vector((0.0, 1.0, 0.0))
    t = n.cross(helper).normalized()
    b = n.cross(t).normalized()
    return t, b, n


def cosine_weighted_hemisphere(normal: mathutils.Vector) -> mathutils.Vector:
    """Generate a cosine-weighted sample about a given normal."""
    u1 = random.random()
    u2 = random.random()
    r = sqrt(u1)
    theta = 2.0 * pi * u2
    x = r * cos(theta)
    y = r * sin(theta)
    z = sqrt(max(0.0, 1.0 - u1))
    t, b, n = orthonormal_basis(normal)
    v = (t * x + b * y + n * z).normalized()
    return v


def fibonacci_sphere(samples: int) -> List[Tuple[float, float, float]]:
    """Generate uniformly distributed points on a sphere using Fibonacci spiral."""
    points = []
    phi = pi * (3.0 - sqrt(5.0))  # Golden angle
    
    for i in range(samples):
        y = 1.0 - (i / float(samples - 1)) * 2.0  # y goes from 1 to -1
        radius = sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = cos(theta) * radius
        z = sin(theta) * radius
        points.append((x, y, z))
    
    return points


def generate_ray_directions(samples: int) -> List[Tuple[float, float, float]]:
    """Generate ray directions with random rotation and jitter."""
    if samples <= 0:
        return []
    
    base = fibonacci_sphere(samples)
    
    # Random rotation axis and angle
    axis = mathutils.Vector((np.random.normal(), np.random.normal(), np.random.normal()))
    if axis.length == 0.0:
        axis = mathutils.Vector((0.0, 0.0, 1.0))
    axis.normalize()
    
    angle = 2.0 * pi * np.random.random()
    rot = mathutils.Matrix.Rotation(angle, 3, axis)
    
    # Add jitter to reduce aliasing
    jitter_strength = 0.75 / float(max(samples, 1))
    dirs = []
    
    for p in base:
        v = rot @ mathutils.Vector(p)
        jitter = mathutils.Vector((
            np.random.uniform(-jitter_strength, jitter_strength),
            np.random.uniform(-jitter_strength, jitter_strength),
            np.random.uniform(-jitter_strength, jitter_strength)
        ))
        candidate = v + jitter
        if candidate.length == 0.0:
            candidate = v
        candidate = candidate.normalized()
        dirs.append((float(candidate.x), float(candidate.y), float(candidate.z)))
    
    random.shuffle(dirs)
    return dirs


def segment_hits_sphere(p0: mathutils.Vector, p1: mathutils.Vector, 
                       center: mathutils.Vector, radius: float) -> Tuple[bool, Optional[float], Optional[mathutils.Vector]]:
    """Check if segment p0->p1 intersects a sphere at center with radius.
    
    Returns:
        Tuple of (hit, t, point) where t is in [0,1] along the segment from p0
    """
    v = p1 - p0
    w = p0 - center
    a = v.dot(v)
    b = 2.0 * v.dot(w)
    c = w.dot(w) - radius * radius
    
    disc = b * b - 4 * a * c
    if disc < 0.0 or a <= 0.0:
        return False, None, None
    
    sd = sqrt(disc)
    t1 = (-b - sd) / (2 * a)
    t2 = (-b + sd) / (2 * a)
    
    t_hit = None
    if 0.0 <= t1 <= 1.0:
        t_hit = t1
    elif 0.0 <= t2 <= 1.0:
        t_hit = t2
    
    if t_hit is None:
        return False, None, None
    
    point = p0 + v * t_hit
    return True, float(t_hit), point


def jitter_specular_direction(direction: mathutils.Vector, roughness_rad: float) -> mathutils.Vector:
    """Apply micro-roughness jitter to a specular reflection direction."""
    v = direction.normalized()
    if roughness_rad <= 1e-6:
        return v
    
    # Create orthonormal basis
    if abs(v.x) < 0.5:
        helper = mathutils.Vector((1.0, 0.0, 0.0))
    else:
        helper = mathutils.Vector((0.0, 1.0, 0.0))
    
    t = v.cross(helper).normalized()
    b = v.cross(t).normalized()
    
    # Sample within cone
    u = random.random()
    vphi = 2.0 * pi * random.random()
    cos_a = 1.0 - u * (1.0 - cos(roughness_rad))
    sin_a = sqrt(max(0.0, 1.0 - cos_a * cos_a))
    
    return (v * cos_a + t * (sin_a * cos(vphi)) + b * (sin_a * sin(vphi))).normalized()