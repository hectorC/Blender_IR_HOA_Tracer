# -*- coding: utf-8 -*-
"""
Scene and Blender-specific utilities for the Ambisonic IR Tracer.
"""
import bpy
import mathutils
import mathutils.bvhtree
import numpy as np
from typing import Tuple, List, Optional, Any
import os
import tempfile


def build_bvh(context) -> Tuple[Optional[mathutils.bvhtree.BVHTree], List[Any]]:
    """Build BVH tree from scene geometry.
    
    Returns:
        Tuple of (BVH tree, object map) where object map maps polygon indices to objects
    """
    verts = []
    polys = []
    obj_map = []  # polygon index -> object (for absorption/scatter)
    
    scene = getattr(context, "scene", None) or bpy.context.scene
    view_layer = getattr(context, "view_layer", None)
    
    # Get depsgraph for evaluated objects
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    if callable(depsgraph_get):
        depsgraph = depsgraph_get()
    else:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    
    for obj in scene.objects:
        # Skip source and receiver objects, only process visible mesh objects
        visible = obj.visible_get(view_layer=view_layer) if view_layer else obj.visible_get()
        is_source = getattr(obj, 'is_acoustic_source', False)
        is_receiver = getattr(obj, 'is_acoustic_receiver', False)
        
        if (obj.type == 'MESH' and not is_source and not is_receiver and visible):
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            
            if mesh is None:
                continue
            
            # Transform to world coordinates
            mesh.transform(obj.matrix_world)
            
            # Add vertices and polygons
            base_index = len(verts)
            verts.extend([v.co.copy() for v in mesh.vertices])
            polys.extend([tuple(base_index + vi for vi in p.vertices) for p in mesh.polygons])
            obj_map.extend([obj] * len(mesh.polygons))
            
            # Clean up
            obj_eval.to_mesh_clear()
    
    # Build BVH tree
    bvh = mathutils.bvhtree.BVHTree.FromPolygons(verts, polys) if polys else None
    return bvh, obj_map


def get_scene_sources(context) -> List[mathutils.Vector]:
    """Get all acoustic source positions from the scene."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    sources = []
    for obj in scene.objects:
        if getattr(obj, 'is_acoustic_source', False):
            sources.append(obj.location.copy())
    return sources


def get_scene_receivers(context) -> List[mathutils.Vector]:
    """Get all acoustic receiver positions from the scene."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    receivers = []
    for obj in scene.objects:
        if getattr(obj, 'is_acoustic_receiver', False):
            receivers.append(obj.location.copy())
    return receivers


def los_clear(p0: mathutils.Vector, p1: mathutils.Vector, 
              bvh: mathutils.bvhtree.BVHTree, eps: float = 1e-4) -> bool:
    """Check if line of sight is clear between two points."""
    if bvh is None:
        return True
    
    d = (p1 - p0)
    dist = d.length
    if dist <= 1e-9:
        return True
    
    dirn = d.normalized()
    hit, nrm, idx, t = bvh.ray_cast(p0 + dirn * eps, dirn)
    
    if hit is None:
        return True
    
    return t >= dist - 1e-4


def get_writable_path(filename: str) -> str:
    """Find a writable path for output files.
    
    Tries in order:
    1. Next to the .blend file
    2. Blender temp directory
    3. System temp directory
    4. User home directory
    """
    # Try blend folder first
    try:
        base = bpy.path.abspath("//")
        if base and os.path.isdir(base):
            path = os.path.join(base, filename)
            with open(path, 'ab') as _f:
                pass
            return path
    except Exception:
        pass
    
    # Blender temp dir
    try:
        if getattr(bpy.app, 'tempdir', None):
            path = os.path.join(bpy.app.tempdir, filename)
            with open(path, 'ab') as _f:
                pass
            return path
    except Exception:
        pass
    
    # System temp
    try:
        path = os.path.join(tempfile.gettempdir(), filename)
        with open(path, 'ab') as _f:
            pass
        return path
    except Exception:
        pass
    
    # Home as last resort
    return os.path.join(os.path.expanduser('~'), filename)


def get_scene_unit_scale(context) -> float:
    """Get the scene's unit scale factor."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    scale_length = getattr(unit_settings, "scale_length", 1.0) if unit_settings else 1.0
    return float(scale_length or 1.0)


def speed_of_sound_ms(context) -> float:
    """Calculate speed of sound in m/s using scene air properties."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    temp_c = float(getattr(scene, 'airt_air_temp_c', 20.0))
    rh = float(getattr(scene, 'airt_air_humidity', 50.0))
    # Cramer's formula approximation
    return float(331.3 + 0.606 * temp_c + 0.0124 * rh)


def speed_of_sound_bu(context) -> float:
    """Calculate speed of sound in Blender units."""
    unit_scale = get_scene_unit_scale(context)
    c_ms = speed_of_sound_ms(context)
    return c_ms / max(unit_scale, 1e-9)