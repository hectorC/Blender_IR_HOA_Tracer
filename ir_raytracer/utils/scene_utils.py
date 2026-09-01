# -*- coding: utf-8 -*-
"""Blender scene extraction and acoustic visibility utilities."""
from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Any, List, Optional, Sequence, Tuple

import bpy
import mathutils
import mathutils.bvhtree
import numpy as np


@dataclass
class AcousticFace:
    """One evaluated mesh polygon in world space."""

    vertices: Tuple[mathutils.Vector, ...]
    normal: mathutils.Vector
    object_ref: Any
    material_ref: Any = None
    material_snapshot: Any = None

    @property
    def acoustic_ref(self) -> Any:
        """Return an enabled Blender material or the legacy object fallback."""
        if (
            self.material_ref is not None
            and bool(getattr(self.material_ref, 'airt_acoustic_enabled', False))
        ):
            return self.material_ref
        return self.object_ref


@dataclass
class AcousticScene:
    """Acceleration structure and face metadata for one render."""

    bvh: Optional[mathutils.bvhtree.BVHTree]
    faces: List[AcousticFace]

    @property
    def object_map(self) -> List[Any]:
        return [face.object_ref for face in self.faces]


def _face_normal(vertices: Sequence[mathutils.Vector]) -> mathutils.Vector:
    if len(vertices) < 3:
        return mathutils.Vector((0.0, 0.0, 1.0))
    origin = vertices[0]
    for index in range(1, len(vertices) - 1):
        normal = (vertices[index] - origin).cross(vertices[index + 1] - origin)
        if normal.length_squared > 1e-16:
            return normal.normalized()
    return mathutils.Vector((0.0, 0.0, 1.0))


def build_acoustic_scene(context) -> AcousticScene:
    """Build a BVH from visible evaluated meshes using evaluated transforms."""
    from ..core.acoustics import MaterialProperties

    scene = getattr(context, "scene", None) or bpy.context.scene
    view_layer = getattr(context, "view_layer", None)
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    depsgraph = (
        depsgraph_get()
        if callable(depsgraph_get)
        else bpy.context.evaluated_depsgraph_get()
    )

    vertices: List[mathutils.Vector] = []
    polygons: List[Tuple[int, ...]] = []
    faces: List[AcousticFace] = []
    material_snapshots = {}

    for obj in scene.objects:
        visible = obj.visible_get(view_layer=view_layer) if view_layer else obj.visible_get()
        is_selected_source = obj == getattr(scene, 'airt_source_object', None)
        is_selected_receiver = obj == getattr(scene, 'airt_receiver_object', None)
        if (
            obj.type != 'MESH'
            or not visible
            or is_selected_source
            or is_selected_receiver
            or getattr(obj, 'is_acoustic_source', False)
            or getattr(obj, 'is_acoustic_receiver', False)
        ):
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        if mesh is None:
            continue
        try:
            transform = obj_eval.matrix_world
            world_vertices = [transform @ vertex.co for vertex in mesh.vertices]
            base_index = len(vertices)
            vertices.extend(vertex.copy() for vertex in world_vertices)

            for polygon in mesh.polygons:
                local_indices = tuple(int(index) for index in polygon.vertices)
                if len(local_indices) < 3:
                    continue
                face_vertices = tuple(world_vertices[index].copy() for index in local_indices)
                material_index = int(polygon.material_index)
                material_ref = (
                    mesh.materials[material_index]
                    if 0 <= material_index < len(mesh.materials)
                    else None
                )
                if material_ref is not None:
                    material_ref = getattr(
                        material_ref, "original", material_ref
                    )
                acoustic_owner = (
                    material_ref
                    if (
                        material_ref is not None
                        and bool(getattr(
                            material_ref, 'airt_acoustic_enabled', False
                        ))
                    )
                    else obj
                )
                owner_key = id(acoustic_owner)
                material_snapshot = material_snapshots.get(owner_key)
                if material_snapshot is None:
                    material_snapshot = MaterialProperties(acoustic_owner)
                    material_snapshots[owner_key] = material_snapshot
                polygons.append(tuple(base_index + index for index in local_indices))
                faces.append(AcousticFace(
                    vertices=face_vertices,
                    normal=_face_normal(face_vertices),
                    object_ref=obj,
                    material_ref=material_ref,
                    material_snapshot=material_snapshot,
                ))
        finally:
            obj_eval.to_mesh_clear()

    bvh = (
        mathutils.bvhtree.BVHTree.FromPolygons(vertices, polygons)
        if polygons
        else None
    )
    return AcousticScene(bvh=bvh, faces=faces)


def build_bvh(context) -> Tuple[Optional[mathutils.bvhtree.BVHTree], List[Any]]:
    """Compatibility wrapper returning the BVH and polygon-to-object map."""
    acoustic_scene = build_acoustic_scene(context)
    return acoustic_scene.bvh, acoustic_scene.object_map


def _evaluated_world_position(obj, depsgraph) -> mathutils.Vector:
    try:
        return obj.evaluated_get(depsgraph).matrix_world.translation.copy()
    except Exception:
        return obj.matrix_world.translation.copy()


def _evaluated_world_rotation(obj, depsgraph) -> mathutils.Quaternion:
    """Return evaluated world rotation with scale and translation removed."""
    try:
        matrix = obj.evaluated_get(depsgraph).matrix_world
    except Exception:
        matrix = obj.matrix_world
    rotation = matrix.to_quaternion()
    rotation.normalize()
    return rotation


def get_scene_source_objects(context) -> List[Any]:
    scene = getattr(context, "scene", None) or bpy.context.scene
    return [obj for obj in scene.objects if getattr(obj, 'is_acoustic_source', False)]


def get_scene_receiver_objects(context) -> List[Any]:
    scene = getattr(context, "scene", None) or bpy.context.scene
    return [obj for obj in scene.objects if getattr(obj, 'is_acoustic_receiver', False)]


def object_world_position(context, obj) -> mathutils.Vector:
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    depsgraph = (
        depsgraph_get()
        if callable(depsgraph_get)
        else bpy.context.evaluated_depsgraph_get()
    )
    return _evaluated_world_position(obj, depsgraph)


def object_world_rotation(context, obj) -> mathutils.Quaternion:
    """Return an object's evaluated world-space orientation."""
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    depsgraph = (
        depsgraph_get()
        if callable(depsgraph_get)
        else bpy.context.evaluated_depsgraph_get()
    )
    return _evaluated_world_rotation(obj, depsgraph)


def get_scene_sources(context) -> List[mathutils.Vector]:
    return [object_world_position(context, obj) for obj in get_scene_source_objects(context)]


def get_scene_receivers(context) -> List[mathutils.Vector]:
    return [object_world_position(context, obj) for obj in get_scene_receiver_objects(context)]


def los_clear(
    p0: mathutils.Vector,
    p1: mathutils.Vector,
    bvh: Optional[mathutils.bvhtree.BVHTree],
    eps: float = 1e-4,
) -> bool:
    """Return whether an opaque straight segment is unobstructed."""
    if bvh is None:
        return True
    delta = p1 - p0
    distance = delta.length
    if distance <= eps:
        return True
    direction = delta / distance
    hit, _normal, _index, hit_distance = bvh.ray_cast(
        p0 + direction * eps, direction
    )
    return hit is None or hit_distance >= distance - eps * 2.0


def spectral_visibility(
    p0: mathutils.Vector,
    p1: mathutils.Vector,
    acoustic_scene: AcousticScene,
    eps: float = 1e-4,
    max_surfaces: int = 32,
) -> np.ndarray:
    """Accumulate per-band energy transmission along a straight segment."""
    from ..core.acoustics import MaterialProperties, NUM_BANDS

    bvh = acoustic_scene.bvh
    if bvh is None:
        return np.ones(NUM_BANDS, dtype=np.float32)

    delta = p1 - p0
    total_distance = delta.length
    if total_distance <= eps:
        return np.ones(NUM_BANDS, dtype=np.float32)
    direction = delta / total_distance
    origin = p0 + direction * eps
    travelled = eps
    gain = np.ones(NUM_BANDS, dtype=np.float64)

    for _surface in range(max_surfaces):
        hit, _normal, face_index, hit_distance = bvh.ray_cast(origin, direction)
        if hit is None or face_index is None:
            break
        if travelled + hit_distance >= total_distance - eps * 2.0:
            break
        if not (0 <= face_index < len(acoustic_scene.faces)):
            return np.zeros(NUM_BANDS, dtype=np.float32)

        face = acoustic_scene.faces[face_index]
        material = face.material_snapshot
        if material is None:
            material = MaterialProperties(face.acoustic_ref)
        gain *= material.transmission_spectrum
        if float(np.max(gain)) <= 1e-10:
            return np.zeros(NUM_BANDS, dtype=np.float32)

        step = float(hit_distance) + eps * 4.0
        travelled += step
        if travelled >= total_distance - eps:
            break
        origin = mathutils.Vector(hit) + direction * (eps * 4.0)
    else:
        return np.zeros(NUM_BANDS, dtype=np.float32)

    return np.clip(gain, 0.0, 1.0).astype(np.float32)


def point_in_face(point: mathutils.Vector, face: AcousticFace, eps: float = 1e-5) -> bool:
    """Test a coplanar point against a possibly concave polygon."""
    drop_axis = max(range(3), key=lambda axis: abs(face.normal[axis]))
    axes = [axis for axis in range(3) if axis != drop_axis]
    px, py = float(point[axes[0]]), float(point[axes[1]])
    polygon = [
        (float(vertex[axes[0]]), float(vertex[axes[1]]))
        for vertex in face.vertices
    ]

    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        edge_x = x2 - x1
        edge_y = y2 - y1
        cross = (px - x1) * edge_y - (py - y1) * edge_x
        if abs(cross) <= eps:
            dot = (px - x1) * edge_x + (py - y1) * edge_y
            if -eps <= dot <= edge_x * edge_x + edge_y * edge_y + eps:
                return True
        if (y1 > py) != (y2 > py):
            intersection_x = x1 + (py - y1) * edge_x / (edge_y or 1e-20)
            if px < intersection_x:
                inside = not inside
    return inside


def get_writable_path(filename: str) -> str:
    """Resolve a writable output path without creating a probe file."""
    requested = bpy.path.abspath(filename) if filename else ""
    if requested:
        parent = os.path.dirname(requested)
        if parent and os.path.isdir(parent) and os.access(parent, os.W_OK):
            return requested

    basename = os.path.basename(filename or "ambisonic_ir.wav")
    blend_directory = bpy.path.abspath("//")
    if blend_directory and os.path.isdir(blend_directory) and os.access(blend_directory, os.W_OK):
        return os.path.join(blend_directory, basename)
    if getattr(bpy.app, 'tempdir', None):
        return os.path.join(bpy.app.tempdir, basename)
    return os.path.join(tempfile.gettempdir(), basename)


def get_scene_unit_scale(context) -> float:
    scene = getattr(context, "scene", None) or bpy.context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    scale_length = getattr(unit_settings, "scale_length", 1.0) if unit_settings else 1.0
    return float(scale_length or 1.0)


def speed_of_sound_ms(context) -> float:
    scene = getattr(context, "scene", None) or bpy.context.scene
    temperature_c = float(getattr(scene, 'airt_air_temp_c', 20.0))
    relative_humidity = float(getattr(scene, 'airt_air_humidity', 50.0))
    return float(331.3 + 0.606 * temperature_c + 0.0124 * relative_humidity)


def speed_of_sound_bu(context) -> float:
    return speed_of_sound_ms(context) / max(get_scene_unit_scale(context), 1e-9)
