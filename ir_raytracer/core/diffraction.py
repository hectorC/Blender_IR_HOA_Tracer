"""Geometry-aware, artist-oriented single-edge diffraction helpers."""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi
from typing import List, Sequence

import mathutils
from mathutils.kdtree import KDTree
import numpy as np

from .acoustics import BAND_CENTERS_HZ


@dataclass
class DiffractionEdge:
    """A boundary or sharp mesh edge in world space."""

    start: mathutils.Vector
    end: mathutils.Vector

    @property
    def midpoint(self) -> mathutils.Vector:
        return (self.start + self.end) * 0.5


@dataclass
class DiffractionPath:
    """One visible broken path through a candidate diffraction edge."""

    point: mathutils.Vector
    distance_bu: float
    path_difference_m: float
    bend_angle_rad: float


class DiffractionEdgeIndex:
    """Midpoint KD-tree for bounded edge candidate searches."""

    def __init__(self, edges: Sequence[DiffractionEdge]):
        self.edges = list(edges)
        self._tree = None
        if self.edges:
            tree = KDTree(len(self.edges))
            for index, edge in enumerate(self.edges):
                tree.insert(edge.midpoint, index)
            tree.balance()
            self._tree = tree

    def nearest(self, point: mathutils.Vector, count: int) -> List[DiffractionEdge]:
        if not self.edges or self._tree is None:
            return []
        count = max(1, min(int(count), len(self.edges)))
        return [
            self.edges[index]
            for _coordinate, index, _distance in self._tree.find_n(point, count)
        ]


def _polygon_normal(
    vertices: Sequence[mathutils.Vector], polygon: Sequence[int]
) -> mathutils.Vector:
    if len(polygon) < 3:
        return mathutils.Vector((0.0, 0.0, 0.0))
    origin = vertices[polygon[0]]
    for index in range(1, len(polygon) - 1):
        normal = (vertices[polygon[index]] - origin).cross(
            vertices[polygon[index + 1]] - origin
        )
        if normal.length_squared > 1e-16:
            return normal.normalized()
    return mathutils.Vector((0.0, 0.0, 0.0))


def extract_diffraction_edges(
    vertices: Sequence[mathutils.Vector],
    polygons: Sequence[Sequence[int]],
    min_dihedral_rad: float = 15.0 * pi / 180.0,
) -> List[DiffractionEdge]:
    """Extract boundary and non-coplanar polygon edges from a mesh."""
    adjacency = {}
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        normal = _polygon_normal(vertices, polygon)
        for offset, vertex_index in enumerate(polygon):
            next_index = polygon[(offset + 1) % len(polygon)]
            key = tuple(sorted((int(vertex_index), int(next_index))))
            adjacency.setdefault(key, []).append(normal)

    result = []
    for (start_index, end_index), normals in adjacency.items():
        is_boundary = len(normals) == 1
        is_sharp = False
        if len(normals) >= 2:
            for first_index in range(len(normals) - 1):
                for second_index in range(first_index + 1, len(normals)):
                    dot = abs(float(normals[first_index].dot(normals[second_index])))
                    angle = acos(float(np.clip(dot, -1.0, 1.0)))
                    if angle >= min_dihedral_rad:
                        is_sharp = True
                        break
                if is_sharp:
                    break
        if is_boundary or is_sharp:
            result.append(DiffractionEdge(
                vertices[start_index].copy(), vertices[end_index].copy()
            ))
    return result


def build_diffraction_edge_index(context) -> DiffractionEdgeIndex:
    """Extract diffraction candidates from visible evaluated scene meshes."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return DiffractionEdgeIndex([])

    view_layer = getattr(context, "view_layer", None)
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    if not callable(depsgraph_get):
        return DiffractionEdgeIndex([])
    depsgraph = depsgraph_get()
    edges = []

    for obj in scene.objects:
        visible = obj.visible_get(view_layer=view_layer) if view_layer else obj.visible_get()
        if (
            obj.type != 'MESH'
            or not visible
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
            vertices = [transform @ vertex.co for vertex in mesh.vertices]
            polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
            edges.extend(extract_diffraction_edges(vertices, polygons))
        finally:
            obj_eval.to_mesh_clear()

    return DiffractionEdgeIndex(edges)


def _minimum_broken_path_point(
    origin: mathutils.Vector,
    receiver: mathutils.Vector,
    edge: DiffractionEdge,
) -> mathutils.Vector:
    """Minimize origin-edge-receiver length along a finite edge segment."""
    segment = edge.end - edge.start
    if segment.length_squared <= 1e-16:
        return edge.start.copy()

    low = 0.0
    high = 1.0
    for _iteration in range(24):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        first_point = edge.start + segment * first
        second_point = edge.start + segment * second
        first_length = (first_point - origin).length + (receiver - first_point).length
        second_length = (second_point - origin).length + (receiver - second_point).length
        if first_length <= second_length:
            high = second
        else:
            low = first
    return edge.start + segment * ((low + high) * 0.5)


def find_diffraction_paths(
    origin: mathutils.Vector,
    receiver: mathutils.Vector,
    edge_index: DiffractionEdgeIndex,
    bvh,
    unit_scale: float,
    max_angle_rad: float,
    max_paths: int,
    eps: float = 1e-4,
) -> List[DiffractionPath]:
    """Find short, visible single-edge detours between two points."""
    from ..utils.scene_utils import los_clear

    direct = receiver - origin
    direct_distance = direct.length
    if direct_distance <= eps or max_paths <= 0 or max_angle_rad <= 0.0:
        return []

    query_point = (origin + receiver) * 0.5
    if bvh is not None:
        hit, _normal, _index, _distance = bvh.ray_cast(
            origin + direct.normalized() * eps, direct.normalized()
        )
        if hit is not None:
            query_point = mathutils.Vector(hit)

    candidates = edge_index.nearest(query_point, max(8, max_paths * 3))
    paths = []
    for edge in candidates:
        edge_point = _minimum_broken_path_point(origin, receiver, edge)
        toward_origin = origin - edge_point
        toward_receiver = receiver - edge_point
        if toward_origin.length <= eps or toward_receiver.length <= eps:
            continue

        # Nudge the visibility test into the open wedge at the edge so the
        # adjacent polygons are not mistaken for blockers at their endpoint.
        offset_direction = toward_origin.normalized() + toward_receiver.normalized()
        visibility_points = [edge_point.copy()]
        if offset_direction.length_squared > 1e-16:
            offset = offset_direction.normalized() * (eps * 4.0)
            visibility_points = [edge_point + offset, edge_point - offset, edge_point]

        visibility_point = next((
            point
            for point in visibility_points
            if los_clear(origin, point, bvh, eps)
            and los_clear(point, receiver, bvh, eps)
        ), None)
        if visibility_point is None:
            continue

        first_direction = (visibility_point - origin).normalized()
        second_direction = (receiver - visibility_point).normalized()
        bend_angle = acos(float(np.clip(first_direction.dot(second_direction), -1.0, 1.0)))
        if bend_angle > max_angle_rad:
            continue

        distance_bu = (
            (visibility_point - origin).length
            + (receiver - visibility_point).length
        )
        path_difference_m = max(
            0.0, (distance_bu - direct_distance) * max(float(unit_scale), 1e-9)
        )
        paths.append(DiffractionPath(
            visibility_point,
            distance_bu,
            path_difference_m,
            bend_angle,
        ))

    paths.sort(key=lambda path: path.distance_bu)
    return paths[:max_paths]


def maekawa_diffraction_gains(
    path_difference_m: float,
    speed_of_sound_ms: float,
    bend_angle_rad: float = 0.0,
    max_angle_rad: float = pi / 2.0,
) -> np.ndarray:
    """Return capped pressure gains from a Fresnel/Maekawa shadow model."""
    speed = max(float(speed_of_sound_ms), 1e-6)
    difference = max(float(path_difference_m), 0.0)
    frequencies = np.array(BAND_CENTERS_HZ, dtype=np.float64)
    fresnel_number = 2.0 * difference * frequencies / speed
    insertion_loss_db = np.minimum(
        25.0,
        10.0 * np.log10(3.0 + 20.0 * fresnel_number),
    )
    gains = 10.0 ** (-insertion_loss_db / 20.0)

    if max_angle_rad > 1e-9:
        angle_fraction = np.clip(bend_angle_rad / max_angle_rad, 0.0, 1.0)
        gains *= np.cos(0.5 * pi * angle_fraction) ** 2
    return gains.astype(np.float32)
