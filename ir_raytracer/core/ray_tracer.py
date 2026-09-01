"""Receiver-centric acoustic energy transport and ambisonic IR rendering."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import mathutils
import numpy as np

from .acoustics import (
    MaterialProperties,
    NUM_BANDS,
    air_attenuation_bands,
)
from .ambisonic import AmbisonicEncoder
from .diffraction import (
    build_diffraction_edge_index,
    find_diffraction_paths,
    maekawa_diffraction_gains,
)
from .synthesis import AcousticEvent, SynthesisStats, synthesize_ambisonic_ir
from ..utils.math_utils import reflect
from ..utils.scene_utils import (
    AcousticScene,
    build_acoustic_scene,
    get_scene_unit_scale,
    object_world_rotation,
    point_in_face,
    spectral_visibility,
    speed_of_sound_bu,
)


@dataclass
class AcousticRenderConfig:
    """Immutable settings used by one acoustic render."""

    ray_count: int
    max_bounces: int
    sample_rate: int
    duration_seconds: float
    output_content: str
    early_reflections: bool
    early_order: int
    early_path_budget: int
    seed: int
    min_energy: float
    rr_enabled: bool
    rr_start: int
    rr_survival: float
    specular_roughness_rad: float
    unit_scale: float
    speed_of_sound_bu: float
    air_enabled: bool
    air_temperature_c: float
    air_humidity_pct: float
    air_pressure_kpa: float
    diffraction_enabled: bool
    diffraction_paths: int
    diffraction_max_angle_rad: float
    early_gain_db: float
    diffuse_gain_db: float
    encoder: AmbisonicEncoder
    eps: float = 1e-4

    @classmethod
    def from_context(cls, context) -> "AcousticRenderConfig":
        scene = context.scene
        output_content = getattr(scene, 'airt_output_content', 'FULL')
        if output_content not in {'FULL', 'REFLECTIONS', 'DIFFUSE'}:
            output_content = 'FULL'
        use_receiver_orientation = bool(
            getattr(scene, 'airt_use_receiver_orientation', True)
        )
        receiver_object = getattr(scene, 'airt_receiver_object', None)
        if receiver_object is None:
            receiver_object = next((
                obj for obj in scene.objects
                if getattr(obj, 'is_acoustic_receiver', False)
            ), None)
        receiver_rotation = (
            object_world_rotation(context, receiver_object)
            if use_receiver_orientation and receiver_object is not None
            else None
        )
        return cls(
            ray_count=max(1, int(getattr(scene, 'airt_num_rays', 1024))),
            max_bounces=max(0, int(getattr(scene, 'airt_max_order', 32))),
            sample_rate=max(8000, int(getattr(scene, 'airt_sr', 48000))),
            duration_seconds=max(0.1, float(getattr(scene, 'airt_ir_seconds', 2.0))),
            output_content=output_content,
            early_reflections=bool(getattr(scene, 'airt_early_reflections', True)),
            early_order=int(np.clip(
                getattr(scene, 'airt_early_order', 2), 1, 3
            )),
            early_path_budget=max(
                1, int(getattr(scene, 'airt_early_path_budget', 1_000_000))
            ),
            seed=int(getattr(scene, 'airt_seed', 1)),
            min_energy=max(1e-12, float(getattr(scene, 'airt_min_throughput', 1e-6))),
            rr_enabled=bool(getattr(scene, 'airt_rr_enable', True)),
            rr_start=max(0, int(getattr(scene, 'airt_rr_start', 20))),
            rr_survival=float(np.clip(getattr(scene, 'airt_rr_p', 0.97), 0.05, 1.0)),
            specular_roughness_rad=max(
                0.0, float(getattr(scene, 'airt_spec_rough_deg', 8.0)) * pi / 180.0
            ),
            unit_scale=get_scene_unit_scale(context),
            speed_of_sound_bu=speed_of_sound_bu(context),
            air_enabled=bool(getattr(scene, 'airt_air_enable', True)),
            air_temperature_c=float(getattr(scene, 'airt_air_temp_c', 20.0)),
            air_humidity_pct=float(getattr(scene, 'airt_air_humidity', 50.0)),
            air_pressure_kpa=float(getattr(scene, 'airt_air_pressure_kpa', 101.325)),
            diffraction_enabled=bool(getattr(scene, 'airt_enable_diffraction', False)),
            diffraction_paths=max(0, int(getattr(scene, 'airt_diffraction_samples', 4))),
            diffraction_max_angle_rad=max(
                0.0,
                float(getattr(scene, 'airt_diffraction_max_deg', 45.0)) * pi / 180.0,
            ),
            early_gain_db=float(getattr(scene, 'airt_early_gain_db', 0.0)),
            diffuse_gain_db=float(getattr(scene, 'airt_diffuse_gain_db', 0.0)),
            encoder=AmbisonicEncoder(
                float(getattr(scene, 'airt_yaw_offset_deg', 0.0)),
                bool(getattr(scene, 'airt_invert_z', False)),
                receiver_rotation,
            ),
        )


@dataclass
class TransportStats:
    rays_traced: int = 0
    surface_interactions: int = 0
    source_connections: int = 0
    direct_events: int = 0
    early_events: int = 0
    diffraction_events: int = 0
    early_sequences_tested: int = 0
    early_highest_order: int = 0
    early_orders_skipped: int = 0


@dataclass
class AcousticRenderResult:
    ir: np.ndarray
    events: List[AcousticEvent]
    transport: TransportStats
    synthesis: SynthesisStats


@dataclass
class _EarlyPathCandidate:
    """Internal deterministic path plus its acoustic-object sequence."""

    event: AcousticEvent
    surface_id: object


@dataclass
class _SpecularSurface:
    """Coplanar finite polygons treated as one image-source reflector."""

    face_indices: List[int]
    plane_point: mathutils.Vector
    normal: mathutils.Vector
    object_ref: object

    def face_at(
        self, point: mathutils.Vector, scene: AcousticScene
    ) -> Optional[int]:
        """Return the finite polygon hit at this point, including its material."""
        return next((
            index for index in self.face_indices
            if point_in_face(point, scene.faces[index])
        ), None)


def _canonical_plane(face) -> Tuple[mathutils.Vector, float]:
    """Return an orientation-independent normalized plane equation."""
    normal = face.normal.normalized()
    for component in normal:
        if abs(float(component)) <= 1e-10:
            continue
        if component < 0.0:
            normal = -normal
        break
    return normal, float(normal.dot(face.vertices[0]))


def _build_specular_surfaces(scene: AcousticScene) -> List[_SpecularSurface]:
    """Merge triangulated coplanar faces without merging finite boundaries."""
    surfaces = []
    grouped = {}
    for face_index, face in enumerate(scene.faces):
        if not face.vertices or face.normal.length_squared <= 1e-16:
            continue
        normal, distance = _canonical_plane(face)
        key = (
            id(face.object_ref),
            round(float(normal.x), 6),
            round(float(normal.y), 6),
            round(float(normal.z), 6),
            round(distance, 5),
        )
        surface = grouped.get(key)
        if surface is None:
            surface = _SpecularSurface(
                face_indices=[],
                plane_point=face.vertices[0].copy(),
                normal=normal,
                object_ref=face.object_ref,
            )
            grouped[key] = surface
            surfaces.append(surface)
        surface.face_indices.append(face_index)
    return surfaces


def _surface_sequence_count(surface_count: int, order: int) -> int:
    """Count sequences after rejecting consecutive hits on one plane."""
    surface_count = max(0, int(surface_count))
    order = max(0, int(order))
    if order == 0:
        return 1
    if surface_count == 0:
        return 0
    return surface_count * max(0, surface_count - 1) ** (order - 1)


def _surface_sequences(surface_count: int, order: int) -> Iterator[Tuple[int, ...]]:
    """Yield fixed-order reflector sequences without consecutive duplicates."""
    if surface_count <= 0 or order <= 0:
        return

    sequence = []

    def visit() -> Iterator[Tuple[int, ...]]:
        if len(sequence) == order:
            yield tuple(sequence)
            return
        previous = sequence[-1] if sequence else None
        for surface_index in range(surface_count):
            if surface_index == previous:
                continue
            sequence.append(surface_index)
            yield from visit()
            sequence.pop()

    yield from visit()


def _cluster_unresolved_early_paths(
    candidates: List[_EarlyPathCandidate],
    time_tolerance_seconds: float = 0.00025,
    angle_tolerance_rad: float = 12.0 * pi / 180.0,
) -> List[AcousticEvent]:
    """Consolidate paths that the third-order output cannot resolve.

    A faceted concave surface can produce dozens of almost identical image
    paths, each carrying the infinite-plane amplitude. Summing those pressures
    makes the result depend on mesh subdivision and creates unbounded caustics.
    Candidates from the same object that arrive within a quarter millisecond
    and a twelve-degree cone are therefore represented by one conservative
    path. Per-band maxima retain the strongest material response without
    multiplying it by the number of modeling facets.
    """
    if not candidates:
        return []

    cosine_threshold = cos(float(angle_tolerance_rad))
    clusters = []
    for candidate in sorted(candidates, key=lambda item: item.event.delay_seconds):
        matching_cluster = None
        for cluster in reversed(clusters):
            reference = cluster[0]
            if (
                candidate.event.delay_seconds
                - reference.event.delay_seconds
                > time_tolerance_seconds
            ):
                break
            if candidate.surface_id != reference.surface_id:
                continue
            if candidate.event.order != reference.event.order:
                continue
            if (
                candidate.event.arrival_direction.dot(
                    reference.event.arrival_direction
                )
                < cosine_threshold
            ):
                continue
            matching_cluster = cluster
            break
        if matching_cluster is None:
            clusters.append([candidate])
        else:
            matching_cluster.append(candidate)

    events: List[AcousticEvent] = []
    for cluster in clusters:
        if len(cluster) == 1:
            events.append(cluster[0].event)
            continue

        energies = np.stack([
            np.maximum(candidate.event.energy_bands, 0.0)
            for candidate in cluster
        ]).astype(np.float64)
        weights = np.maximum(np.mean(energies, axis=1), 1e-20)
        delay = float(np.average(
            [candidate.event.delay_seconds for candidate in cluster],
            weights=weights,
        ))
        direction = mathutils.Vector((0.0, 0.0, 0.0))
        for candidate, weight in zip(cluster, weights):
            direction += candidate.event.arrival_direction * float(weight)
        if direction.length_squared <= 1e-20:
            direction = cluster[0].event.arrival_direction.copy()
        else:
            direction.normalize()
        events.append(AcousticEvent(
            delay_seconds=delay,
            arrival_direction=direction,
            energy_bands=np.max(energies, axis=0).astype(np.float32),
            kind='EARLY',
            order=cluster[0].event.order,
        ))
    return events


class ReceiverPathTracer:
    """Trace time-resolved acoustic energy from the listener toward the source."""

    def __init__(self, config: AcousticRenderConfig, acoustic_scene: AcousticScene):
        self.config = config
        self.scene = acoustic_scene
        self.stats = TransportStats()
        self.rng = np.random.default_rng(config.seed if config.seed != 0 else None)
        self._material_cache = {}
        self.deterministic_specular_order = 0

    def material_for_face(self, face_index: int) -> MaterialProperties:
        """Return one immutable coefficient snapshot per assignment owner."""
        owner = self.scene.faces[face_index].acoustic_ref
        key = id(owner)
        material = self._material_cache.get(key)
        if material is None:
            material = MaterialProperties(owner)
            self._material_cache[key] = material
        return material

    def _air_energy(self, distance_bu: float) -> np.ndarray:
        if not self.config.air_enabled:
            return np.ones(NUM_BANDS, dtype=np.float32)
        pressure_gain = air_attenuation_bands(
            distance_bu * self.config.unit_scale,
            self.config.air_temperature_c,
            self.config.air_humidity_pct,
            self.config.air_pressure_kpa,
        )
        return pressure_gain * pressure_gain

    def _directions(self) -> List[mathutils.Vector]:
        count = self.config.ray_count
        golden_angle = pi * (3.0 - sqrt(5.0))
        phase = float(self.rng.uniform(0.0, 2.0 * pi))
        axis_values = self.rng.normal(size=3)
        axis_length = float(np.linalg.norm(axis_values))
        axis = mathutils.Vector(
            tuple(axis_values / axis_length)
            if axis_length > 1e-12
            else (0.0, 0.0, 1.0)
        )
        rotation = mathutils.Quaternion(axis, float(self.rng.uniform(0.0, 2.0 * pi)))
        directions = []
        for index in range(count):
            z = 1.0 - 2.0 * (index + 0.5) / count
            radius = sqrt(max(0.0, 1.0 - z * z))
            angle = golden_angle * index + phase
            direction = mathutils.Vector((radius * cos(angle), radius * sin(angle), z))
            directions.append((rotation @ direction).normalized())
        return directions

    def _roughness_exponent(self) -> float:
        sigma = self.config.specular_roughness_rad
        if sigma <= 1e-4:
            return 10000.0
        return float(np.clip(2.0 / (sigma * sigma) - 2.0, 1.0, 10000.0))

    def _connection_brdf(
        self,
        material: MaterialProperties,
        normal: mathutils.Vector,
        source_direction: mathutils.Vector,
        receiver_direction: mathutils.Vector,
        include_specular: bool,
    ) -> np.ndarray:
        diffuse_energy = material.reflection_spectrum * material.diffuse_fraction
        brdf = diffuse_energy / pi
        if not include_specular:
            return brdf

        specular_energy = material.reflection_spectrum * material.specular_fraction
        mirror_direction = reflect(-source_direction, normal)
        alignment = max(0.0, float(mirror_direction.dot(receiver_direction)))
        if alignment <= 0.0:
            return brdf
        exponent = self._roughness_exponent()
        phong = (exponent + 2.0) / (2.0 * pi) * alignment ** exponent
        return brdf + specular_energy * phong

    def _sample_cosine_hemisphere(self, normal: mathutils.Vector) -> mathutils.Vector:
        first = float(self.rng.random())
        second = float(self.rng.random())
        radius = sqrt(first)
        phi = 2.0 * pi * second
        local = mathutils.Vector((radius * cos(phi), radius * sin(phi), sqrt(1.0 - first)))
        helper = (
            mathutils.Vector((1.0, 0.0, 0.0))
            if abs(normal.x) < 0.8
            else mathutils.Vector((0.0, 1.0, 0.0))
        )
        tangent = normal.cross(helper).normalized()
        bitangent = normal.cross(tangent).normalized()
        return (tangent * local.x + bitangent * local.y + normal * local.z).normalized()

    def _sample_specular_lobe(
        self, incoming: mathutils.Vector, normal: mathutils.Vector
    ) -> mathutils.Vector:
        mirror = reflect(incoming, normal)
        exponent = self._roughness_exponent()
        cos_theta = float(self.rng.random()) ** (1.0 / (exponent + 1.0))
        sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * pi * float(self.rng.random())
        helper = (
            mathutils.Vector((1.0, 0.0, 0.0))
            if abs(mirror.x) < 0.8
            else mathutils.Vector((0.0, 1.0, 0.0))
        )
        tangent = mirror.cross(helper).normalized()
        bitangent = mirror.cross(tangent).normalized()
        sampled = (
            mirror * cos_theta
            + tangent * (sin_theta * cos(phi))
            + bitangent * (sin_theta * sin(phi))
        ).normalized()
        return sampled if sampled.dot(normal) > 1e-6 else mirror

    def _sample_surface(
        self,
        direction: mathutils.Vector,
        normal: mathutils.Vector,
        material: MaterialProperties,
        throughput: np.ndarray,
    ) -> Tuple[Optional[mathutils.Vector], Optional[np.ndarray], Optional[str]]:
        transmission = material.transmission_spectrum
        diffuse = material.reflection_spectrum * material.diffuse_fraction
        specular = material.reflection_spectrum * material.specular_fraction
        components = np.stack((transmission, diffuse, specular)).astype(np.float64)
        active = np.maximum(np.asarray(throughput, dtype=np.float64), 0.0)
        active_peak = float(np.max(active))
        if active_peak <= 1e-20:
            return None, None, None

        # Continue according to the strongest still-relevant frequency band,
        # then importance-sample the interaction type. A simple band average
        # is unbiased but can discard nearly all low-frequency paths for a
        # material such as carpet; this max-band strategy greatly reduces that
        # spectral-tail variance while retaining coefficient/probability
        # compensation below.
        importance = active / active_peak
        surviving = np.sum(components, axis=0)
        survival_probability = float(np.clip(np.max(importance * surviving), 0.0, 1.0))
        metrics = np.max(components * importance[None, :], axis=1)
        metric_sum = float(np.sum(metrics))
        if survival_probability <= 1e-12 or metric_sum <= 1e-12:
            return None, None, None
        probabilities = survival_probability * metrics / metric_sum
        total_probability = float(np.sum(probabilities))
        draw = float(self.rng.random())
        if draw >= total_probability:
            return None, None, None

        if draw < probabilities[0]:
            probability = max(float(probabilities[0]), 1e-12)
            return (
                direction.normalized(),
                throughput * transmission / probability,
                'TRANSMISSION',
            )
        draw -= probabilities[0]
        if draw < probabilities[1]:
            probability = max(float(probabilities[1]), 1e-12)
            return (
                self._sample_cosine_hemisphere(normal),
                throughput * diffuse / probability,
                'DIFFUSE',
            )
        probability = max(float(probabilities[2]), 1e-12)
        return (
            self._sample_specular_lobe(direction, normal),
            throughput * specular / probability,
            'SPECULAR',
        )

    def _source_connection(
        self,
        hit_point: mathutils.Vector,
        normal: mathutils.Vector,
        reverse_direction: mathutils.Vector,
        source: mathutils.Vector,
        path_distance_bu: float,
        throughput: np.ndarray,
        material: MaterialProperties,
        first_direction: mathutils.Vector,
        bounce: int,
        all_prior_specular: bool = True,
    ) -> Optional[AcousticEvent]:
        to_source = source - hit_point
        source_distance_bu = to_source.length
        if source_distance_bu <= self.config.eps:
            return None
        source_direction = to_source / source_distance_bu
        cosine_incident = max(0.0, float(normal.dot(source_direction)))
        if cosine_incident <= 1e-8:
            return None

        visibility = spectral_visibility(
            hit_point + normal * (self.config.eps * 4.0),
            source,
            self.scene,
            self.config.eps,
        )
        if not np.any(visibility > 1e-10):
            return None

        deterministic_covers_connection = (
            self.config.early_reflections
            and self.config.output_content != 'DIFFUSE'
            and bounce + 1 <= self.deterministic_specular_order
            and all_prior_specular
        )
        brdf = self._connection_brdf(
            material,
            normal,
            source_direction,
            -reverse_direction,
            include_specular=not deterministic_covers_connection,
        )
        if not np.any(brdf > 1e-12):
            return None

        source_distance_m = max(
            source_distance_bu * self.config.unit_scale, self.config.eps
        )
        total_distance_bu = path_distance_bu + source_distance_bu
        angular_sample_weight = 4.0 * pi / float(self.config.ray_count)
        energy = (
            throughput
            * visibility
            * brdf
            * cosine_incident
            * angular_sample_weight
            / (source_distance_m * source_distance_m)
            * self._air_energy(total_distance_bu)
        )
        if not np.any(energy > 1e-16):
            return None
        self.stats.source_connections += 1
        return AcousticEvent(
            delay_seconds=total_distance_bu / self.config.speed_of_sound_bu,
            arrival_direction=first_direction.copy(),
            energy_bands=energy.astype(np.float32),
            kind='DIFFUSE',
            order=bounce + 1,
        )

    def trace(
        self,
        source: mathutils.Vector,
        receiver: mathutils.Vector,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[AcousticEvent]:
        events: List[AcousticEvent] = []
        bvh = self.scene.bvh
        if bvh is None or self.config.max_bounces <= 0:
            return events

        for ray_index, initial_direction in enumerate(self._directions()):
            if progress and ray_index % 64 == 0:
                progress(ray_index, self.config.ray_count)
            self.stats.rays_traced += 1
            first_direction = initial_direction.copy()
            direction = initial_direction.copy()
            position = receiver.copy()
            throughput = np.ones(NUM_BANDS, dtype=np.float64)
            path_distance_bu = 0.0
            all_prior_specular = True

            for bounce in range(self.config.max_bounces):
                hit, normal, face_index, hit_distance = bvh.ray_cast(
                    position + direction * self.config.eps, direction
                )
                if hit is None or normal is None or face_index is None:
                    break
                if not (0 <= face_index < len(self.scene.faces)):
                    break

                hit_point = mathutils.Vector(hit)
                normal = mathutils.Vector(normal).normalized()
                if normal.dot(direction) > 0.0:
                    normal = -normal
                path_distance_bu += float(hit_distance)
                if path_distance_bu / self.config.speed_of_sound_bu >= self.config.duration_seconds:
                    break

                self.stats.surface_interactions += 1
                material = self.material_for_face(face_index)
                event = self._source_connection(
                    hit_point,
                    normal,
                    direction,
                    source,
                    path_distance_bu,
                    throughput,
                    material,
                    first_direction,
                    bounce,
                    all_prior_specular,
                )
                if event is not None and event.delay_seconds < self.config.duration_seconds:
                    events.append(event)

                new_direction, new_throughput, interaction = self._sample_surface(
                    direction, normal, material, throughput
                )
                if new_direction is None or new_throughput is None:
                    break
                all_prior_specular = (
                    all_prior_specular and interaction == 'SPECULAR'
                )
                throughput = new_throughput
                if float(np.max(throughput)) < self.config.min_energy:
                    break

                if self.config.rr_enabled and bounce + 1 >= self.config.rr_start:
                    if float(self.rng.random()) > self.config.rr_survival:
                        break
                    throughput /= self.config.rr_survival

                direction = new_direction.normalized()
                if interaction == 'TRANSMISSION':
                    position = hit_point + direction * (self.config.eps * 4.0)
                else:
                    position = (
                        hit_point
                        + normal * (self.config.eps * 2.0)
                        + direction * (self.config.eps * 2.0)
                    )

        if progress:
            progress(self.config.ray_count, self.config.ray_count)
        return events


class AmbisonicIREngine:
    """Orchestrate deterministic paths, energy transport, and HOA synthesis."""

    def __init__(
        self,
        context,
        config: Optional[AcousticRenderConfig] = None,
        acoustic_scene: Optional[AcousticScene] = None,
    ):
        self.context = context
        self.config = config or AcousticRenderConfig.from_context(context)
        self.scene = acoustic_scene or build_acoustic_scene(context)
        self.tracer = ReceiverPathTracer(self.config, self.scene)

    def _air_energy(self, distance_bu: float) -> np.ndarray:
        return self.tracer._air_energy(distance_bu)

    def _direct_or_diffraction_events(
        self, source: mathutils.Vector, receiver: mathutils.Vector
    ) -> List[AcousticEvent]:
        direction = source - receiver
        distance_bu = direction.length
        if distance_bu <= self.config.eps:
            return []
        visibility = spectral_visibility(source, receiver, self.scene, self.config.eps)
        events: List[AcousticEvent] = []

        if np.any(visibility > 1e-10):
            if self.config.output_content == 'FULL':
                distance_m = distance_bu * self.config.unit_scale
                energy = (
                    visibility
                    * self._air_energy(distance_bu)
                    / max(distance_m * distance_m, 1e-12)
                )
                events.append(AcousticEvent(
                    delay_seconds=distance_bu / self.config.speed_of_sound_bu,
                    arrival_direction=direction.normalized(),
                    energy_bands=energy.astype(np.float32),
                    kind='DIRECT',
                    order=0,
                ))
                self.tracer.stats.direct_events += 1
            return events

        if (
            not self.config.diffraction_enabled
            or self.config.output_content == 'DIFFUSE'
            or self.config.diffraction_paths <= 0
        ):
            return events
        try:
            edge_index = build_diffraction_edge_index(self.context)
            paths = find_diffraction_paths(
                source,
                receiver,
                edge_index,
                self.scene.bvh,
                self.config.unit_scale,
                self.config.diffraction_max_angle_rad,
                self.config.diffraction_paths,
                self.config.eps,
            )
        except Exception as error:
            print(f"Diffraction disabled for this render: {error}")
            return events

        for path in paths:
            pressure_gain = maekawa_diffraction_gains(
                path.path_difference_m,
                self.config.speed_of_sound_bu * self.config.unit_scale,
                path.bend_angle_rad,
                self.config.diffraction_max_angle_rad,
            )
            distance_m = path.distance_bu * self.config.unit_scale
            energy = (
                pressure_gain * pressure_gain
                * self._air_energy(path.distance_bu)
                / max(distance_m * distance_m * max(len(paths), 1), 1e-12)
            )
            events.append(AcousticEvent(
                delay_seconds=path.distance_bu / self.config.speed_of_sound_bu,
                arrival_direction=(path.point - receiver).normalized(),
                energy_bands=energy.astype(np.float32),
                kind='EARLY',
                order=0,
            ))
            self.tracer.stats.diffraction_events += 1
        return events

    def _reconstruct_specular_path(
        self,
        source: mathutils.Vector,
        receiver: mathutils.Vector,
        surfaces: Sequence[_SpecularSurface],
        sequence: Sequence[int],
    ) -> Optional[
        Tuple[List[mathutils.Vector], List[int], float, np.ndarray]
    ]:
        """Reconstruct and validate one finite image-source path."""
        images = [source.copy()]
        for surface_index in sequence:
            surface = surfaces[surface_index]
            image = images[-1]
            signed_distance = float(
                (image - surface.plane_point).dot(surface.normal)
            )
            images.append(image - surface.normal * (2.0 * signed_distance))

        image_distance = (images[-1] - receiver).length
        if (
            image_distance <= self.config.eps
            or image_distance / self.config.speed_of_sound_bu
            >= self.config.duration_seconds
        ):
            return None

        reflection_points = [None] * len(sequence)
        reflection_faces = [None] * len(sequence)
        endpoint = receiver.copy()
        for path_index in range(len(sequence) - 1, -1, -1):
            surface = surfaces[sequence[path_index]]
            image_ray = images[path_index + 1] - endpoint
            denominator = float(surface.normal.dot(image_ray))
            if abs(denominator) <= 1e-12:
                return None
            fraction = float(
                surface.normal.dot(surface.plane_point - endpoint)
            ) / denominator
            if fraction <= 1e-7 or fraction >= 1.0 - 1e-7:
                return None
            reflection_point = endpoint + image_ray * fraction
            face_index = surface.face_at(reflection_point, self.scene)
            if face_index is None:
                return None
            reflection_points[path_index] = reflection_point
            reflection_faces[path_index] = face_index
            endpoint = reflection_point

        path_points = [source] + reflection_points + [receiver]
        total_distance_bu = 0.0
        visibility = np.ones(NUM_BANDS, dtype=np.float64)
        for segment_start, segment_end in zip(path_points, path_points[1:]):
            segment_distance = (segment_end - segment_start).length
            if segment_distance <= self.config.eps * 4.0:
                return None
            total_distance_bu += segment_distance
            visibility *= spectral_visibility(
                segment_start,
                segment_end,
                self.scene,
                self.config.eps,
            )
            if not np.any(visibility > 1e-10):
                return None

        for path_index, reflection_point in enumerate(reflection_points):
            surface = surfaces[sequence[path_index]]
            incoming_side = float(
                (path_points[path_index] - reflection_point).dot(surface.normal)
            )
            outgoing_side = float(
                (path_points[path_index + 2] - reflection_point).dot(surface.normal)
            )
            if incoming_side * outgoing_side <= self.config.eps ** 2:
                return None
            incoming = (reflection_point - path_points[path_index]).normalized()
            outgoing = (path_points[path_index + 2] - reflection_point).normalized()
            if reflect(incoming, surface.normal).dot(outgoing) < 1.0 - 1e-5:
                return None

        if (
            total_distance_bu <= self.config.eps
            or total_distance_bu / self.config.speed_of_sound_bu
            >= self.config.duration_seconds
        ):
            return None
        return reflection_points, reflection_faces, total_distance_bu, visibility

    def _deterministic_specular_events(
        self,
        source: mathutils.Vector,
        receiver: mathutils.Vector,
        max_order: Optional[int] = None,
    ) -> List[AcousticEvent]:
        """Resolve coherent finite-plane reflections through the selected order."""
        if (
            not self.config.early_reflections
            or self.config.output_content == 'DIFFUSE'
        ):
            return []

        requested_order = int(np.clip(
            self.config.early_order if max_order is None else max_order,
            1,
            3,
        ))
        surfaces = []
        for surface in _build_specular_surfaces(self.scene):
            has_specular_face = False
            for face_index in surface.face_indices:
                material = self.tracer.material_for_face(face_index)
                if np.any(
                    material.reflection_spectrum
                    * material.specular_fraction
                    > 1e-10
                ):
                    has_specular_face = True
                    break
            if not has_specular_face:
                continue
            surfaces.append(surface)

        candidates: List[_EarlyPathCandidate] = []
        seen = set()
        surface_count = len(surfaces)
        for order in range(1, requested_order + 1):
            sequence_count = _surface_sequence_count(surface_count, order)
            if sequence_count <= 0:
                break
            if order > 1 and sequence_count > self.config.early_path_budget:
                self.tracer.stats.early_orders_skipped = requested_order - order + 1
                print(
                    "Deterministic early reflections stopped at order "
                    f"{order - 1}: order {order} needs {sequence_count:,} "
                    f"surface sequences, above the {self.config.early_path_budget:,} "
                    "path budget"
                )
                break

            for sequence in _surface_sequences(surface_count, order):
                self.tracer.stats.early_sequences_tested += 1
                path = self._reconstruct_specular_path(
                    source, receiver, surfaces, sequence
                )
                if path is None:
                    continue
                (
                    reflection_points,
                    reflection_faces,
                    total_distance_bu,
                    visibility,
                ) = path

                key = tuple(
                    component
                    for point in reflection_points
                    for component in (
                        round(float(point.x), 5),
                        round(float(point.y), 5),
                        round(float(point.z), 5),
                    )
                ) + (round(total_distance_bu, 5),)
                if key in seen:
                    continue
                seen.add(key)

                specular_energy = np.ones(NUM_BANDS, dtype=np.float64)
                for face_index in reflection_faces:
                    material = self.tracer.material_for_face(face_index)
                    specular_energy *= (
                        material.reflection_spectrum * material.specular_fraction
                    )
                if not np.any(specular_energy > 1e-12):
                    continue

                distance_m = total_distance_bu * self.config.unit_scale
                energy = (
                    specular_energy
                    * visibility
                    * self._air_energy(total_distance_bu)
                    / max(distance_m * distance_m, 1e-12)
                )
                candidates.append(_EarlyPathCandidate(
                    event=AcousticEvent(
                        delay_seconds=(
                            total_distance_bu / self.config.speed_of_sound_bu
                        ),
                        arrival_direction=(
                            reflection_points[-1] - receiver
                        ).normalized(),
                        energy_bands=energy.astype(np.float32),
                        kind='EARLY',
                        order=order,
                    ),
                    surface_id=tuple(
                        id(surfaces[index].object_ref) for index in sequence
                    ),
                ))
            self.tracer.stats.early_highest_order = order
            self.tracer.deterministic_specular_order = order

        events = _cluster_unresolved_early_paths(candidates)
        self.tracer.stats.early_events += len(events)
        return events

    def _first_order_specular_events(
        self, source: mathutils.Vector, receiver: mathutils.Vector
    ) -> List[AcousticEvent]:
        """Compatibility helper for tests and external diagnostic scripts."""
        return self._deterministic_specular_events(source, receiver, max_order=1)

    def render(
        self,
        source: mathutils.Vector,
        receiver: mathutils.Vector,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> AcousticRenderResult:
        events = self._direct_or_diffraction_events(source, receiver)
        events.extend(self._deterministic_specular_events(source, receiver))
        events.extend(self.tracer.trace(source, receiver, progress))
        ir, synthesis_stats = synthesize_ambisonic_ir(
            events,
            self.config.sample_rate,
            self.config.duration_seconds,
            self.config.encoder,
            seed=self.config.seed,
            early_gain_db=self.config.early_gain_db,
            diffuse_gain_db=self.config.diffuse_gain_db,
        )
        return AcousticRenderResult(
            ir=ir,
            events=events,
            transport=self.tracer.stats,
            synthesis=synthesis_stats,
        )


def render_impulse_response(
    context,
    source: mathutils.Vector,
    receiver: mathutils.Vector,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AcousticRenderResult:
    """Public entry point for the unified acoustic renderer."""
    return AmbisonicIREngine(context).render(source, receiver, progress)


def trace_impulse_response(
    context,
    source: mathutils.Vector,
    receiver: mathutils.Vector,
    _bvh=None,
    _obj_map=None,
    **_unused,
) -> np.ndarray:
    """Compatibility entry point returning only the rendered IR array."""
    return render_impulse_response(context, source, receiver).ir


# Transitional alias for external scripts that constructed the old config.
RayTracingConfig = AcousticRenderConfig
