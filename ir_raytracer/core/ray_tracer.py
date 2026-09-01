"""Receiver-centric acoustic energy transport and ambisonic IR rendering."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, cos, exp, pi, sin, sqrt
from typing import Callable, List, Optional, Tuple

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
        return cls(
            ray_count=max(1, int(getattr(scene, 'airt_num_rays', 1024))),
            max_bounces=max(0, int(getattr(scene, 'airt_max_order', 32))),
            sample_rate=max(8000, int(getattr(scene, 'airt_sr', 48000))),
            duration_seconds=max(0.1, float(getattr(scene, 'airt_ir_seconds', 2.0))),
            output_content=output_content,
            early_reflections=bool(getattr(scene, 'airt_early_reflections', True)),
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


@dataclass
class AcousticRenderResult:
    ir: np.ndarray
    events: List[AcousticEvent]
    transport: TransportStats
    synthesis: SynthesisStats


class ReceiverPathTracer:
    """Trace time-resolved acoustic energy from the listener toward the source."""

    def __init__(self, config: AcousticRenderConfig, acoustic_scene: AcousticScene):
        self.config = config
        self.scene = acoustic_scene
        self.stats = TransportStats()
        self.rng = np.random.default_rng(config.seed if config.seed != 0 else None)
        self._material_cache = {}

    def material_for_face(self, face_index: int) -> MaterialProperties:
        """Return one immutable coefficient snapshot per acoustic object."""
        obj = self.scene.faces[face_index].object_ref
        key = id(obj)
        material = self._material_cache.get(key)
        if material is None:
            material = MaterialProperties(obj)
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
    ) -> Tuple[Optional[mathutils.Vector], Optional[np.ndarray], bool]:
        transmission = material.transmission_spectrum
        diffuse = material.reflection_spectrum * material.diffuse_fraction
        specular = material.reflection_spectrum * material.specular_fraction
        components = np.stack((transmission, diffuse, specular)).astype(np.float64)
        active = np.maximum(np.asarray(throughput, dtype=np.float64), 0.0)
        active_peak = float(np.max(active))
        if active_peak <= 1e-20:
            return None, None, False

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
            return None, None, False
        probabilities = survival_probability * metrics / metric_sum
        total_probability = float(np.sum(probabilities))
        draw = float(self.rng.random())
        if draw >= total_probability:
            return None, None, False

        if draw < probabilities[0]:
            probability = max(float(probabilities[0]), 1e-12)
            return direction.normalized(), throughput * transmission / probability, True
        draw -= probabilities[0]
        if draw < probabilities[1]:
            probability = max(float(probabilities[1]), 1e-12)
            return (
                self._sample_cosine_hemisphere(normal),
                throughput * diffuse / probability,
                False,
            )
        probability = max(float(probabilities[2]), 1e-12)
        return (
            self._sample_specular_lobe(direction, normal),
            throughput * specular / probability,
            False,
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

        brdf = self._connection_brdf(
            material,
            normal,
            source_direction,
            -reverse_direction,
            include_specular=not (bounce == 0 and self.config.early_reflections),
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
                )
                if event is not None and event.delay_seconds < self.config.duration_seconds:
                    events.append(event)

                new_direction, new_throughput, transmitted = self._sample_surface(
                    direction, normal, material, throughput
                )
                if new_direction is None or new_throughput is None:
                    break
                throughput = new_throughput
                if float(np.max(throughput)) < self.config.min_energy:
                    break

                if self.config.rr_enabled and bounce + 1 >= self.config.rr_start:
                    if float(self.rng.random()) > self.config.rr_survival:
                        break
                    throughput /= self.config.rr_survival

                direction = new_direction.normalized()
                if transmitted:
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

    def _first_order_specular_events(
        self, source: mathutils.Vector, receiver: mathutils.Vector
    ) -> List[AcousticEvent]:
        if (
            not self.config.early_reflections
            or self.config.output_content == 'DIFFUSE'
        ):
            return []

        events: List[AcousticEvent] = []
        seen = set()
        for face_index, face in enumerate(self.scene.faces):
            if not face.vertices:
                continue
            plane_point = face.vertices[0]
            normal = face.normal
            source_side = float((source - plane_point).dot(normal))
            receiver_side = float((receiver - plane_point).dot(normal))
            if source_side * receiver_side <= self.config.eps * self.config.eps:
                continue

            image_source = source - normal * (2.0 * source_side)
            image_ray = image_source - receiver
            denominator = float(normal.dot(image_ray))
            if abs(denominator) <= 1e-12:
                continue
            fraction = float(normal.dot(plane_point - receiver)) / denominator
            if fraction <= self.config.eps or fraction >= 1.0 - self.config.eps:
                continue
            reflection_point = receiver + image_ray * fraction
            if not point_in_face(reflection_point, face):
                continue

            first_leg = (source - reflection_point).length
            second_leg = (receiver - reflection_point).length
            total_distance_bu = first_leg + second_leg
            if (
                total_distance_bu <= self.config.eps
                or total_distance_bu / self.config.speed_of_sound_bu
                >= self.config.duration_seconds
            ):
                continue

            visibility_source = spectral_visibility(
                source, reflection_point, self.scene, self.config.eps
            )
            visibility_receiver = spectral_visibility(
                reflection_point, receiver, self.scene, self.config.eps
            )
            visibility = visibility_source * visibility_receiver
            if not np.any(visibility > 1e-10):
                continue

            material = self.tracer.material_for_face(face_index)
            specular_energy = (
                material.reflection_spectrum * material.specular_fraction
            )
            if not np.any(specular_energy > 1e-10):
                continue

            key = (
                round(float(reflection_point.x), 5),
                round(float(reflection_point.y), 5),
                round(float(reflection_point.z), 5),
                round(total_distance_bu, 5),
            )
            if key in seen:
                continue
            seen.add(key)
            distance_m = total_distance_bu * self.config.unit_scale
            energy = (
                specular_energy
                * visibility
                * self._air_energy(total_distance_bu)
                / max(distance_m * distance_m, 1e-12)
            )
            events.append(AcousticEvent(
                delay_seconds=total_distance_bu / self.config.speed_of_sound_bu,
                arrival_direction=(reflection_point - receiver).normalized(),
                energy_bands=energy.astype(np.float32),
                kind='EARLY',
                order=1,
            ))
            self.tracer.stats.early_events += 1
        return events

    def render(
        self,
        source: mathutils.Vector,
        receiver: mathutils.Vector,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> AcousticRenderResult:
        events = self._direct_or_diffraction_events(source, receiver)
        events.extend(self._first_order_specular_events(source, receiver))
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
