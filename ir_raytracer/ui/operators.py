"""Blender operators for rendering and configuring ambisonic IRs."""
from __future__ import annotations

import json
import os
import sys

import bpy
import numpy as np

from ..core.acoustics import BAND_CENTERS_HZ, NUM_BANDS
from ..core.ambisonic import get_ambi_channel_names
from ..core.ray_tracer import AcousticRenderConfig, AmbisonicIREngine
from ..utils.scene_utils import (
    build_acoustic_scene,
    get_scene_receiver_objects,
    get_scene_source_objects,
    get_writable_path,
    object_world_position,
)


def check_soundfile_availability():
    try:
        import soundfile as soundfile
        return True, soundfile
    except ImportError as error:
        return False, str(error)


def prepare_ir_for_export(
    ir: np.ndarray, normalization: str, peak_db: float
) -> tuple[np.ndarray, float]:
    """Return an export copy and the applied linear gain."""
    output = np.asarray(ir, dtype=np.float32).copy()
    if normalization != 'PEAK':
        return output, 1.0
    current_peak = float(np.max(np.abs(output))) if output.size else 0.0
    if current_peak <= 1e-12:
        return output, 1.0
    target_peak = 10.0 ** (float(peak_db) / 20.0)
    gain = target_peak / current_peak
    output *= gain
    return output, gain


def _selected_source(context):
    scene = context.scene
    selected = getattr(scene, 'airt_source_object', None)
    if selected is not None:
        return selected
    tagged = get_scene_source_objects(context)
    return tagged[0] if tagged else None


def _selected_receiver(context):
    scene = context.scene
    selected = getattr(scene, 'airt_receiver_object', None)
    if selected is not None:
        return selected
    tagged = get_scene_receiver_objects(context)
    return tagged[0] if tagged else None


def _acoustic_assignment_metadata(acoustic_scene):
    """Describe the unique coefficient owners used by evaluated faces."""
    assignments = {}
    for face in acoustic_scene.faces:
        owner = face.acoustic_ref
        key = id(owner)
        entry = assignments.get(key)
        if entry is None:
            entry = {
                "type": (
                    "BLENDER_MATERIAL"
                    if isinstance(owner, bpy.types.Material)
                    else "OBJECT_FALLBACK"
                ),
                "name": owner.name,
                "preset": owner.airt_material_preset,
                "absorption_bands": list(owner.absorption_bands),
                "scattering_bands": list(owner.scatter_bands),
                "transmission_bands": list(owner.transmission_bands),
                "evaluated_face_count": 0,
            }
            assignments[key] = entry
        entry["evaluated_face_count"] += 1
    return sorted(
        assignments.values(),
        key=lambda entry: (entry["type"], entry["name"]),
    )


class AIRT_OT_RenderIR(bpy.types.Operator):
    """Render a third-order ACN/SN3D impulse response."""

    bl_idname = "airt.render_ir"
    bl_label = "Render Ambisonic IR"
    bl_description = "Render direct, early, and diffuse acoustic energy into a 16-channel HOA WAV"
    bl_options = {'REGISTER'}

    def execute(self, context):
        available, soundfile_or_error = check_soundfile_availability()
        if not available:
            command = f"{sys.executable} -m pip install soundfile"
            self.report({'ERROR'}, f"python-soundfile is required: {soundfile_or_error}. Install with {command}")
            return {'CANCELLED'}
        soundfile = soundfile_or_error

        source_object = _selected_source(context)
        receiver_object = _selected_receiver(context)
        if source_object is None or receiver_object is None:
            self.report({'ERROR'}, "Choose one Source and one Receiver")
            return {'CANCELLED'}
        if source_object == receiver_object:
            self.report({'ERROR'}, "Source and Receiver must be different objects")
            return {'CANCELLED'}

        source = object_world_position(context, source_object)
        receiver = object_world_position(context, receiver_object)
        acoustic_scene = build_acoustic_scene(context)
        if acoustic_scene.bvh is None:
            self.report({'WARNING'}, "No acoustic geometry found; rendering direct sound only")

        config = AcousticRenderConfig.from_context(context)
        window_manager = context.window_manager
        window_manager.progress_begin(0, config.ray_count)

        def update_progress(done, total):
            window_manager.progress_update(min(done, total))

        try:
            engine = AmbisonicIREngine(context, config, acoustic_scene)
            result = engine.render(source, receiver, update_progress)
        except Exception as error:
            self.report({'ERROR'}, f"Acoustic render failed: {error}")
            return {'CANCELLED'}
        finally:
            window_manager.progress_end()

        scene = context.scene
        export_ir, applied_gain = prepare_ir_for_export(
            result.ir,
            scene.airt_normalization,
            scene.airt_peak_db,
        )
        output_path = get_writable_path(scene.airt_output_path)
        try:
            soundfile.write(
                output_path,
                export_ir.T,
                int(scene.airt_sr),
                subtype=scene.airt_wav_subtype,
            )
            metadata = {
                "format": "third-order ambisonic impulse response",
                "channel_convention": "ACN/SN3D (AmbiX)",
                "channels": get_ambi_channel_names(),
                "sample_rate": int(scene.airt_sr),
                "duration_seconds": float(scene.airt_ir_seconds),
                "source": source_object.name,
                "receiver": receiver_object.name,
                "source_position_bu": list(source),
                "receiver_position_bu": list(receiver),
                "content": scene.airt_output_content,
                "quality": scene.airt_quality_preset,
                "listener_rays": config.ray_count,
                "maximum_bounces": config.max_bounces,
                "seed": config.seed,
                "scene_unit_scale_metres": config.unit_scale,
                "speed_of_sound_bu_per_second": config.speed_of_sound_bu,
                "deterministic_early_reflections": config.early_reflections,
                "deterministic_reflection_order": config.early_order,
                "deterministic_path_budget": config.early_path_budget,
                "early_gain_db": config.early_gain_db,
                "diffuse_gain_db": config.diffuse_gain_db,
                "air": {
                    "enabled": config.air_enabled,
                    "temperature_c": config.air_temperature_c,
                    "relative_humidity_percent": config.air_humidity_pct,
                    "pressure_kpa": config.air_pressure_kpa,
                },
                "diffraction": {
                    "enabled": config.diffraction_enabled,
                    "maximum_paths": config.diffraction_paths,
                },
                "orientation": {
                    "reference": (
                        "receiver_local"
                        if config.encoder.use_receiver_orientation
                        else "blender_world"
                    ),
                    "use_receiver_orientation": (
                        config.encoder.use_receiver_orientation
                    ),
                    "receiver_world_quaternion_wxyz": (
                        list(config.encoder.receiver_rotation)
                        if config.encoder.use_receiver_orientation
                        else None
                    ),
                    "yaw_degrees": config.encoder.yaw_offset_deg,
                    "invert_z": config.encoder.invert_z,
                },
                "frequency_bands_hz": list(BAND_CENTERS_HZ),
                "acoustic_assignments": _acoustic_assignment_metadata(
                    acoustic_scene
                ),
                "normalization": scene.airt_normalization,
                "applied_gain": applied_gain,
                "events": {
                    "direct": result.synthesis.direct_events,
                    "early": result.synthesis.early_events,
                    "diffuse": result.synthesis.diffuse_events,
                },
                "deterministic_path_stats": {
                    "surface_sequences_tested": result.transport.early_sequences_tested,
                    "highest_order_evaluated": result.transport.early_highest_order,
                    "orders_skipped_by_budget": result.transport.early_orders_skipped,
                    "events_by_order": {
                        str(order): sum(
                            event.kind == 'EARLY' and event.order == order
                            for event in result.events
                        )
                        for order in range(1, config.early_order + 1)
                    },
                },
            }
            with open(output_path + ".json", "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
        except Exception as error:
            self.report({'ERROR'}, f"Failed to write IR: {error}")
            return {'CANCELLED'}

        summary = (
            f"{result.synthesis.direct_events} direct, "
            f"{result.synthesis.early_events} early, "
            f"{result.synthesis.diffuse_events} diffuse events"
        )
        scene.airt_last_render_summary = summary
        if result.transport.early_orders_skipped:
            self.report({'WARNING'}, (
                "Deterministic early reflections reached the path budget; "
                f"highest completed order was {result.transport.early_highest_order}"
            ))
        self.report({'INFO'}, f"Saved {os.path.basename(output_path)} — {summary}")
        return {'FINISHED'}


class AIRT_OT_AssignSource(bpy.types.Operator):
    bl_idname = "airt.assign_source"
    bl_label = "Use Active as Source"
    bl_description = "Use the active object's evaluated world position as the source"

    def execute(self, context):
        active = context.active_object
        if active is None:
            self.report({'ERROR'}, "Select an object first")
            return {'CANCELLED'}
        for obj in context.scene.objects:
            obj.is_acoustic_source = False
        active.is_acoustic_source = True
        if active.is_acoustic_receiver:
            active.is_acoustic_receiver = False
        context.scene.airt_source_object = active
        if context.scene.airt_receiver_object == active:
            context.scene.airt_receiver_object = None
        return {'FINISHED'}


class AIRT_OT_AssignReceiver(bpy.types.Operator):
    bl_idname = "airt.assign_receiver"
    bl_label = "Use Active as Receiver"
    bl_description = "Use the active object's evaluated world position as the HOA receiver"

    def execute(self, context):
        active = context.active_object
        if active is None:
            self.report({'ERROR'}, "Select an object first")
            return {'CANCELLED'}
        for obj in context.scene.objects:
            obj.is_acoustic_receiver = False
        active.is_acoustic_receiver = True
        if active.is_acoustic_source:
            active.is_acoustic_source = False
        context.scene.airt_receiver_object = active
        if context.scene.airt_source_object == active:
            context.scene.airt_source_object = None
        return {'FINISHED'}


class AIRT_OT_ValidateScene(bpy.types.Operator):
    bl_idname = "airt.validate_scene"
    bl_label = "Validate Acoustic Scene"
    bl_description = "Check the selected endpoints, geometry, scale, and output settings"

    def execute(self, context):
        issues = []
        warnings = []
        source = _selected_source(context)
        receiver = _selected_receiver(context)
        if source is None:
            issues.append("No Source selected")
        if receiver is None:
            issues.append("No Receiver selected")
        if source is not None and source == receiver:
            issues.append("Source and Receiver are the same object")

        acoustic_scene = build_acoustic_scene(context)
        if acoustic_scene.bvh is None:
            warnings.append("No visible mesh geometry; only direct sound can be rendered")
        elif len(acoustic_scene.faces) > 100000:
            warnings.append("Very dense evaluated geometry may slow early-reflection searches")
        if context.scene.unit_settings.scale_length <= 0.0:
            issues.append("Scene Unit Scale must be positive")
        if context.scene.airt_enable_diffraction and len(acoustic_scene.faces) > 50000:
            warnings.append("Diffraction edge extraction may be expensive for this scene")

        if issues:
            self.report({'ERROR'}, "; ".join(issues))
            return {'CANCELLED'}
        message = "Scene is ready"
        if warnings:
            message += "; " + "; ".join(warnings)
        self.report({'WARNING'} if warnings else {'INFO'}, message)
        return {'FINISHED'}


class AIRT_OT_ResetMaterial(bpy.types.Operator):
    bl_idname = "airt.reset_material"
    bl_label = "Reset Acoustic Material"

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            return {'CANCELLED'}
        material = obj.active_material
        owner = (
            material
            if material is not None and material.airt_acoustic_enabled
            else obj
        )
        owner.airt_material_preset = 'CUSTOM'
        owner.absorption = 0.2
        owner.scatter = 0.35
        owner.transmission = 0.0
        owner.absorption_bands = tuple(0.2 for _ in range(NUM_BANDS))
        owner.scatter_bands = tuple(0.35 for _ in range(NUM_BANDS))
        owner.transmission_bands = tuple(0.0 for _ in range(NUM_BANDS))
        return {'FINISHED'}


class AIRT_OT_CopyMaterial(bpy.types.Operator):
    bl_idname = "airt.copy_material"
    bl_label = "Copy Acoustic Material to Selected"
    bl_description = "Copy active acoustic coefficients to matching selected assignments"

    def execute(self, context):
        source_object = context.active_object
        target_objects = [
            obj for obj in context.selected_objects
            if obj != source_object and obj.type == 'MESH'
        ]
        if source_object is None or not target_objects:
            self.report({'ERROR'}, "Select a source object and at least one target")
            return {'CANCELLED'}

        source_material = source_object.active_material
        using_material = (
            source_material is not None
            and source_material.airt_acoustic_enabled
        )
        source = source_material if using_material else source_object
        targets = []
        if using_material:
            seen = {id(source_material)}
            for obj in target_objects:
                material = obj.active_material
                if material is None or id(material) in seen:
                    continue
                material.airt_acoustic_enabled = True
                targets.append(material)
                seen.add(id(material))
        else:
            targets = target_objects

        if not targets:
            self.report({'ERROR'}, "Selected targets have no distinct active assignment")
            return {'CANCELLED'}
        for target in targets:
            target.airt_material_preset = source.airt_material_preset
            target.absorption = source.absorption
            target.scatter = source.scatter
            target.transmission = source.transmission
            target.absorption_bands = tuple(source.absorption_bands)
            target.scatter_bands = tuple(source.scatter_bands)
            target.transmission_bands = tuple(source.transmission_bands)
        assignment = "material(s)" if using_material else "object fallback(s)"
        self.report({'INFO'}, f"Copied acoustic settings to {len(targets)} {assignment}")
        return {'FINISHED'}


class AIRT_OT_CheckDependencies(bpy.types.Operator):
    bl_idname = "airt.check_dependencies"
    bl_label = "Check Audio Dependency"

    def execute(self, _context):
        available, module_or_error = check_soundfile_availability()
        if available:
            self.report({'INFO'}, f"python-soundfile {module_or_error.__version__} is available")
            return {'FINISHED'}
        self.report({'ERROR'}, f"python-soundfile is unavailable: {module_or_error}")
        return {'CANCELLED'}
