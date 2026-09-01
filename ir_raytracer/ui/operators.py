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
                    "yaw_degrees": config.encoder.yaw_offset_deg,
                    "invert_z": config.encoder.invert_z,
                },
                "frequency_bands_hz": list(BAND_CENTERS_HZ),
                "normalization": scene.airt_normalization,
                "applied_gain": applied_gain,
                "events": {
                    "direct": result.synthesis.direct_events,
                    "early": result.synthesis.early_events,
                    "diffuse": result.synthesis.diffuse_events,
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
        obj.airt_material_preset = 'CUSTOM'
        obj.absorption = 0.2
        obj.scatter = 0.35
        obj.transmission = 0.0
        obj.absorption_bands = tuple(0.2 for _ in range(NUM_BANDS))
        obj.scatter_bands = tuple(0.35 for _ in range(NUM_BANDS))
        obj.transmission_bands = tuple(0.0 for _ in range(NUM_BANDS))
        return {'FINISHED'}


class AIRT_OT_CopyMaterial(bpy.types.Operator):
    bl_idname = "airt.copy_material"
    bl_label = "Copy Acoustic Material to Selected"
    bl_description = "Copy active-object acoustic coefficients to the other selected objects"

    def execute(self, context):
        source = context.active_object
        targets = [obj for obj in context.selected_objects if obj != source]
        if source is None or not targets:
            self.report({'ERROR'}, "Select a source object and at least one target")
            return {'CANCELLED'}
        for target in targets:
            target.airt_material_preset = source.airt_material_preset
            target.absorption = source.absorption
            target.scatter = source.scatter
            target.transmission = source.transmission
            target.absorption_bands = tuple(source.absorption_bands)
            target.scatter_bands = tuple(source.scatter_bands)
            target.transmission_bands = tuple(source.transmission_bands)
        self.report({'INFO'}, f"Copied acoustic material to {len(targets)} object(s)")
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
