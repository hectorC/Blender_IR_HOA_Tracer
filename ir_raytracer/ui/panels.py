"""Compact artist-facing UI for the ambisonic IR renderer."""
from __future__ import annotations

import bpy

from ..core.acoustics import BAND_LABELS


class _AIRTPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'IR Tracer'


def _draw_acoustic_coefficients(layout, owner):
    layout.prop(owner, "airt_material_preset", text="Preset")
    column = layout.column(align=True)
    column.prop(owner, "absorption")
    column.prop(owner, "scatter")
    column.prop(owner, "transmission")
    layout.label(
        text="Scattering softens echoes using detail outside the mesh",
        icon='INFO',
    )
    layout.prop(owner, "show_frequency_details", toggle=True)
    if owner.show_frequency_details:
        box = layout.box()
        box.label(text="Tone by frequency: Dampen / Spread / Leak")
        for index, label in enumerate(BAND_LABELS):
            row = box.row(align=True)
            row.label(text=label)
            row.prop(owner, "absorption_bands", index=index, text="D")
            row.prop(owner, "scatter_bands", index=index, text="S")
            row.prop(owner, "transmission_bands", index=index, text="L")


class AIRT_PT_Panel(_AIRTPanel, bpy.types.Panel):
    bl_idname = "AIRT_PT_main"
    bl_label = "Ambisonic IR Tracer"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        endpoints = layout.box()
        endpoints.label(text="Virtual Source and Listener", icon='OUTLINER_OB_SPEAKER')
        endpoints.prop(scene, "airt_source_object")
        endpoints.operator("airt.assign_source", icon='EYEDROPPER')
        endpoints.prop(scene, "airt_receiver_object")
        endpoints.operator("airt.assign_receiver", icon='EYEDROPPER')

        render = layout.box()
        render.prop(scene, "airt_quality_preset", text="Quality")
        row = render.row(align=True)
        row.prop(scene, "airt_output_content", text="Content")
        render.operator("airt.render_ir", icon='SOUND')
        if not bpy.ops.airt.render_ir.poll():
            render.label(text="Rendering acoustic IR…", icon='TIME')
        elif scene.airt_last_render_summary:
            render.label(text=scene.airt_last_render_summary, icon='CHECKMARK')


class AIRT_PT_MaterialPanel(_AIRTPanel, bpy.types.Panel):
    bl_idname = "AIRT_PT_material"
    bl_label = "Acoustic Material"
    bl_parent_id = "AIRT_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        material = obj.active_material
        if material is not None:
            assignment = layout.box()
            assignment.label(
                text=f"Active Slot: {material.name}",
                icon='MATERIAL',
            )
            assignment.prop(material, "airt_acoustic_enabled")
            if material.airt_acoustic_enabled:
                owner = material
                layout.label(text="Editing Blender material", icon='MATERIAL')
            else:
                owner = obj
                layout.label(
                    text="Material acoustics disabled — editing object fallback",
                    icon='INFO',
                )
        else:
            owner = obj
            layout.label(text=f"Object Fallback: {obj.name}", icon='MESH_DATA')

        _draw_acoustic_coefficients(layout, owner)

        row = layout.row(align=True)
        row.operator("airt.reset_material", text="Reset")
        row.operator("airt.copy_material", text="Copy Settings")


class AIRT_PT_AudioPanel(_AIRTPanel, bpy.types.Panel):
    bl_idname = "AIRT_PT_audio"
    bl_label = "IR and Output"
    bl_parent_id = "AIRT_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        timing = layout.column(align=True)
        timing.prop(scene, "airt_sr")
        timing.prop(scene, "airt_ir_seconds")
        timing.prop(scene, "airt_output_content")

        early = layout.box()
        early.enabled = scene.airt_output_content != 'DIFFUSE'
        early.prop(scene, "airt_early_reflections")
        early_settings = early.column(align=True)
        early_settings.enabled = scene.airt_early_reflections
        early_settings.prop(scene, "airt_early_order")
        early_settings.prop(scene, "airt_early_gain_db")
        layout.prop(scene, "airt_diffuse_gain_db")

        output = layout.box()
        output.prop(scene, "airt_output_path")
        output.prop(scene, "airt_wav_subtype")
        output.prop(scene, "airt_normalization")
        if scene.airt_normalization == 'PEAK':
            output.prop(scene, "airt_peak_db")
            output.label(text="Source distance will no longer set loudness", icon='ERROR')
        output.label(text="16 channels — ACN/SN3D (AmbiX)", icon='INFO')


class AIRT_PT_AdvancedPanel(_AIRTPanel, bpy.types.Panel):
    bl_idname = "AIRT_PT_advanced"
    bl_label = "Transport and Space"
    bl_parent_id = "AIRT_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        sampling = layout.box()
        sampling.label(text="Reverberant Texture")
        sampling.prop(scene, "airt_num_rays")
        sampling.prop(scene, "airt_max_order")
        sampling.prop(scene, "airt_seed")
        sampling.prop(scene, "airt_spec_rough_deg")
        sampling.prop(scene, "airt_min_throughput")
        sampling.prop(scene, "airt_rr_enable")
        if scene.airt_rr_enable:
            row = sampling.row(align=True)
            row.prop(scene, "airt_rr_start")
            row.prop(scene, "airt_rr_p")

        if scene.airt_early_reflections:
            deterministic = layout.box()
            deterministic.label(text="Distinct Planar Echo Search")
            deterministic.prop(scene, "airt_early_path_budget")
            deterministic.label(
                text="Complex geometry can make higher echo orders expensive",
                icon='INFO',
            )

        air = layout.box()
        air.prop(scene, "airt_air_enable")
        if scene.airt_air_enable:
            air.prop(scene, "airt_air_temp_c")
            air.prop(scene, "airt_air_humidity")
            air.prop(scene, "airt_air_pressure_kpa")

        diffraction = layout.box()
        diffraction.prop(scene, "airt_enable_diffraction")
        if scene.airt_enable_diffraction:
            diffraction.prop(scene, "airt_diffraction_samples")
            diffraction.prop(scene, "airt_diffraction_max_deg")
            diffraction.label(text="Softens acoustic shadows behind one edge", icon='INFO')

        orientation = layout.box()
        orientation.label(text="Ambisonic Orientation")
        orientation.prop(scene, "airt_use_receiver_orientation")
        orientation.prop(scene, "airt_yaw_offset_deg")
        orientation.prop(scene, "airt_invert_z")


class AIRT_PT_DiagnosticsPanel(_AIRTPanel, bpy.types.Panel):
    bl_idname = "AIRT_PT_diagnostics"
    bl_label = "Diagnostics"
    bl_parent_id = "AIRT_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, _context):
        layout = self.layout
        layout.operator("airt.validate_scene", icon='CHECKMARK')
        layout.operator("airt.check_dependencies", icon='FILE_REFRESH')
