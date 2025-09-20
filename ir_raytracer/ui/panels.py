# -*- coding: utf-8 -*-
"""
UI panel for the Ambisonic IR Tracer.
"""
import bpy
from ..core.acoustics import BAND_LABELS, NUM_BANDS


def _draw_band_vector(layout, obj, prop_name: str, title: str):
    """Draw frequency band controls as a grid."""
    box = layout.box()
    box.label(text=title)
    grid = box.grid_flow(
        row_major=True, 
        columns=NUM_BANDS, 
        even_columns=True, 
        even_rows=True, 
        align=True
    )
    
    for idx, label in enumerate(BAND_LABELS):
        grid.prop(obj, prop_name, index=idx, text=label, slider=True)


class AIRT_PT_Panel(bpy.types.Panel):
    """Main panel for the Ambisonic IR Tracer."""
    bl_idname = "AIRT_PT_panel"
    bl_label = "Ambisonic IR Tracer"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        """Draw the main panel."""
        layout = self.layout
        obj = context.object
        
        # Object Properties Section
        col = layout.column(align=True)
        col.label(text="Object Properties", icon='OBJECT_DATA')
        
        if obj:
            col.prop(obj, "airt_material_preset")
            col.separator()
            
            col.prop(obj, "absorption")
            _draw_band_vector(col, obj, "absorption_bands", "Absorption Spectrum (octave centers)")
            
            col.separator()
            col.prop(obj, "scatter")
            _draw_band_vector(col, obj, "scatter_bands", "Scatter Spectrum (octave centers)")
            
            col.separator()
            col.prop(obj, "transmission")
            
            col.separator()
            col.prop(obj, "is_acoustic_source")
            col.prop(obj, "is_acoustic_receiver")
        else:
            col.label(text="No object selected", icon='INFO')
        
        layout.separator()
        
        # Render Settings Section
        self._draw_render_settings(layout, context)
        
        # Render Button
        layout.separator()
        row = layout.row(align=True)
        row.operator("airt.render_ir", icon='RENDER_ANIMATION')
        row.operator("airt.diagnose_scene", icon='QUESTION', text="Diagnose")

    def _draw_render_settings(self, layout, context):
        """Draw render settings section."""
        col = layout.column(align=True)
        col.label(text="Render Settings", icon='SETTINGS')
        
        # Basic settings
        box = col.box()
        box.label(text="Basic Parameters")
        box.prop(context.scene, "airt_trace_mode")
        box.prop(context.scene, "airt_recv_radius")
        box.prop(context.scene, "airt_num_rays")
        box.prop(context.scene, "airt_passes")
        box.prop(context.scene, "airt_max_order")
        
        # Audio settings
        box = col.box()
        box.label(text="Audio Parameters")
        box.prop(context.scene, "airt_sr")
        box.prop(context.scene, "airt_ir_seconds")
        box.prop(context.scene, "airt_wav_subtype")
        
        # Advanced settings (collapsible)
        box = col.box()
        box.label(text="Advanced Settings")
        box.prop(context.scene, "airt_angle_tol_deg")
        box.prop(context.scene, "airt_spec_rough_deg")
        box.prop(context.scene, "airt_enable_seg_capture")
        
        # Diffraction settings
        box.prop(context.scene, "airt_enable_diffraction")
        if context.scene.airt_enable_diffraction:
            sub = box.column()
            sub.prop(context.scene, "airt_diffraction_samples")
            sub.prop(context.scene, "airt_diffraction_max_deg")
        
        # Orientation settings
        box = col.box()
        box.label(text="Orientation")
        box.prop(context.scene, "airt_yaw_offset_deg")
        box.prop(context.scene, "airt_invert_z")
        box.prop(context.scene, "airt_calibrate_direct")
        
        # Air absorption settings
        box = col.box()
        box.label(text="Air Absorption")
        box.prop(context.scene, "airt_air_enable")
        if context.scene.airt_air_enable:
            sub = box.column()
            sub.prop(context.scene, "airt_air_temp_c")
            sub.prop(context.scene, "airt_air_humidity")
            sub.prop(context.scene, "airt_air_pressure_kpa")
        
        # Performance settings
        box = col.box()
        box.label(text="Performance & Quality")
        box.prop(context.scene, "airt_quick_broadband")
        box.prop(context.scene, "airt_omit_direct")
        
        box.prop(context.scene, "airt_rr_enable")
        if context.scene.airt_rr_enable:
            sub = box.column()
            sub.prop(context.scene, "airt_rr_start")
            sub.prop(context.scene, "airt_rr_p")
        
        box.prop(context.scene, "airt_seed")


class AIRT_PT_MaterialPanel(bpy.types.Panel):
    """Separate panel for material properties with better organization."""
    bl_idname = "AIRT_PT_material_panel"
    bl_label = "Acoustic Materials"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "AIRT_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw material properties panel."""
        layout = self.layout
        obj = context.object
        
        if not obj:
            layout.label(text="No object selected", icon='INFO')
            return
        
        col = layout.column(align=True)
        
        # Material preset selector
        col.prop(obj, "airt_material_preset")
        col.separator()
        
        # Quick material controls
        row = col.row(align=True)
        row.prop(obj, "absorption", text="Absorption")
        row.prop(obj, "scatter", text="Scatter")
        row.prop(obj, "transmission", text="Transmission")
        
        col.separator()
        
        # Detailed frequency controls
        _draw_band_vector(col, obj, "absorption_bands", "Absorption Spectrum")
        _draw_band_vector(col, obj, "scatter_bands", "Scatter Spectrum")
        
        col.separator()
        
        # Object role
        row = col.row(align=True)
        row.prop(obj, "is_acoustic_source", text="Source")
        row.prop(obj, "is_acoustic_receiver", text="Receiver")


class AIRT_PT_AdvancedPanel(bpy.types.Panel):
    """Advanced settings panel."""
    bl_idname = "AIRT_PT_advanced_panel"
    bl_label = "Advanced Ray Tracing"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "AIRT_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw advanced settings panel."""
        layout = self.layout
        scene = context.scene
        
        # Performance tuning
        box = layout.box()
        box.label(text="Performance Tuning", icon='PREFERENCES')
        box.prop(scene, "airt_rr_enable")
        if scene.airt_rr_enable:
            col = box.column()
            col.prop(scene, "airt_rr_start")
            col.prop(scene, "airt_rr_p")
        
        box.prop(scene, "airt_quick_broadband")
        box.prop(scene, "airt_seed")
        
        # Physics accuracy
        box = layout.box()
        box.label(text="Physics Accuracy", icon='PHYSICS')
        box.prop(scene, "airt_enable_diffraction")
        if scene.airt_enable_diffraction:
            col = box.column()
            col.prop(scene, "airt_diffraction_samples")
            col.prop(scene, "airt_diffraction_max_deg")
        
        box.prop(scene, "airt_spec_rough_deg")
        box.prop(scene, "airt_enable_seg_capture")
        
        # Air modeling
        box = layout.box()
        box.label(text="Air Absorption Model", icon='MOD_FLUID')
        box.prop(scene, "airt_air_enable")
        if scene.airt_air_enable:
            col = box.column()
            col.prop(scene, "airt_air_temp_c")
            col.prop(scene, "airt_air_humidity")
            col.prop(scene, "airt_air_pressure_kpa")