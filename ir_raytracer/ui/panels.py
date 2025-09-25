# -*- coding: utf-8 -*-
"""
UI panel for the Ambisonic IR Tracer.
"""
import bpy
from ..core.acoustics import BAND_LABELS, NUM_BANDS


def _avg(values):
    """Calculate average of values."""
    return float(sum(values)) / max(len(values), 1)


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
    bl_label = "IR Tracer"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        """Draw the main panel."""
        layout = self.layout
        scene = context.scene
        
        # Scene Validation Status (always visible at top)
        self._draw_scene_status(layout, context)
        
        layout.separator()
        
        # Core Ray Tracing Settings
        box = layout.box()
        box.label(text="Ray Tracing Setup", icon='LIGHT_SUN')
        
        # Tracing mode (most important setting) - prominent
        row = box.row()
        row.scale_y = 1.2  # Make it bigger
        row.prop(scene, "airt_trace_mode", text="")
        
        box.separator()
        
        # Essential parameters in logical grid layout (ALWAYS VISIBLE)
        col = box.column(align=True)
        
        # Performance vs Quality trade-off
        row = col.row(align=True)
        row.prop(scene, "airt_num_rays", text="Rays")
        row.prop(scene, "airt_passes", text="Passes")
        
        # Physics accuracy settings
        row = col.row(align=True)
        row.prop(scene, "airt_max_order", text="Max Bounces")
        row.prop(scene, "airt_recv_radius", text="Receiver (m)")
        
        box.separator()
        
        # Quick quality preset (prominent toggle)
        row = box.row()
        row.scale_y = 1.1
        icon = 'PLAY' if scene.airt_quick_broadband else 'RENDER_ANIMATION'
        row.prop(scene, "airt_quick_broadband", text="Fast Preview Mode", toggle=True, icon=icon)
        
        # HYBRID CROSSFADE CONTROLS - Show only when Hybrid mode is selected
        if scene.airt_trace_mode == 'HYBRID':
            hybrid_box = box.box()
            hybrid_box.label(text="🎛️ Hybrid Crossfade Controls", icon='PREFERENCES')
            
            # Advanced controls in a clean layout
            col = hybrid_box.column(align=True)
            
            # Forward/Reverse gain controls
            col.label(text="Gain Adjustments:")
            row = col.row(align=True)
            row.prop(scene, "airt_hybrid_forward_gain_db", text="Early")
            row.prop(scene, "airt_hybrid_reverse_gain_db", text="Reverb") 
            
            # Crossfade timing controls
            col.separator()
            col.label(text="Crossfade Timing:")
            col.prop(scene, "airt_hybrid_crossfade_start_ms", text="Start Time")
            col.prop(scene, "airt_hybrid_crossfade_length_ms", text="Fade Length")
            
            # Final level controls - both independent
            col.separator()
            col.label(text="Final Levels:")
            col.prop(scene, "airt_hybrid_forward_final_level", text="Forward Final Level")
            col.prop(scene, "airt_hybrid_reverse_final_level", text="Reverse Final Level")
            
            # Cache controls - Show re-mix button when cache is valid
            col.separator()
            
            # Import the cache validation function
            try:
                from .operators import _is_cache_valid
                cache_valid = _is_cache_valid(context)
            except ImportError:
                cache_valid = scene.airt_hybrid_cache_valid
            
            if cache_valid:
                # Cache is valid - show re-mix controls
                cache_box = col.box()
                cache_box.label(text="✓ IRs Cached - Ready for Re-mixing", icon='FILE_TICK')
                
                # Re-mix button (large and prominent)
                remix_row = cache_box.row()
                remix_row.scale_y = 1.5
                remix_row.operator("airt.remix_hybrid_ir", text="🔄 Re-mix & Export", icon='FILE_REFRESH')
                
                # Clear cache button (smaller, less prominent)
                clear_row = cache_box.row()
                clear_row.scale_y = 0.8
                clear_row.operator("airt.clear_hybrid_cache", text="Clear Cache", icon='TRASH')
                
                # Show last export path if available
                if scene.airt_hybrid_last_export_path:
                    import os
                    filename = os.path.basename(scene.airt_hybrid_last_export_path)
                    cache_box.label(text=f"Last: {filename}", icon='FILE_SOUND')
            else:
                # No cache - show status
                if scene.airt_hybrid_cache_valid:
                    col.label(text="⚠️ Cache invalid (scene changed)", icon='ERROR')
                else:
                    col.label(text="ℹ️ No cached IRs - full trace needed", icon='INFO')
        
        layout.separator()
        
        # Render Controls - Most Important Section!
        self._draw_render_controls(layout, context)
        
        layout.separator()
        
        # Current Object Quick Info (compact)
        self._draw_object_quick_info(layout, context)

    def _draw_render_controls(self, layout, context):
        """Draw render control buttons with proper hierarchy."""
        box = layout.box()
        box.label(text="Generate Impulse Response", icon='RENDER_ANIMATION')
        
        # Main render button (big and prominent)
        col = box.column()
        row = col.row()
        row.scale_y = 1.5  # Make render button bigger
        row.operator("airt.render_ir", icon='RENDER_ANIMATION', text="Generate IR")
        
        col.separator()
        
        # Secondary validation actions (smaller)
        row = col.row(align=True)
        row.operator("airt.validate_scene", icon='CHECKMARK', text="Validate Scene")
        row.operator("airt.diagnose_scene", icon='QUESTION', text="Diagnose Issues")

    def _draw_scene_status(self, layout, context):
        """Draw scene validation status."""
        try:
            from ..utils.scene_utils import get_scene_sources, get_scene_receivers
            sources = get_scene_sources(context)
            receivers = get_scene_receivers(context)
            
            box = layout.box()
            if sources and receivers:
                box.label(text=f"Scene Ready: {len(sources)} source(s), {len(receivers)} receiver(s)", icon='CHECKMARK')
            elif not sources:
                box.label(text="Missing: Mark objects as acoustic sources", icon='ERROR')
            elif not receivers:
                box.label(text="Missing: Mark objects as acoustic receivers", icon='ERROR')
            else:
                box.label(text="Scene validation error", icon='CANCEL')
        except:
            # Fallback if utils not available
            pass

    def _draw_object_quick_info(self, layout, context):
        """Draw quick info about selected object."""
        obj = context.object
        if not obj:
            return
            
        box = layout.box()
        box.label(text=f"Selected: {obj.name}", icon='OBJECT_DATA')
        
        # Object role in simulation (important for ray tracing)
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(obj, "is_acoustic_source", text="Source", toggle=True)
        row.prop(obj, "is_acoustic_receiver", text="Receiver", toggle=True)
        
        # Material preset for quick reference
        if hasattr(obj, 'airt_material_preset'):
            col.prop(obj, "airt_material_preset", text="Material Preset")

    def _draw_render_settings(self, layout, context):
        """Draw render settings section - DEPRECATED, moved to other panels."""
        pass


class AIRT_PT_AudioPanel(bpy.types.Panel):
    """Audio output settings panel."""
    bl_idname = "AIRT_PT_audio_panel"
    bl_label = "Audio Output"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "AIRT_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw audio settings panel."""
        layout = self.layout
        scene = context.scene
        
        # Audio format settings
        box = layout.box()
        box.label(text="File Output", icon='FILE_SOUND')
        
        col = box.column(align=True)
        col.prop(scene, "airt_sr", text="Sample Rate")
        col.prop(scene, "airt_ir_seconds", text="Duration (s)")
        col.prop(scene, "airt_wav_subtype", text="WAV Format")
        
        # Spatial settings
        box = layout.box()
        box.label(text="Spatial Processing", icon='ORIENTATION_GIMBAL')
        
        col = box.column(align=True)
        col.prop(scene, "airt_yaw_offset_deg", text="Yaw Offset (°)")
        col.prop(scene, "airt_invert_z", text="Invert Z-axis")
        col.prop(scene, "airt_calibrate_direct", text="Calibrate Direct Path")


class AIRT_PT_MaterialPanel(bpy.types.Panel):
    """Material properties panel - redesigned for better workflow."""
    bl_idname = "AIRT_PT_material_panel"
    bl_label = "Acoustic Materials"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "AIRT_PT_panel"

    def draw(self, context):
        """Draw material properties panel."""
        layout = self.layout
        obj = context.object
        
        if not obj:
            layout.label(text="Select an object to edit its acoustic properties", icon='INFO')
            return
        
        # Object header
        box = layout.box()
        box.label(text=f"Material: {obj.name}", icon='MATERIAL')
        
        # Material preset selector (prominent)
        box.prop(obj, "airt_material_preset", text="")
        
        # Material properties section
        col = layout.column(align=True)
        
        # Basic properties (always visible)
        box = col.box()
        box.label(text="Basic Properties", icon='MODIFIER')
        
        grid = box.grid_flow(row_major=True, columns=2, align=True)
        grid.prop(obj, "absorption", text="Absorption")
        grid.prop(obj, "scatter", text="Scatter")
        grid.prop(obj, "transmission", text="Transmission", slider=False)
        
        # Show average values for quick reference
        if hasattr(obj, 'absorption_bands') and hasattr(obj, 'scatter_bands'):
            abs_avg = _avg(obj.absorption_bands[:])
            scat_avg = _avg(obj.scatter_bands[:])
            info_text = f"Spectrum averages: Abs {abs_avg:.2f}, Scatter {scat_avg:.2f}"
            box.label(text=info_text, icon='INFO')
        
        # Frequency-dependent properties (collapsible)
        self._draw_frequency_controls(col, obj)
        
        # Material operations
        col.separator()
        row = col.row(align=True)
        row.operator("airt.copy_material", icon='COPYDOWN')
        row.operator("airt.reset_material", icon='LOOP_BACK')

    def _draw_frequency_controls(self, layout, obj):
        """Draw frequency-dependent material controls."""
        box = layout.box()
        row = box.row()
        row.prop(obj, "show_frequency_details", 
                text="Frequency Response Details", 
                icon='TRIA_DOWN' if getattr(obj, 'show_frequency_details', False) else 'TRIA_RIGHT',
                emboss=False)
        
        # Add custom property if it doesn't exist
        if not hasattr(obj, 'show_frequency_details'):
            obj['show_frequency_details'] = False
            
        if getattr(obj, 'show_frequency_details', False):
            _draw_band_vector(box, obj, "absorption_bands", "Absorption Spectrum")
            _draw_band_vector(box, obj, "scatter_bands", "Scatter Spectrum")


class AIRT_PT_AdvancedPanel(bpy.types.Panel):
    """Advanced ray tracing physics settings."""
    bl_idname = "AIRT_PT_advanced_panel"
    bl_label = "Advanced Physics"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "AIRT_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw advanced physics settings panel."""
        layout = self.layout
        scene = context.scene
        
        # Ray termination settings
        box = layout.box()
        box.label(text="Ray Termination", icon='X')
        
        box.prop(scene, "airt_rr_enable", text="Russian Roulette Termination")
        if scene.airt_rr_enable:
            col = box.column()
            col.prop(scene, "airt_rr_start", text="Start Bounce")
            col.prop(scene, "airt_rr_p", text="Survival Probability")
        
        if hasattr(scene, 'airt_min_throughput'):
            box.prop(scene, "airt_min_throughput", text="Min Energy Threshold")
        
        # Surface physics
        box = layout.box()
        box.label(text="Surface Physics", icon='MESH_ICOSPHERE')
        
        col = box.column(align=True)
        col.prop(scene, "airt_spec_rough_deg", text="Specular Roughness (°)")
        col.prop(scene, "airt_angle_tol_deg", text="Angle Tolerance (°)")
        col.prop(scene, "airt_enable_seg_capture", text="Segment Capture")
        
        # Warning for Forward mode without segment capture
        if scene.airt_trace_mode == 'FORWARD' and not scene.airt_enable_seg_capture:
            warning_box = col.box()
            warning_box.alert = True
            warning_box.label(text="⚠ Forward tracing needs Segment Capture!", icon='ERROR')
        
        # Diffraction modeling
        box.prop(scene, "airt_enable_diffraction", text="Edge Diffraction")
        if scene.airt_enable_diffraction:
            sub = box.column()
            sub.prop(scene, "airt_diffraction_samples", text="Samples")
            sub.prop(scene, "airt_diffraction_max_deg", text="Max Angle (°)")
        
        # Air absorption modeling
        box = layout.box()
        box.label(text="Air Absorption", icon='MOD_FLUID')
        
        box.prop(scene, "airt_air_enable", text="Enable Air Absorption")
        if scene.airt_air_enable:
            col = box.column()
            col.prop(scene, "airt_air_temp_c", text="Temperature (°C)")
            col.prop(scene, "airt_air_humidity", text="Humidity (%)")
            col.prop(scene, "airt_air_pressure_kpa", text="Pressure (kPa)")
        
        # Random seed
        layout.separator()
        layout.prop(scene, "airt_seed", text="Random Seed")


class AIRT_PT_DiagnosticsPanel(bpy.types.Panel):
    """Scene diagnostics and validation panel."""
    bl_idname = "AIRT_PT_diagnostics_panel"
    bl_label = "Diagnostics & Validation"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "AIRT_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw diagnostics panel."""
        layout = self.layout
        
        # Scene analysis tools
        box = layout.box()
        box.label(text="Scene Analysis", icon='VIEWZOOM')
        
        col = box.column(align=True)
        col.operator("airt.validate_scene", 
                    text="Validate Scene Setup", 
                    icon='CHECKMARK')
        col.operator("airt.diagnose_scene", 
                    text="Diagnose Reverb Issues", 
                    icon='QUESTION')
        col.operator("airt.check_dependencies",
                    text="Check Dependencies",
                    icon='PACKAGE')
        
        # Material tools
        box = layout.box()
        box.label(text="Material Tools", icon='MATERIAL')
        
        if context.object:
            col = box.column(align=True)
            col.operator("airt.copy_material", 
                        text=f"Copy from {context.object.name}", 
                        icon='COPYDOWN')
            col.operator("airt.reset_material", 
                        text=f"Reset {context.object.name}", 
                        icon='LOOP_BACK')
        else:
            box.label(text="Select object for material tools", icon='INFO')
        
        # Performance info
        if hasattr(context.scene, 'airt_num_rays'):
            box = layout.box()
            box.label(text="Performance Estimate", icon='TIME')
            
            rays = context.scene.airt_num_rays
            passes = getattr(context.scene, 'airt_passes', 1)
            bounces = getattr(context.scene, 'airt_max_order', 50)
            
            total_rays = rays * passes
            complexity = "Low" if total_rays < 5000 else "Medium" if total_rays < 20000 else "High"
            
            col = box.column()
            col.label(text=f"Total rays: {total_rays:,}")
            col.label(text=f"Max bounces: {bounces}")
            col.label(text=f"Complexity: {complexity}")
            
            if total_rays > 50000:
                col.label(text="Warning: Very high ray count", icon='ERROR')