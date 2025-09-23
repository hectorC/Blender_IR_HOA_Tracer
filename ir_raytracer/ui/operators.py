# -*- coding: utf-8 -*-
"""
Operators for the Ambisonic IR Tracer.
"""
import bpy
import sys
import random
import numpy as np
import importlib

def check_soundfile_availability():
    """Check if soundfile is available and attempt to import it."""
    try:
        # Force reload of soundfile module to avoid cache issues
        if 'soundfile' in sys.modules:
            importlib.reload(sys.modules['soundfile'])
        
        import soundfile as sf
        return True, sf
    except ImportError as e:
        return False, str(e)

# Check for required dependencies with better error handling
HAVE_SF, SF_ERROR = check_soundfile_availability()
if HAVE_SF:
    import soundfile as sf
else:
    sf = None

try:
    from ..core.ray_tracer import trace_impulse_response
    from ..utils.scene_utils import build_bvh, get_scene_sources, get_scene_receivers, get_writable_path
except ImportError:
    # Fallback for development/testing
    trace_impulse_response = None
    build_bvh = get_scene_sources = get_scene_receivers = get_writable_path = None


def calibrate_direct_1_over_r(ir: np.ndarray, context, source, receiver):
    """Scale the entire IR so that the direct-path amplitude in W matches 1/dist."""
    try:
        from ..utils.scene_utils import speed_of_sound_bu
        
        sr = int(context.scene.airt_sr)
        c = speed_of_sound_bu(context)
        dist = (receiver - source).length
        
        if dist <= 1e-9 or ir.shape[1] <= 22:
            return ir, "Calibrate: skipped (zero distance or IR too short)"
        
        delay = (dist / c) * sr
        n = int(round(delay))
        n0 = max(0, n - 10)
        n1 = min(ir.shape[1], n + 11)
        
        a_meas = float(np.max(np.abs(ir[0, n0:n1])))
        if a_meas <= 1e-9:
            # Direct path blocked - try early reflection calibration
            # Look for strongest reflection in first 200ms
            early_limit_ms = 200.0  # 200ms window for early reflections
            early_samples = int((early_limit_ms / 1000.0) * sr)
            early_samples = min(early_samples, ir.shape[1])
            
            # Find the strongest peak in early window
            early_peak = float(np.max(np.abs(ir[0, :early_samples])))
            
            if early_peak > 1e-9:
                # Find the time of the strongest peak
                peak_idx = int(np.argmax(np.abs(ir[0, :early_samples])))
                peak_time_s = peak_idx / sr
                
                # Estimate reflection path distance (approximate)
                # For wall occlusion: source -> wall -> receiver is roughly 1.5-2x direct distance
                estimated_reflection_dist = dist * 1.7  # Conservative estimate
                
                # Use reflection-based calibration
                a_exp = 1.0 / max(estimated_reflection_dist, 1e-9)
                k = a_exp / early_peak
                ir *= k
                
                return ir, f"Calibrate: blocked direct, using reflection at {peak_time_s*1000:.1f}ms, est_dist={estimated_reflection_dist:.2f}m, k={k:.4f}"
            else:
                return ir, f"Calibrate: skipped (no direct energy near n~{n}, no early reflections found)"
        
        a_exp = 1.0 / max(dist, 1e-9)
        k = a_exp / a_meas
        ir *= k
        
        return ir, f"Calibrate: dist={dist:.3f}m, expW={a_exp:.6f}, measW={a_meas:.6f}, k={k:.4f}, n~{n}, window=+/-10"
    
    except Exception as e:
        return ir, f"Calibrate: skipped (err: {e})"


class AIRT_OT_RenderIR(bpy.types.Operator):
    """Render ambisonic impulse response operator."""
    bl_idname = "airt.render_ir"
    bl_label = "Render Ambisonic IR"
    bl_description = "Render ambisonic impulse response using ray tracing"

    def execute(self, context):
        """Execute the render operation."""
        # Check dependencies with runtime re-check
        sf_available, sf_error = check_soundfile_availability()
        if not sf_available:
            cmd = f"{sys.executable} -m pip install soundfile"
            error_msg = f"python-soundfile is required but not available.\nError: {sf_error}\nInstall with: {cmd}"
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
        
        # Import soundfile after confirming it's available
        import soundfile as sf
        
        from ..core.ambisonic import HAVE_SCIPY
        if not HAVE_SCIPY:
            cmd = f"{sys.executable} -m pip install scipy"
            self.report({'ERROR'}, "scipy is required for SH encoding. Install with:\n" + cmd)
            return {'CANCELLED'}
        
        # Validate scene setup
        scene = context.scene
        sources = get_scene_sources(context)
        receivers = get_scene_receivers(context)
        
        if not sources or not receivers:
            self.report({'WARNING'}, "Need at least one source and one receiver object")
            return {'CANCELLED'}
        
        # Use first source and receiver
        source = sources[0]
        receiver = receivers[0]
        
        # Build BVH for scene geometry
        self.report({'INFO'}, "Building BVH tree...")
        bvh, obj_map = build_bvh(context)
        
        if bvh is None:
            self.report({'WARNING'}, "No geometry found for ray tracing")
            return {'CANCELLED'}
        
        # Perform multiple passes with averaging
        passes = max(1, int(scene.airt_passes))
        ir = None
        
        self.report({'INFO'}, f"Starting {passes} render pass(es)...")
        
        # Accumulate results with better energy handling
        for pass_idx in range(passes):
            # Set random seed for reproducible results
            if scene.airt_seed:
                seed = int(scene.airt_seed) + pass_idx
                random.seed(seed)
                np.random.seed(seed)
            
            # Trace impulse response for this pass
            self.report({'INFO'}, f"Tracing pass {pass_idx + 1}/{passes}...")
            ir_pass = trace_impulse_response(context, source, receiver, bvh, obj_map)
            
            
            # Accumulate results with float64 precision to avoid rounding errors
            if ir is None:
                ir = ir_pass.astype(np.float64)  # Use higher precision during accumulation
            else:
                ir += ir_pass.astype(np.float64)
        
        # Average across passes and convert back to float32
        ir = (ir / float(passes)).astype(np.float32)
        
        # Optional direct path calibration - BUT NOT if direct path is omitted!
        should_calibrate = bool(getattr(scene, 'airt_calibrate_direct', False))
        
        if should_calibrate:
            ir, cal_info = calibrate_direct_1_over_r(ir, context, source, receiver)
            self.report({'INFO'}, cal_info)
        
        
        # Write output file
        try:
            sr = scene.airt_sr
            subtype = scene.airt_wav_subtype
            wav_path = get_writable_path("ir_output.wav")
            
            # Transpose for soundfile (expects channels x samples -> samples x channels)
            sf.write(wav_path, ir.T.astype(np.float32), samplerate=sr, subtype=subtype)
            
            self.report({'INFO'}, f"IR saved to {wav_path} ({subtype}, {ir.shape[0]} channels, {ir.shape[1]} samples)")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write WAV file: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class AIRT_OT_ValidateScene(bpy.types.Operator):
    """Validate scene setup for ray tracing."""
    bl_idname = "airt.validate_scene"
    bl_label = "Validate Scene"
    bl_description = "Check scene setup for common issues"

    def execute(self, context):
        """Execute scene validation."""
        issues = []
        warnings = []
        
        # Check for sources and receivers
        sources = get_scene_sources(context)
        receivers = get_scene_receivers(context)
        
        if not sources:
            issues.append("No acoustic source objects found")
        elif len(sources) > 1:
            warnings.append(f"Multiple sources found ({len(sources)}), using first one")
        
        if not receivers:
            issues.append("No acoustic receiver objects found")
        elif len(receivers) > 1:
            warnings.append(f"Multiple receivers found ({len(receivers)}), using first one")
        
        # Check for geometry
        bvh, obj_map = build_bvh(context)
        if bvh is None:
            issues.append("No geometry found for ray tracing")
        else:
            geo_count = len(obj_map)
            if geo_count < 4:
                warnings.append(f"Very few surfaces found ({geo_count}), consider adding more geometry")
        
        # Check render settings
        scene = context.scene
        if scene.airt_num_rays < 1000:
            warnings.append("Low ray count may produce noisy results")
        
        if scene.airt_max_order > 100:
            warnings.append("Very high bounce count may slow rendering significantly")
        
        # Report results
        if issues:
            self.report({'ERROR'}, "; ".join(issues))
            return {'CANCELLED'}
        elif warnings:
            self.report({'WARNING'}, "; ".join(warnings))
        else:
            self.report({'INFO'}, "Scene validation passed")
        
        return {'FINISHED'}


class AIRT_OT_ResetMaterial(bpy.types.Operator):
    """Reset material properties to defaults."""
    bl_idname = "airt.reset_material"
    bl_label = "Reset Material"
    bl_description = "Reset selected object's acoustic material to defaults"

    def execute(self, context):
        """Execute material reset."""
        obj = context.object
        if not obj:
            self.report({'WARNING'}, "No object selected")
            return {'CANCELLED'}
        
        # Reset to defaults
        obj.absorption = 0.2
        obj.scatter = 0.35
        obj.transmission = 0.0
        obj.airt_material_preset = 'CUSTOM'
        
        # Reset frequency bands
        default_abs = [0.2] * 7
        default_scat = [0.35] * 7
        obj.absorption_bands = default_abs
        obj.scatter_bands = default_scat
        
        self.report({'INFO'}, f"Reset material properties for {obj.name}")
        return {'FINISHED'}


class AIRT_OT_CopyMaterial(bpy.types.Operator):
    """Copy material properties between objects."""
    bl_idname = "airt.copy_material"
    bl_label = "Copy Material"
    bl_description = "Copy acoustic material from active to selected objects"

    @classmethod
    def poll(cls, context):
        """Check if operator can run."""
        return context.object is not None and len(context.selected_objects) > 1

    def execute(self, context):
        """Execute material copying."""
        source_obj = context.object
        target_objects = [obj for obj in context.selected_objects if obj != source_obj]
        
        if not target_objects:
            self.report({'WARNING'}, "Need to select target objects")
            return {'CANCELLED'}
        
        # Copy properties with error handling
        copied_count = 0
        for target_obj in target_objects:
            try:
                # Ensure target object has acoustic properties (they should be registered on all objects)
                target_obj.absorption = source_obj.absorption
                target_obj.scatter = source_obj.scatter
                target_obj.transmission = source_obj.transmission
                target_obj.airt_material_preset = source_obj.airt_material_preset
                
                # Copy frequency bands - ensure they exist and have correct length
                if hasattr(source_obj, 'absorption_bands') and hasattr(target_obj, 'absorption_bands'):
                    target_obj.absorption_bands = source_obj.absorption_bands[:]
                if hasattr(source_obj, 'scatter_bands') and hasattr(target_obj, 'scatter_bands'):
                    target_obj.scatter_bands = source_obj.scatter_bands[:]
                
                copied_count += 1
            except AttributeError as e:
                self.report({'WARNING'}, f"Failed to copy material to {target_obj.name}: {str(e)}")
        
        if copied_count > 0:
            self.report({'INFO'}, f"Copied material from {source_obj.name} to {copied_count} object(s)")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to copy material to any objects")
            return {'CANCELLED'}


class AIRT_OT_DiagnoseScene(bpy.types.Operator):
    """Diagnose potential reverb tail issues."""
    bl_idname = "airt.diagnose_scene"
    bl_label = "Diagnose Reverb Tail"
    bl_description = "Analyze scene for issues that could cause short reverb tails"

    def execute(self, context):
        """Execute scene diagnosis."""
        scene = context.scene
        issues = []
        suggestions = []
        
        # Check Russian Roulette settings
        if scene.airt_rr_enable:
            if scene.airt_rr_start < 15:
                issues.append(f"Russian Roulette starts too early (bounce {scene.airt_rr_start})")
                suggestions.append("Increase RR start bounce to 20+ for longer reverb")
            
            if scene.airt_rr_p < 0.93:
                issues.append(f"Russian Roulette survival too low ({scene.airt_rr_p:.2f})")
                suggestions.append("Increase survival probability to 0.95+ for longer tails")
        
        # Check throughput threshold
        if hasattr(scene, 'airt_min_throughput'):
            threshold = scene.airt_min_throughput
            if threshold > 5e-6:
                issues.append(f"Throughput threshold too high ({threshold:.1e})")
                suggestions.append("Reduce min throughput to 1e-6 or lower")
        
        # Check material properties
        try:
            sources = get_scene_sources(context)
            receivers = get_scene_receivers(context)
            
            if sources and receivers:
                # Check if room is too absorptive
                bvh, obj_map = build_bvh(context)
                if obj_map:
                    absorptions = []
                    for obj in obj_map:
                        if obj and hasattr(obj, 'absorption'):
                            absorptions.append(obj.absorption)
                    
                    if absorptions:
                        avg_absorption = sum(absorptions) / len(absorptions)
                        if avg_absorption > 0.3:
                            issues.append(f"Room very absorptive (avg {avg_absorption:.2f})")
                            suggestions.append("Reduce wall absorption for longer reverb")
                        elif avg_absorption < 0.05:
                            suggestions.append(f"Room reflective (avg {avg_absorption:.2f}) - good for reverb")
        
        except Exception as e:
            issues.append(f"Error analyzing geometry: {e}")
        
        # Check ray count
        if scene.airt_num_rays < 4000:
            suggestions.append(f"Low ray count ({scene.airt_num_rays}) may cause noise")
        
        # Check max bounces
        if scene.airt_max_order < 50:
            issues.append(f"Max bounces too low ({scene.airt_max_order})")
            suggestions.append("Increase max bounces to 100+ for full reverb decay")
        
        # Report results
        if issues:
            self.report({'ERROR'}, f"Found {len(issues)} issue(s): " + "; ".join(issues[:3]))
            for suggestion in suggestions[:3]:
                self.report({'INFO'}, f"Suggestion: {suggestion}")
        else:
            self.report({'INFO'}, "No obvious issues found with reverb settings")
            if suggestions:
                for suggestion in suggestions[:2]:
                    self.report({'INFO'}, f"Tip: {suggestion}")
        
        return {'FINISHED'}


class AIRT_OT_CheckDependencies(bpy.types.Operator):
    """Check and report addon dependencies status."""
    bl_idname = "airt.check_dependencies"
    bl_label = "Check Dependencies"
    bl_description = "Check if required Python packages are installed"

    def execute(self, context):
        """Check dependencies and provide installation instructions."""
        import sys
        import subprocess
        
        # Check soundfile
        sf_available, sf_error = check_soundfile_availability()
        
        if sf_available:
            self.report({'INFO'}, "✓ soundfile is available")
        else:
            self.report({'ERROR'}, f"✗ soundfile not available: {sf_error}")
            
            # Provide installation commands
            blender_python = sys.executable
            pip_cmd = f'"{blender_python}" -m pip install soundfile'
            
            self.report({'INFO'}, f"Install command: {pip_cmd}")
            self.report({'INFO'}, "Or restart Blender after installing soundfile")
        
        # Check scipy
        try:
            import scipy
            self.report({'INFO'}, "✓ scipy is available")
        except ImportError:
            self.report({'WARNING'}, "✗ scipy not available (required for spherical harmonics)")
            pip_cmd = f'"{sys.executable}" -m pip install scipy'
            self.report({'INFO'}, f"Install command: {pip_cmd}")
        
        # Check numpy
        try:
            import numpy
            self.report({'INFO'}, f"✓ numpy is available (version {numpy.__version__})")
        except ImportError:
            self.report({'ERROR'}, "✗ numpy not available (critical dependency)")
        
        return {'FINISHED'}


class AIRT_OT_HybridPreset(bpy.types.Operator):
    """Apply hybrid balance presets for common acoustic scenarios."""
    bl_idname = "airt.hybrid_preset"
    bl_label = "Apply Hybrid Preset"
    bl_description = "Apply preset hybrid balance settings for specific acoustic scenarios"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Properties for the preset values
    forward_gain: bpy.props.FloatProperty(name="Forward Gain (dB)", default=0.0, min=-24.0, max=24.0)
    reverse_gain: bpy.props.FloatProperty(name="Reverse Gain (dB)", default=0.0, min=-24.0, max=24.0)
    ramp_time: bpy.props.FloatProperty(name="Ramp Time (s)", default=0.2)
    
    def execute(self, context):
        """Apply the preset values to the scene."""
        scene = context.scene
        
        # Apply the preset values
        scene.airt_hybrid_forward_gain_db = self.forward_gain
        scene.airt_hybrid_reverse_gain_db = self.reverse_gain  
        scene.airt_hybrid_reverb_ramp_time = self.ramp_time
        
        # Report what was applied
        if self.forward_gain == 0.0 and self.reverse_gain == 0.0 and self.ramp_time == 0.2:
            preset_name = "Reset to defaults"
        elif self.forward_gain > 0 and self.reverse_gain < 0:
            preset_name = "Tunnel/Corridor (enhanced echoes)"
        elif self.forward_gain < 0 and self.reverse_gain > 0:
            preset_name = "Cathedral (lush reverb)"
        else:
            preset_name = "Custom settings"
            
        self.report({'INFO'}, f"Applied hybrid preset: {preset_name}")
        
        return {'FINISHED'}