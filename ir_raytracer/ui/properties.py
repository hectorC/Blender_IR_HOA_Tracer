# -*- coding: utf-8 -*-
"""
Property definitions and registration for the Ambisonic IR Tracer.
"""
import bpy
from ..core.acoustics import MATERIAL_PRESET_DATA, MATERIAL_PRESETS, NUM_BANDS


def _avg(values):
    """Calculate average of values."""
    return float(sum(values)) / max(len(values), 1)


def _band_label(freq_hz: float) -> str:
    """Generate human-readable label for frequency band."""
    if freq_hz >= 1000.0:
        return f"{int(freq_hz / 1000.0)} kHz"
    return f"{int(freq_hz)} Hz"


# Global state for preventing circular updates
_MATERIAL_PRESET_GUARD = set()


def _update_material_preset(self, context):
    """Update callback for material preset changes."""
    preset = getattr(self, 'airt_material_preset', 'CUSTOM')
    if preset == 'CUSTOM':
        return
    
    values = MATERIAL_PRESETS.get(preset)
    if not values:
        return
    
    key = id(self)
    _MATERIAL_PRESET_GUARD.add(key)
    try:
        self.absorption = float(values['absorption'])
        self.scatter = float(values['scatter'])
        self.absorption_bands = values['absorption_spectrum']
        self.scatter_bands = values['scatter_spectrum']
    finally:
        _MATERIAL_PRESET_GUARD.discard(key)


def _mark_material_custom(self, context):
    """Mark material as custom when properties are manually changed."""
    key = id(self)
    if key in _MATERIAL_PRESET_GUARD:
        return
    if getattr(self, 'airt_material_preset', 'CUSTOM') != 'CUSTOM':
        self.airt_material_preset = 'CUSTOM'


def create_material_preset_items():
    """Create material preset items for EnumProperty."""
    items = [('CUSTOM', 'Custom', 'User-defined absorption/scatter spectra')]
    
    for identifier, label, _, _ in MATERIAL_PRESET_DATA:
        preset = MATERIAL_PRESETS[identifier]
        desc = (
            f"{label}: avg absorption {preset['absorption']:.2f}, "
            f"avg scatter {preset['scatter']:.2f}"
        )
        items.append((identifier, label, desc))
    
    return items


def register_acoustic_props():
    """Register acoustic properties on Blender object and scene types."""
    # Object properties for acoustic materials
    bpy.types.Object.airt_material_preset = bpy.props.EnumProperty(
        name="Material Preset",
        description="Apply common material absorption/scatter values",
        items=create_material_preset_items(),
        default='CUSTOM',
        update=_update_material_preset
    )
    
    bpy.types.Object.absorption = bpy.props.FloatProperty(
        name="Absorption",
        description="Wideband absorption coefficient (0 = reflective, 1 = fully absorbent)",
        default=0.2,
        min=0.0,
        max=1.0,
        update=_mark_material_custom
    )
    
    bpy.types.Object.absorption_bands = bpy.props.FloatVectorProperty(
        name="Absorption Spectrum",
        description="Frequency-dependent absorption for octave bands (125 Hz - 8 kHz)",
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        subtype='NONE',
        default=tuple(0.2 for _ in range(NUM_BANDS)),
        update=_mark_material_custom
    )
    
    bpy.types.Object.scatter = bpy.props.FloatProperty(
        name="Scatter",
        description="Surface scattering (0 = purely specular, 1 = fully diffuse/cosine)",
        default=0.35,
        min=0.0,
        max=1.0,
        update=_mark_material_custom
    )
    
    bpy.types.Object.scatter_bands = bpy.props.FloatVectorProperty(
        name="Scatter Spectrum",
        description="Frequency-dependent scattering (fraction of energy sent to diffuse lobe)",
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        subtype='NONE',
        default=tuple(0.35 for _ in range(NUM_BANDS)),
        update=_mark_material_custom
    )
    
    bpy.types.Object.transmission = bpy.props.FloatProperty(
        name="Transmission",
        description="Portion of incident energy transmitted through the surface (0 opaque, 1 fully transparent)",
        default=0.0,
        min=0.0,
        max=1.0,
        update=_mark_material_custom
    )
    
    bpy.types.Object.is_acoustic_source = bpy.props.BoolProperty(
        name="Acoustic Source", 
        default=False
    )
    
    bpy.types.Object.is_acoustic_receiver = bpy.props.BoolProperty(
        name="Acoustic Receiver", 
        default=False
    )
    
    # Scene properties for render settings
    scene = bpy.types.Scene
    
    # Basic ray tracing parameters
    scene.airt_num_rays = bpy.props.IntProperty(
        name="Rays", 
        default=8192, 
        min=128, 
        max=131072
    )
    
    scene.airt_passes = bpy.props.IntProperty(
        name="Averaging passes", 
        default=4, 
        min=1, 
        max=32
    )
    
    scene.airt_max_order = bpy.props.IntProperty(
        name="Max Bounces", 
        default=64, 
        min=0, 
        max=1000
    )
    
    scene.airt_sr = bpy.props.IntProperty(
        name="Sample Rate", 
        default=48000, 
        min=8000, 
        max=192000
    )
    
    scene.airt_ir_seconds = bpy.props.FloatProperty(
        name="IR Length (s)", 
        default=2.0, 
        min=0.1, 
        max=20.0
    )
    
    scene.airt_angle_tol_deg = bpy.props.FloatProperty(
        name="Specular tol (deg)", 
        default=8.0, 
        min=0.1, 
        max=30.0
    )
    
    scene.airt_wav_subtype = bpy.props.EnumProperty(
        name="WAV Subtype",
        description="Audio sample format for output WAV",
        items=[
            ('FLOAT', '32-bit Float', '32-bit IEEE float (recommended for IRs)'),
            ('PCM_24', '24-bit PCM', '24-bit integer PCM'),
            ('PCM_16', '16-bit PCM', '16-bit integer PCM')
        ],
        default='FLOAT'
    )
    
    scene.airt_seed = bpy.props.IntProperty(
        name="Random Seed", 
        default=0, 
        min=0
    )
    
    scene.airt_recv_radius = bpy.props.FloatProperty(
        name="Receiver radius (m)", 
        default=0.25, 
        min=0.001, 
        max=2.0
    )
    
    scene.airt_trace_mode = bpy.props.EnumProperty(
        name="Tracing mode",
        description="Reverse = early specular; Forward = stochastic tail & early",
        items=[
            ('FORWARD','Forward (source->room)','Stochastic forward tracing with receiver capture + path connection'),
            ('REVERSE','Reverse (receiver->room)','Specular reverse tracing with LOS-to-source check')
        ],
        default='FORWARD'
    )
    
    # Russian roulette settings
    scene.airt_rr_enable = bpy.props.BoolProperty(
        name="Russian roulette", 
        default=True
    )
    
    scene.airt_rr_start = bpy.props.IntProperty(
        name="RR start bounce", 
        default=20,  # Changed from 8 to 20 - allow more bounces before RR
        min=0, 
        max=1000
    )
    
    scene.airt_rr_p = bpy.props.FloatProperty(
        name="RR survive prob", 
        default=0.95,  # Changed from 0.9 to 0.95 - higher survival rate
        min=0.05, 
        max=1.0
    )
    
    # Surface roughness
    scene.airt_spec_rough_deg = bpy.props.FloatProperty(
        name="Specular roughness (deg)", 
        default=5.0, 
        min=0.0, 
        max=30.0
    )
    
    # Advanced features
    scene.airt_enable_seg_capture = bpy.props.BoolProperty(
        name="Capture along segments", 
        default=False
    )
    
    scene.airt_enable_diffraction = bpy.props.BoolProperty(
        name="Enable diffraction", 
        default=True
    )
    
    scene.airt_diffraction_samples = bpy.props.IntProperty(
        name="Diffraction samples", 
        default=6, 
        min=0, 
        max=64
    )
    
    scene.airt_diffraction_max_deg = bpy.props.FloatProperty(
        name="Diffraction max angle", 
        default=45.0, 
        min=0.0, 
        max=90.0
    )
    
    # Orientation controls
    scene.airt_yaw_offset_deg = bpy.props.FloatProperty(
        name="Yaw offset (deg)", 
        default=0.0, 
        min=-180.0, 
        max=180.0
    )
    
    scene.airt_invert_z = bpy.props.BoolProperty(
        name="Flip Z (up/down)", 
        default=False
    )
    
    scene.airt_calibrate_direct = bpy.props.BoolProperty(
        name="Calibrate direct (1/r)", 
        default=True
    )
    
    # Air absorption settings
    scene.airt_air_enable = bpy.props.BoolProperty(
        name="Air absorption (freq)", 
        default=True
    )
    
    scene.airt_air_temp_c = bpy.props.FloatProperty(
        name="Air temp (deg C)", 
        default=20.0, 
        min=-30.0, 
        max=50.0
    )
    
    scene.airt_air_humidity = bpy.props.FloatProperty(
        name="Rel humidity (%)", 
        default=50.0, 
        min=0.0, 
        max=100.0
    )
    
    scene.airt_air_pressure_kpa = bpy.props.FloatProperty(
        name="Air pressure (kPa)", 
        default=101.325, 
        min=80.0, 
        max=110.0
    )
    
    # Output options
    scene.airt_quick_broadband = bpy.props.BoolProperty(
        name="Quick mode (broadband)",
        description="Bypass multi-band path filtering for speed; writes broadband impulses",
        default=False
    )
    
    scene.airt_omit_direct = bpy.props.BoolProperty(
        name="Omit direct (reverb-only)",
        description="Do not write the direct sound into the IR; useful for general reverb buses",
        default=False
    )
    
    # Internal threshold for ray termination
    scene.airt_min_throughput = bpy.props.FloatProperty(
        name="Min throughput",
        description="Minimum ray energy before termination",
        default=1e-6,  # Changed from 1e-4 to 1e-6 - allow much weaker rays
        min=1e-8,      # Changed from 1e-6 to 1e-8
        max=1e-2
    )


def unregister_acoustic_props():
    """Unregister all acoustic properties."""
    # Object properties
    object_attrs = [
        "absorption",
        "absorption_bands", 
        "scatter",
        "scatter_bands",
        "transmission",
        "is_acoustic_source",
        "is_acoustic_receiver",
        "airt_material_preset"
    ]
    
    for attr in object_attrs:
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    
    # Scene properties  
    scene_attrs = [
        "airt_num_rays", "airt_passes", "airt_max_order", "airt_sr", "airt_ir_seconds",
        "airt_angle_tol_deg", "airt_wav_subtype", "airt_seed", "airt_recv_radius",
        "airt_trace_mode", "airt_rr_enable", "airt_rr_start", "airt_rr_p",
        "airt_spec_rough_deg", "airt_enable_seg_capture", "airt_enable_diffraction",
        "airt_diffraction_samples", "airt_diffraction_max_deg",
        "airt_yaw_offset_deg", "airt_invert_z", "airt_calibrate_direct",
        "airt_air_enable", "airt_air_temp_c", "airt_air_humidity", "airt_air_pressure_kpa",
        "airt_quick_broadband", "airt_omit_direct", "airt_min_throughput"
    ]
    
    for attr in scene_attrs:
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)