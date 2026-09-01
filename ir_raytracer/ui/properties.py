"""Artist-facing Blender properties for the unified acoustic renderer."""
from __future__ import annotations

import bpy

from ..core.acoustics import MATERIAL_PRESET_DATA, MATERIAL_PRESETS, NUM_BANDS


_MATERIAL_GUARD = set()
_QUALITY_GUARD = set()


QUALITY_PROFILES = {
    'PREVIEW': {
        'airt_num_rays': 512,
        'airt_max_order': 24,
        'airt_rr_start': 12,
        'airt_rr_p': 0.95,
        'airt_min_throughput': 1e-5,
    },
    'BALANCED': {
        'airt_num_rays': 1024,
        'airt_max_order': 32,
        'airt_rr_start': 20,
        'airt_rr_p': 0.97,
        'airt_min_throughput': 1e-6,
    },
    'HIGH': {
        'airt_num_rays': 4096,
        'airt_max_order': 64,
        'airt_rr_start': 32,
        'airt_rr_p': 0.98,
        'airt_min_throughput': 1e-6,
    },
    'ULTRA': {
        'airt_num_rays': 16384,
        'airt_max_order': 128,
        'airt_rr_start': 48,
        'airt_rr_p': 0.99,
        'airt_min_throughput': 1e-8,
    },
}


def _material_items():
    items = [('CUSTOM', 'Custom', 'User-defined acoustic coefficients')]
    for identifier, label, _absorption, _scatter in MATERIAL_PRESET_DATA:
        preset = MATERIAL_PRESETS[identifier]
        items.append((
            identifier,
            label,
            f"Average absorption {preset['absorption']:.2f}, scattering {preset['scatter']:.2f}",
        ))
    return items


def _update_material_preset(self, _context):
    preset = MATERIAL_PRESETS.get(getattr(self, 'airt_material_preset', 'CUSTOM'))
    if preset is None:
        return
    key = id(self)
    _MATERIAL_GUARD.add(key)
    try:
        self.absorption = float(preset['absorption'])
        self.scatter = float(preset['scatter'])
        self.transmission = 0.0
        self.absorption_bands = preset['absorption_spectrum']
        self.scatter_bands = preset['scatter_spectrum']
        self.transmission_bands = tuple(0.0 for _ in range(NUM_BANDS))
    finally:
        _MATERIAL_GUARD.discard(key)


def _mark_material_custom(self, _context):
    if id(self) not in _MATERIAL_GUARD and self.airt_material_preset != 'CUSTOM':
        self.airt_material_preset = 'CUSTOM'


def _update_quality_profile(self, _context):
    profile = QUALITY_PROFILES.get(getattr(self, 'airt_quality_preset', 'CUSTOM'))
    if profile is None:
        return
    key = id(self)
    _QUALITY_GUARD.add(key)
    try:
        for property_name, value in profile.items():
            setattr(self, property_name, value)
    finally:
        _QUALITY_GUARD.discard(key)


def _mark_quality_custom(self, _context):
    if id(self) not in _QUALITY_GUARD and self.airt_quality_preset != 'CUSTOM':
        self.airt_quality_preset = 'CUSTOM'


def register_acoustic_props():
    obj = bpy.types.Object
    scene = bpy.types.Scene

    obj.airt_material_preset = bpy.props.EnumProperty(
        name="Acoustic Material",
        description="Starting point for frequency-dependent absorption and scattering",
        items=_material_items(),
        default='CUSTOM',
        update=_update_material_preset,
    )
    obj.absorption = bpy.props.FloatProperty(
        name="Absorption",
        description="Broadband absorbed-energy fraction",
        default=0.2,
        min=0.0,
        max=1.0,
        update=_mark_material_custom,
    )
    obj.absorption_bands = bpy.props.FloatVectorProperty(
        name="Absorption Bands",
        description="Energy absorption at 125 Hz through 8 kHz",
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=tuple(0.2 for _ in range(NUM_BANDS)),
        update=_mark_material_custom,
    )
    obj.scatter = bpy.props.FloatProperty(
        name="Scattering",
        description="Fraction of reflected energy distributed diffusely",
        default=0.35,
        min=0.0,
        max=1.0,
        update=_mark_material_custom,
    )
    obj.scatter_bands = bpy.props.FloatVectorProperty(
        name="Scattering Bands",
        description="Diffuse-reflection fraction at 125 Hz through 8 kHz",
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=tuple(0.35 for _ in range(NUM_BANDS)),
        update=_mark_material_custom,
    )
    obj.transmission = bpy.props.FloatProperty(
        name="Transmission",
        description="Broadband energy fraction passing through the surface",
        default=0.0,
        min=0.0,
        max=1.0,
        update=_mark_material_custom,
    )
    obj.transmission_bands = bpy.props.FloatVectorProperty(
        name="Transmission Bands",
        description="Transmitted-energy fraction at 125 Hz through 8 kHz",
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=tuple(0.0 for _ in range(NUM_BANDS)),
        update=_mark_material_custom,
    )
    obj.is_acoustic_source = bpy.props.BoolProperty(
        name="Acoustic Source",
        description="Use this object's evaluated world position as a sound source",
        default=False,
    )
    obj.is_acoustic_receiver = bpy.props.BoolProperty(
        name="Acoustic Receiver",
        description="Use this object's evaluated world position as the HOA receiver",
        default=False,
    )
    obj.show_frequency_details = bpy.props.BoolProperty(
        name="Band Details",
        default=False,
    )

    scene.airt_source_object = bpy.props.PointerProperty(
        name="Source",
        description="Object whose evaluated world position defines the source",
        type=bpy.types.Object,
    )
    scene.airt_receiver_object = bpy.props.PointerProperty(
        name="Receiver",
        description="Object whose evaluated world position defines the ambisonic listener",
        type=bpy.types.Object,
    )
    scene.airt_quality_preset = bpy.props.EnumProperty(
        name="Render Quality",
        description="Choose a practical ray budget or customize the advanced settings",
        items=[
            ('PREVIEW', 'Preview', 'Fast spatial and decay preview'),
            ('BALANCED', 'Balanced', 'Recommended starting point'),
            ('HIGH', 'High', 'Smoother final render'),
            ('ULTRA', 'Ultra High', 'Maximum convergence for long, complex spaces'),
            ('CUSTOM', 'Custom', 'Manually configured ray settings'),
        ],
        default='BALANCED',
        update=_update_quality_profile,
    )
    scene.airt_num_rays = bpy.props.IntProperty(
        name="Listener Rays",
        description="Directions sampled from the receiver; more rays reduce diffuse-tail variance",
        default=1024,
        min=128,
        max=131072,
        update=_mark_quality_custom,
    )
    scene.airt_max_order = bpy.props.IntProperty(
        name="Maximum Bounces",
        description="Maximum stochastic surface interactions",
        default=32,
        min=1,
        max=512,
        update=_mark_quality_custom,
    )
    scene.airt_sr = bpy.props.EnumProperty(
        name="Sample Rate",
        items=[
            ('44100', '44.1 kHz', '44,100 samples per second'),
            ('48000', '48 kHz', '48,000 samples per second'),
            ('96000', '96 kHz', '96,000 samples per second'),
            ('192000', '192 kHz', '192,000 samples per second'),
        ],
        default='48000',
    )
    scene.airt_ir_seconds = bpy.props.FloatProperty(
        name="IR Duration",
        description="Rendered impulse-response duration in seconds",
        default=2.0,
        min=0.1,
        max=20.0,
        precision=2,
    )
    scene.airt_output_content = bpy.props.EnumProperty(
        name="IR Content",
        items=[
            ('FULL', 'Full IR', 'Direct sound, early reflections, and diffuse field'),
            ('REFLECTIONS', 'Wet Reflections', 'Early reflections and diffuse field without direct sound'),
            ('DIFFUSE', 'Diffuse Field Only', 'Stochastic reflected field without direct or deterministic early events'),
        ],
        default='FULL',
    )
    scene.airt_early_reflections = bpy.props.BoolProperty(
        name="Deterministic Early Reflections",
        description="Resolve first-order planar specular reflections explicitly",
        default=True,
    )
    scene.airt_early_gain_db = bpy.props.FloatProperty(
        name="Early Gain",
        description="Artistic gain applied to deterministic early reflections",
        default=0.0,
        min=-24.0,
        max=24.0,
    )
    scene.airt_diffuse_gain_db = bpy.props.FloatProperty(
        name="Diffuse Gain",
        description="Artistic gain applied to the stochastic reverberant field",
        default=0.0,
        min=-24.0,
        max=24.0,
    )
    scene.airt_seed = bpy.props.IntProperty(
        name="Random Seed",
        description="Use a nonzero seed for repeatable renders; zero creates a new realization",
        default=1,
        min=0,
    )
    scene.airt_spec_rough_deg = bpy.props.FloatProperty(
        name="Specular Roughness",
        description="Angular width of stochastic glossy reflection lobes",
        default=8.0,
        min=0.0,
        max=45.0,
    )
    scene.airt_rr_enable = bpy.props.BoolProperty(
        name="Russian Roulette",
        description="Unbiased probabilistic termination of long paths",
        default=True,
    )
    scene.airt_rr_start = bpy.props.IntProperty(
        name="Start Bounce",
        default=20,
        min=1,
        max=512,
        update=_mark_quality_custom,
    )
    scene.airt_rr_p = bpy.props.FloatProperty(
        name="Survival Probability",
        default=0.97,
        min=0.5,
        max=1.0,
        update=_mark_quality_custom,
    )
    scene.airt_min_throughput = bpy.props.FloatProperty(
        name="Minimum Path Energy",
        default=1e-6,
        min=1e-10,
        max=1e-2,
        update=_mark_quality_custom,
    )
    scene.airt_air_enable = bpy.props.BoolProperty(
        name="Air Absorption",
        default=True,
    )
    scene.airt_air_temp_c = bpy.props.FloatProperty(
        name="Temperature",
        default=20.0,
        min=-30.0,
        max=50.0,
    )
    scene.airt_air_humidity = bpy.props.FloatProperty(
        name="Relative Humidity",
        default=50.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE',
    )
    scene.airt_air_pressure_kpa = bpy.props.FloatProperty(
        name="Pressure",
        default=101.325,
        min=80.0,
        max=110.0,
    )
    scene.airt_enable_diffraction = bpy.props.BoolProperty(
        name="Edge Diffraction",
        description="Add the bounded single-edge shadow approximation",
        default=False,
    )
    scene.airt_diffraction_samples = bpy.props.IntProperty(
        name="Maximum Edge Paths",
        default=4,
        min=1,
        max=32,
    )
    scene.airt_diffraction_max_deg = bpy.props.FloatProperty(
        name="Maximum Bend Angle",
        default=45.0,
        min=1.0,
        max=90.0,
    )
    scene.airt_yaw_offset_deg = bpy.props.FloatProperty(
        name="Ambisonic Yaw",
        default=0.0,
        min=-180.0,
        max=180.0,
    )
    scene.airt_invert_z = bpy.props.BoolProperty(
        name="Flip Ambisonic Z",
        default=False,
    )
    scene.airt_output_path = bpy.props.StringProperty(
        name="Output WAV",
        description="Destination for the 16-channel ACN/SN3D impulse response",
        default="//ambisonic_ir.wav",
        subtype='FILE_PATH',
    )
    scene.airt_wav_subtype = bpy.props.EnumProperty(
        name="WAV Format",
        items=[
            ('FLOAT', '32-bit Float', 'Recommended for convolution impulse responses'),
            ('PCM_24', '24-bit PCM', 'Integer output with less headroom'),
        ],
        default='FLOAT',
    )
    scene.airt_normalization = bpy.props.EnumProperty(
        name="Output Level",
        items=[
            ('PEAK', 'Normalize Peak', 'Normalize to the selected peak level'),
            ('PRESERVE', 'Preserve Relative Level', 'Keep the renderer\'s 1/r reference level'),
        ],
        default='PEAK',
    )
    scene.airt_peak_db = bpy.props.FloatProperty(
        name="Normalized Peak",
        default=-1.0,
        min=-24.0,
        max=0.0,
    )
    scene.airt_last_render_summary = bpy.props.StringProperty(
        name="Last Render",
        default="",
    )


def unregister_acoustic_props():
    object_names = (
        'airt_material_preset', 'absorption', 'absorption_bands', 'scatter',
        'scatter_bands', 'transmission', 'transmission_bands',
        'is_acoustic_source', 'is_acoustic_receiver', 'show_frequency_details',
    )
    scene_names = (
        'airt_source_object', 'airt_receiver_object', 'airt_quality_preset',
        'airt_num_rays', 'airt_max_order', 'airt_sr', 'airt_ir_seconds',
        'airt_output_content', 'airt_early_reflections', 'airt_early_gain_db',
        'airt_diffuse_gain_db', 'airt_seed', 'airt_spec_rough_deg',
        'airt_rr_enable', 'airt_rr_start', 'airt_rr_p', 'airt_min_throughput',
        'airt_air_enable', 'airt_air_temp_c', 'airt_air_humidity',
        'airt_air_pressure_kpa', 'airt_enable_diffraction',
        'airt_diffraction_samples', 'airt_diffraction_max_deg',
        'airt_yaw_offset_deg', 'airt_invert_z', 'airt_output_path',
        'airt_wav_subtype', 'airt_normalization', 'airt_peak_db',
        'airt_last_render_summary',
    )
    for name in object_names:
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)
    for name in scene_names:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
