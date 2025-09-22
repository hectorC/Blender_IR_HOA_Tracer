# -*- coding: utf-8 -*-
bl_info = {
    "name": "Ambisonic IR Tracer",
    "blender": (4, 5, 0),
    "category": "Object",
    "author": "ChatGPT + Hector Centeno",
    "description": "Trace impulse responses with 3rd-order ambisonic encoding (ACN/SN3D) using reverse ray tracing with specular and diffuse reflections"
}

import bpy
import mathutils
from math import pi, sqrt, sin, cos, atan2, asin, acos
import numpy as np
import os
import sys
import random
from functools import lru_cache

# --- Required deps ------------------------------------------------------------
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False

try:
    from scipy.special import lpmv
except Exception:
    lpmv = None

def _band_label(freq_hz: float) -> str:
    if freq_hz >= 1000.0:
        return f"{int(freq_hz / 1000.0)} kHz"
    return f"{int(freq_hz)} Hz"


BAND_CENTERS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
BAND_LABELS = tuple(_band_label(f) for f in BAND_CENTERS_HZ)
_NUM_BANDS = len(BAND_CENTERS_HZ)
DEFAULT_ABSORPTION_SPECTRUM = tuple(0.2 for _ in BAND_CENTERS_HZ)
DEFAULT_SCATTER_SPECTRUM = tuple(0.35 for _ in BAND_CENTERS_HZ)

MATERIAL_PRESET_DATA = [
    (
        'WOOD',
        'Wood (panel)',
        (0.18, 0.17, 0.16, 0.14, 0.12, 0.10, 0.10),
        (0.30, 0.32, 0.34, 0.36, 0.38, 0.38, 0.38)
    ),
    (
        'CONCRETE',
        'Concrete',
        (0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06),
        (0.10, 0.12, 0.14, 0.16, 0.18, 0.18, 0.18)
    ),
    (
        'CARPET',
        'Carpet',
        (0.08, 0.12, 0.30, 0.55, 0.65, 0.70, 0.70),
        (0.55, 0.57, 0.60, 0.62, 0.62, 0.62, 0.62)
    ),
    (
        'TILE',
        'Tile',
        (0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05),
        (0.15, 0.17, 0.18, 0.20, 0.22, 0.22, 0.22)
    ),
    (
        'BRICK',
        'Brick',
        (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09),
        (0.35, 0.37, 0.40, 0.43, 0.45, 0.45, 0.45)
    )
]


def _avg(values):
    return float(sum(values)) / max(len(values), 1)


MATERIAL_PRESETS = {
    identifier: {
        'absorption_spectrum': tuple(float(max(0.0, min(1.0, v))) for v in absorption),
        'scatter_spectrum': tuple(float(max(0.0, min(1.0, v))) for v in scatter),
        'absorption': _avg(absorption),
        'scatter': _avg(scatter)
    }
    for identifier, _, absorption, scatter in MATERIAL_PRESET_DATA
}
_MATERIAL_PRESET_ITEMS = [('CUSTOM', 'Custom', 'User-defined absorption/scatter spectra')]
for identifier, label, _, _ in MATERIAL_PRESET_DATA:
    preset = MATERIAL_PRESETS[identifier]
    desc = (
        f"{label}: avg absorption {preset['absorption']:.2f}, "
        f"avg scatter {preset['scatter']:.2f}"
    )
    _MATERIAL_PRESET_ITEMS.append((identifier, label, desc))
_MATERIAL_PRESET_GUARD = set()


def _update_material_preset(self, context):
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
    key = id(self)
    if key in _MATERIAL_PRESET_GUARD:
        return
    if getattr(self, 'airt_material_preset', 'CUSTOM') != 'CUSTOM':
        self.airt_material_preset = 'CUSTOM'


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------

def register_acoustic_props():
    bpy.types.Object.airt_material_preset = bpy.props.EnumProperty(
        name="Material Preset",
        description="Apply common material absorption/scatter values",
        items=_MATERIAL_PRESET_ITEMS,
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
        size=_NUM_BANDS,
        min=0.0,
        max=1.0,
        subtype='NONE',
        default=DEFAULT_ABSORPTION_SPECTRUM,
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
        size=_NUM_BANDS,
        min=0.0,
        max=1.0,
        subtype='NONE',
        default=DEFAULT_SCATTER_SPECTRUM,
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
    bpy.types.Object.is_acoustic_source = bpy.props.BoolProperty(name="Acoustic Source", default=False)
    bpy.types.Object.is_acoustic_receiver = bpy.props.BoolProperty(name="Acoustic Receiver", default=False)

    scene = bpy.types.Scene
    scene.airt_num_rays = bpy.props.IntProperty(name="Rays", default=8192, min=128, max=131072)
    scene.airt_passes = bpy.props.IntProperty(name="Averaging passes", default=4, min=1, max=32)
    scene.airt_max_order = bpy.props.IntProperty(name="Max Bounces", default=64, min=0, max=1000)
    scene.airt_sr = bpy.props.IntProperty(name="Sample Rate", default=48000, min=8000, max=192000)
    scene.airt_ir_seconds = bpy.props.FloatProperty(name="IR Length (s)", default=2.0, min=0.1, max=20.0)
    scene.airt_angle_tol_deg = bpy.props.FloatProperty(name="Specular tol (deg)", default=8.0, min=0.1, max=30.0)
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
    scene.airt_seed = bpy.props.IntProperty(name="Random Seed", default=0, min=0)
    scene.airt_recv_radius = bpy.props.FloatProperty(name="Receiver radius (m)", default=0.25, min=0.001, max=2.0)
    scene.airt_trace_mode = bpy.props.EnumProperty(
        name="Tracing mode",
        description="Reverse = early specular; Forward = stochastic tail & early",
        items=[
            ('FORWARD','Forward (source->room)','Stochastic forward tracing with receiver capture + path connection'),
            ('REVERSE','Reverse (receiver->room)','Specular reverse tracing with LOS-to-source check')
        ],
        default='FORWARD'
    )
    scene.airt_rr_enable = bpy.props.BoolProperty(name="Russian roulette", default=True)
    scene.airt_rr_start = bpy.props.IntProperty(name="RR start bounce", default=8, min=0, max=1000)
    scene.airt_rr_p = bpy.props.FloatProperty(name="RR survive prob", default=0.9, min=0.05, max=1.0)
    scene.airt_spec_rough_deg = bpy.props.FloatProperty(name="Specular roughness (deg)", default=5.0, min=0.0, max=30.0)
    scene.airt_enable_seg_capture = bpy.props.BoolProperty(name="Capture along segments", default=False)
    scene.airt_enable_diffraction = bpy.props.BoolProperty(name="Enable diffraction", default=True)
    scene.airt_diffraction_samples = bpy.props.IntProperty(name="Diffraction samples", default=6, min=0, max=64)
    scene.airt_diffraction_max_deg = bpy.props.FloatProperty(name="Diffraction max angle", default=45.0, min=0.0, max=90.0)
    # Orientation controls (to match downstream decoder conventions)
    scene.airt_yaw_offset_deg = bpy.props.FloatProperty(name="Yaw offset (deg)", default=0.0, min=-180.0, max=180.0)
    scene.airt_invert_z = bpy.props.BoolProperty(name="Flip Z (up/down)", default=False)
    scene.airt_calibrate_direct = bpy.props.BoolProperty(name="Calibrate direct (1/r)", default=True)
    # Air absorption (frequency-dependent)
    scene.airt_air_enable = bpy.props.BoolProperty(name="Air absorption (freq)", default=True)
    scene.airt_air_temp_c = bpy.props.FloatProperty(name="Air temp (deg C)", default=20.0, min=-30.0, max=50.0)
    scene.airt_air_humidity = bpy.props.FloatProperty(name="Rel humidity (%)", default=50.0, min=0.0, max=100.0)
    scene.airt_air_pressure_kpa = bpy.props.FloatProperty(name="Air pressure (kPa)", default=101.325, min=80.0, max=110.0)
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


def unregister_acoustic_props():
    for attr in (
        "absorption",
        "absorption_bands",
        "scatter",
        "scatter_bands",
        "transmission",
        "is_acoustic_source",
        "is_acoustic_receiver",
        "airt_material_preset"
    ):
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    scene = bpy.types.Scene
    for k in (
        "airt_num_rays","airt_passes","airt_max_order","airt_sr","airt_ir_seconds",
        "airt_angle_tol_deg","airt_wav_subtype","airt_seed","airt_recv_radius",
        "airt_trace_mode","airt_rr_enable","airt_rr_start","airt_rr_p",
        "airt_spec_rough_deg","airt_enable_seg_capture","airt_enable_diffraction",
        "airt_diffraction_samples","airt_diffraction_max_deg",
        "airt_yaw_offset_deg","airt_invert_z","airt_calibrate_direct",
        "airt_air_enable","airt_air_temp_c","airt_air_humidity","airt_air_pressure_kpa",
        "airt_quick_broadband","airt_omit_direct","airt_min_throughput"
    ):
        if hasattr(scene, k):
            delattr(scene, k)

# ----------------------------------------------------------------------------
# UI Panel
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


def _draw_band_vector(layout, obj, prop_name: str, title: str):
    box = layout.box()
    box.label(text=title)
    grid = box.grid_flow(row_major=True, columns=_NUM_BANDS, even_columns=True, even_rows=True, align=True)
    for idx, label in enumerate(BAND_LABELS):
        grid.prop(obj, prop_name, index=idx, text=label, slider=True)


class AIRT_PT_Panel(bpy.types.Panel):
    bl_idname = "AIRT_PT_panel"
    bl_label = "Ambisonic IR Tracer"
    bl_category = "IR Tracer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        col = layout.column(align=True)
        col.label(text="Object Properties")
        if obj:
            col.prop(obj, "airt_material_preset")
            col.prop(obj, "absorption")
            _draw_band_vector(col, obj, "absorption_bands", "Absorption Spectrum (octave centers)")
            col.prop(obj, "scatter")
            _draw_band_vector(col, obj, "scatter_bands", "Scatter Spectrum (octave centers)")
            col.prop(obj, "transmission")
            col.prop(obj, "is_acoustic_source")
            col.prop(obj, "is_acoustic_receiver")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Render Settings")
        col.prop(context.scene, "airt_trace_mode")
        col.prop(context.scene, "airt_recv_radius")
        col.prop(context.scene, "airt_num_rays")
        col.prop(context.scene, "airt_passes")
        col.prop(context.scene, "airt_max_order")  # max bounces
        col.prop(context.scene, "airt_sr")
        col.prop(context.scene, "airt_ir_seconds")
        col.prop(context.scene, "airt_angle_tol_deg")
        col.prop(context.scene, "airt_wav_subtype")
        col.prop(context.scene, "airt_spec_rough_deg")
        col.prop(context.scene, "airt_enable_seg_capture")
        col.prop(context.scene, "airt_enable_diffraction")
        if context.scene.airt_enable_diffraction:
            col.prop(context.scene, "airt_diffraction_samples")
            col.prop(context.scene, "airt_diffraction_max_deg")
        col.prop(context.scene, "airt_yaw_offset_deg")
        col.prop(context.scene, "airt_invert_z")
        col.prop(context.scene, "airt_calibrate_direct")
        col.prop(context.scene, "airt_air_enable")
        if context.scene.airt_air_enable:
            col.prop(context.scene, "airt_air_temp_c")
            col.prop(context.scene, "airt_air_humidity")
            col.prop(context.scene, "airt_air_pressure_kpa")
        col.prop(context.scene, "airt_quick_broadband")
        col.prop(context.scene, "airt_omit_direct")
        col.prop(context.scene, "airt_rr_enable")
        if context.scene.airt_rr_enable:
            col.prop(context.scene, "airt_rr_start")
            col.prop(context.scene, "airt_rr_p")
        col.prop(context.scene, "airt_seed")
        layout.separator()
        layout.operator("airt.render_ir", icon='RENDER_ANIMATION')

# ----------------------------------------------------------------------------
# Operator
# ----------------------------------------------------------------------------

class AIRT_OT_RenderIR(bpy.types.Operator):
    bl_idname = "airt.render_ir"
    bl_label = "Render Ambisonic IR"

    def execute(self, context):
        if not HAVE_SF:
            cmd = f"{sys.executable} -m pip install soundfile"
            self.report({'ERROR'}, "python-soundfile is required. Install with:\n" + cmd)
            return {'CANCELLED'}
        if lpmv is None:
            cmd = f"{sys.executable} -m pip install scipy"
            self.report({'ERROR'}, "scipy is required for SH encoding. Install with:\n" + cmd)
            return {'CANCELLED'}

        scene = context.scene
        sources = [o for o in scene.objects if getattr(o, 'is_acoustic_source', False)]
        receivers = [o for o in scene.objects if getattr(o, 'is_acoustic_receiver', False)]
        if not sources or not receivers:
            self.report({'WARNING'}, "Need one source and one receiver")
            return {'CANCELLED'}

        source = sources[0].location.copy()
        receiver = receivers[0].location.copy()

        passes = max(1, int(scene.airt_passes))
        num_rays = max(1, int(scene.airt_num_rays))
        bvh, obj_map = build_bvh(context)

        ir = None
        for i in range(passes):
            if scene.airt_seed:
                seed = int(scene.airt_seed) + i
                random.seed(seed)
                np.random.seed(seed)
            ir_i = trace_ir(context, source, receiver, bvh=bvh, obj_map=obj_map)
            if ir is None:
                ir = ir_i.astype(np.float32)
            else:
                ir += ir_i.astype(np.float32)
        ir /= float(passes)

        # Optional: calibrate direct-path amplitude to 1/r so distance perception is correct
        if bool(getattr(scene, 'airt_calibrate_direct', False)):
            ir, info = calibrate_direct_1_over_r(ir, context, source, receiver)
            self.report({'INFO'}, info)

        sr = scene.airt_sr
        subtype = scene.airt_wav_subtype
        wav_path = get_writable_path("ir_output.wav")
        try:
            sf.write(wav_path, ir.T.astype(np.float32), samplerate=sr, subtype=subtype)
            self.report({'INFO'}, f"IR saved to {wav_path} ({subtype})")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write WAV via soundfile: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}



# ----------------------------------------------------------------------------
# Utility: air absorption (ISO 9613-1), calibration and output path resolution
# ----------------------------------------------------------------------------

def iso9613_alpha_dbpm(f_hz: float, T_c: float, rh_pct: float, p_kpa: float) -> float:
    """Approximate ISO 9613-1 atmospheric absorption coefficient alpha in dB/m.
    Inputs: frequency in Hz, temperature in deg C, relative humidity in %, pressure in kPa.
    Returns alpha [dB/m].
    """
    import numpy as _np
    T = 273.15 + float(T_c)
    T0 = 293.15
    fr_h = max(0.0, min(100.0, float(rh_pct))) / 100.0
    P = max(1e-3, float(p_kpa))
    P0 = 101.325
    # Relaxation frequencies (approximate forms used in practice)
    frO = 24.0 + 4.04e4 * fr_h * (0.02 + fr_h) / (0.391 + fr_h)
    frN = (T / T0)**(-0.5) * (9.0 + 280.0 * fr_h * _np.exp(-4.17 * ((T / T0)**(-1.0/3.0) - 1.0)))
    fk = (float(f_hz) / 1000.0)
    fk2 = fk * fk
    # Classical + rotational/translational + vibrational contributions
    term_class = 1.84e-11 * (P0 / P) * _np.sqrt(T / T0)
    term_O = (T / T0)**(-2.5) * (0.01275 * _np.exp(-2239.1 / T)) * (frO / (frO*frO + fk2))
    term_N = (T / T0)**(-2.5) * (0.1068  * _np.exp(-3352.0 / T)) * (frN / (frN*frN + fk2))
    alpha = 8.686 * fk2 * (term_class + term_O + term_N)  # dB/m
    return float(max(0.0, alpha))


# ----------------------------------------------------------------------------

def _speed_of_sound_ms(context):
    """Speed of sound in air [m/s] using a humidity-aware approximation (Cramer's formula)."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    temp_c = float(getattr(scene, 'airt_air_temp_c', 20.0))
    rh = float(getattr(scene, 'airt_air_humidity', 50.0))
    return float(331.3 + 0.606 * temp_c + 0.0124 * rh)


def _speed_of_sound_bu(context):
    """Convert physical speed of sound to Blender units using the scene scale."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    scale_length = getattr(unit_settings, "scale_length", 1.0) if unit_settings else 1.0
    unit_scale = float(scale_length or 1.0)
    c_ms = _speed_of_sound_ms(context)
    return c_ms / max(unit_scale, 1e-9)


def calibrate_direct_1_over_r(ir: np.ndarray, context, source, receiver):
    """Scale the entire IR so that the direct-path amplitude in W matches 1/dist."""
    try:
        sr = int(context.scene.airt_sr)
        c = _speed_of_sound_bu(context)
        dist = (receiver - source).length
        if dist <= 1e-9 or ir.shape[1] <= 22:
            return ir, "Calibrate: skipped (zero distance or IR too short)"
        delay = (dist / c) * sr
        n = int(round(delay))
        n0 = max(0, n - 10)
        n1 = min(ir.shape[1], n + 11)
        a_meas = float(np.max(np.abs(ir[0, n0:n1])))
        if a_meas <= 1e-9:
            return ir, f"Calibrate: skipped (no direct energy near n~{n})"
        a_exp = 1.0 / max(dist, 1e-9)
        k = a_exp / a_meas
        ir *= k
        return ir, f"Calibrate: dist={dist:.3f}m, expW={a_exp:.6f}, measW={a_meas:.6f}, k={k:.4f}, n~{n}, window=+/-10"
    except Exception as e:
        return ir, f"Calibrate: skipped (err: {e})"

# ----------------------------------------------------------------------------
# Utility: output path resolution
# ----------------------------------------------------------------------------

def get_writable_path(filename: str) -> str:
    import tempfile
    # Try blend folder first
    try:
        base = bpy.path.abspath("//")
        if base and os.path.isdir(base):
            path = os.path.join(base, filename)
            with open(path, 'ab') as _f:
                pass
            return path
    except Exception:
        pass
    # Blender temp dir
    try:
        if getattr(bpy.app, 'tempdir', None):
            path = os.path.join(bpy.app.tempdir, filename)
            with open(path, 'ab') as _f:
                pass
            return path
    except Exception:
        pass
    # System temp
    try:
        import tempfile as _tf
        path = os.path.join(_tf.gettempdir(), filename)
        with open(path, 'ab') as _f:
            pass
        return path
    except Exception:
        pass
    # Home as last resort
    return os.path.join(os.path.expanduser('~'), filename)

# ----------------------------------------------------------------------------
# Ray Tracing (Reverse from Receiver) with Specular + Diffuse Scattering
# ----------------------------------------------------------------------------

def build_bvh(context):
    verts = []
    polys = []
    obj_map = []  # polygon index -> object (for absorption/scatter)
    scene = getattr(context, "scene", None) or bpy.context.scene
    view_layer = getattr(context, "view_layer", None)
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    if callable(depsgraph_get):
        depsgraph = depsgraph_get()
    else:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        visible = obj.visible_get(view_layer=view_layer) if view_layer else obj.visible_get()
        if obj.type == 'MESH' and not getattr(obj, 'is_acoustic_source', False) and not getattr(obj, 'is_acoustic_receiver', False) and visible:
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            if mesh is None:
                continue
            mesh.transform(obj.matrix_world)
            base_index = len(verts)
            verts.extend([v.co.copy() for v in mesh.vertices])
            polys.extend([tuple(base_index + vi for vi in p.vertices) for p in mesh.polygons])
            obj_map.extend([obj] * len(mesh.polygons))
            obj_eval.to_mesh_clear()
    import mathutils.bvhtree
    bvh = mathutils.bvhtree.BVHTree.FromPolygons(verts, polys) if polys else None
    return bvh, obj_map


def reflect(vec, normal):
    return (vec - 2.0 * vec.dot(normal) * normal).normalized()


def _get_obj_spectrum(obj, vec_attr, scalar_attr, default_vec):
    if obj is None:
        return np.array(default_vec, dtype=np.float32)
    if hasattr(obj, vec_attr):
        values = getattr(obj, vec_attr)
        if values is not None and len(values) == _NUM_BANDS:
            return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)
    scalar = float(getattr(obj, scalar_attr, default_vec[0])) if obj else default_vec[0]
    return np.clip(np.full(_NUM_BANDS, scalar, dtype=np.float32), 0.0, 1.0)


def get_absorption_spectrum(obj):
    return _get_obj_spectrum(obj, 'absorption_bands', 'absorption', DEFAULT_ABSORPTION_SPECTRUM)


def get_scatter_spectrum(obj):
    return _get_obj_spectrum(obj, 'scatter_bands', 'scatter', DEFAULT_SCATTER_SPECTRUM)


def get_transmission_coeff(obj):
    return float(np.clip(getattr(obj, 'transmission', 0.0) if obj else 0.0, 0.0, 1.0))


def los_clear(p0, p1, bvh, eps=1e-4):
    if bvh is None:
        return True
    d = (p1 - p0)
    dist = d.length
    if dist <= 1e-9:
        return True
    dirn = d.normalized()
    hit, nrm, idx, t = bvh.ray_cast(p0 + dirn * eps, dirn)
    if hit is None:
        return True
    return t >= dist - 1e-4


def add_impulse(ir, ambi_vec, delay_samples, amp):
    _add_filtered_impulse(ir, ambi_vec, delay_samples, amp, np.ones(_NUM_BANDS, dtype=np.float32), 48000)


def _air_attenuation_bands(distance_m, context):
    if distance_m <= 0.0 or not bool(getattr(context.scene, 'airt_air_enable', True)):
        return np.ones(_NUM_BANDS, dtype=np.float32)
    temp = float(getattr(context.scene, 'airt_air_temp_c', 20.0))
    rh = float(getattr(context.scene, 'airt_air_humidity', 50.0))
    pk = float(getattr(context.scene, 'airt_air_pressure_kpa', 101.325))
    gains = []
    for f in BAND_CENTERS_HZ:
        alpha_dbpm = iso9613_alpha_dbpm(f, temp, rh, pk)
        gains.append(10.0 ** (-(alpha_dbpm * distance_m) / 20.0))
    return np.clip(np.array(gains, dtype=np.float32), 1e-4, 1.0)


@lru_cache(maxsize=4096)
def _band_kernel_cache(band_key, sr, kernel_len):
    band_profile = np.array(band_key, dtype=np.float32)
    kernel_len = int(kernel_len)
    sr = max(1000, int(sr))
    n_fft = max(64, 1 << int(np.ceil(np.log2(kernel_len * 4))))
    freq_axis = np.linspace(0.0, sr * 0.5, n_fft // 2 + 1, dtype=np.float32)
    mags = np.empty_like(freq_axis)
    mags[0] = float(band_profile[0])
    if freq_axis.shape[0] > 1:
        log_freq = np.log10(np.maximum(freq_axis[1:], 1.0))
        log_bands = np.log10(np.array(BAND_CENTERS_HZ, dtype=np.float32))
        interp = np.interp(log_freq, log_bands, band_profile, left=band_profile[0], right=band_profile[-1])
        mags[1:] = interp
    mag_full = np.concatenate([mags, mags[-2:0:-1]])
    ir = np.fft.irfft(mag_full, n=n_fft).real[:kernel_len]
    if kernel_len >= 4:
        window = np.hanning(kernel_len * 2)[kernel_len:kernel_len * 2]
        ir *= window
    s = float(np.sum(ir))
    target_dc = mags[0]
    if abs(s) > 1e-12:
        ir *= target_dc / s
    return ir.astype(np.float32)


def _design_band_kernel(band_profile, sr, kernel_len=16):
    key = tuple(float(round(float(v), 5)) for v in band_profile)
    return _band_kernel_cache(key, int(sr), int(kernel_len))


def _add_filtered_impulse(ir, ambi_vec, delay_samples, amp, band_profile, sr, kernel_len=16):
    kernel = _design_band_kernel(band_profile, sr, kernel_len)
    base = int(np.floor(delay_samples))
    frac = float(delay_samples - base)
    weights = ((base, 1.0 - frac), (base + 1, frac))
    wrote = False
    for start, w in weights:
        if w <= 0.0:
            continue
        for k, kv in enumerate(kernel):
            idx = start + k
            if 0 <= idx < ir.shape[1]:
                ir[:, idx] += ambi_vec * (amp * w * kv)
                wrote = True
    return wrote


def add_filtered_impulse(ir, ambi_vec, delay_samples, amp, band_profile, sr):
    return _add_filtered_impulse(ir, ambi_vec, delay_samples, amp, band_profile, sr)


def add_impulse_simple(ir, ambi_vec, delay_samples, amp):
    n = int(np.floor(delay_samples))
    frac = float(delay_samples - n)
    if 0 <= n < ir.shape[1]:
        ir[:, n] += ambi_vec * amp * (1.0 - frac)
    if 0 <= n + 1 < ir.shape[1]:
        ir[:, n + 1] += ambi_vec * amp * frac


def segment_hits_sphere(p0, p1, center, radius):
    """Check if segment p0->p1 intersects a sphere at center with radius.
    Returns (hit, t, point). t in [0,1] along the segment from p0.
    """
    v = p1 - p0
    w = p0 - center
    a = v.dot(v)
    b = 2.0 * v.dot(w)
    c = w.dot(w) - radius * radius
    disc = b*b - 4*a*c
    if disc < 0.0 or a <= 0.0:
        return False, None, None
    sd = sqrt(disc)
    t1 = (-b - sd) / (2*a)
    t2 = (-b + sd) / (2*a)
    t_hit = None
    if 0.0 <= t1 <= 1.0:
        t_hit = t1
    elif 0.0 <= t2 <= 1.0:
        t_hit = t2
    if t_hit is None:
        return False, None, None
    point = p0 + v * t_hit
    return True, float(t_hit), point


def trace_ir(context, source, receiver, bvh=None, obj_map=None, directions=None):
    if bvh is None or obj_map is None:
        bvh, obj_map = build_bvh(context)
    if directions is None:
        num_rays = max(1, int(context.scene.airt_num_rays))
        directions = generate_ray_directions(num_rays)
    mode = context.scene.airt_trace_mode
    if mode == 'FORWARD':
        return _trace_ir_forward(context, source, receiver, bvh, obj_map, directions)
    else:
        return _trace_ir_reverse(context, source, receiver, bvh, obj_map, directions)


def _trace_ir_forward(context, source, receiver, bvh, obj_map, directions):
    num_channels = 16
    sr = int(context.scene.airt_sr)
    ir_length = int(context.scene.airt_ir_seconds * sr)
    ir = np.zeros((num_channels, ir_length), dtype=np.float32)

    scene = context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    unit_scale = float(getattr(unit_settings, "scale_length", 1.0) or 1.0)
    c = _speed_of_sound_bu(context)
    max_bounces = int(context.scene.airt_max_order)
    tol_rad = context.scene.airt_angle_tol_deg * pi / 180.0
    recv_r_m = max(1e-6, float(context.scene.airt_recv_radius))
    recv_r = recv_r_m / max(unit_scale, 1e-9)
    rr_enable = bool(context.scene.airt_rr_enable)
    rr_start = int(context.scene.airt_rr_start)
    rr_survive = max(0.05, min(1.0, float(context.scene.airt_rr_p)))
    seg_capture = bool(context.scene.airt_enable_seg_capture)
    rough_rad = max(0.0, float(context.scene.airt_spec_rough_deg)) * pi / 180.0
    enable_diffraction = bool(getattr(context.scene, 'airt_enable_diffraction', False))
    diff_samples = int(getattr(context.scene, 'airt_diffraction_samples', 0))
    diff_max_angle = max(0.0, float(getattr(context.scene, 'airt_diffraction_max_deg', 40.0))) * pi / 180.0
    eps = 1e-4

    band_one = np.ones(_NUM_BANDS, dtype=np.float32)
    quick = bool(getattr(context.scene, 'airt_quick_broadband', False))
    num_dirs = max(1, len(directions))
    ray_weight = 1.0 / float(num_dirs)
    omit_direct = bool(getattr(context.scene, 'airt_omit_direct', False))
    thr = float(getattr(context.scene, 'airt_min_throughput', 1e-4))
    pi4 = 4.0 * pi
    wrote_any = False

    def _path_band_profile(band_amp, distance_bu):
        air = _air_attenuation_bands(distance_bu * unit_scale, context)
        profile = np.array(band_amp, dtype=np.float32) * air
        return np.clip(profile, 0.0, 1e6)

    def _ambisonic(direction):
        return encode_ambisonics_3rd_order(_apply_orientation(direction, context))

    def _emit_impulse(band_amp, distance_bu, incoming_dir, amp_scalar):
        if distance_bu <= 0.0:
            return False
        band_profile = _path_band_profile(band_amp, distance_bu)
        if quick:
            gain = float(np.mean(band_profile)) if isinstance(band_profile, np.ndarray) else float(band_profile)
            gain = max(gain, 0.0)
            if gain <= 0.0:
                return False
            delay = (distance_bu / c) * sr
            ambi = _ambisonic(incoming_dir)
            ambi = apply_near_field_compensation(ambi, distance_bu * unit_scale, recv_r_m)
            if not np.any(np.abs(ambi) > 1e-8):
                ambi = np.zeros(16, dtype=np.float32)
                ambi[0] = 1.0
            add_impulse_simple(ir, ambi, delay, amp_scalar * gain)
            return True
        if not np.any(band_profile > 1e-8):
            return False
        delay = (distance_bu / c) * sr
        ambi = _ambisonic(incoming_dir)
        ambi = apply_near_field_compensation(ambi, distance_bu * unit_scale, recv_r_m)
        if not np.any(np.abs(ambi) > 1e-8):
            ambi = np.zeros(16, dtype=np.float32)
            ambi[0] = 1.0
        wrote = add_filtered_impulse(ir, ambi, delay, amp_scalar, band_profile, sr)
        if not wrote:
            add_impulse_simple(ir, ambi, delay, amp_scalar)
            wrote = True
        return wrote

    def _jitter_spec(direction):
        v = direction.normalized()
        if rough_rad <= 1e-6:
            return v
        if abs(v.x) < 0.5:
            helper = mathutils.Vector((1.0, 0.0, 0.0))
        else:
            helper = mathutils.Vector((0.0, 1.0, 0.0))
        t = v.cross(helper).normalized()
        b = v.cross(t).normalized()
        u = random.random()
        vphi = 2.0 * pi * random.random()
        cos_a = 1.0 - u * (1.0 - cos(rough_rad))
        sin_a = sqrt(max(0.0, 1.0 - cos_a * cos_a))
        return (v * cos_a + t * (sin_a * cos(vphi)) + b * (sin_a * sin(vphi))).normalized()

    def _add_diffraction(hit_point, normal, dir_in, to_receiver_vec, throughput_band, refl_amp_band, diff_frac_band, path_len_hit):
        if not enable_diffraction or diff_samples <= 0 or diff_max_angle <= 0.0:
            return
        axis = dir_in.cross(normal)
        if axis.length < 1e-6:
            axis = normal.cross(mathutils.Vector((0.0, 0.0, 1.0)))
            if axis.length < 1e-6:
                axis = mathutils.Vector((1.0, 0.0, 0.0))
        axis.normalize()
        base_dir = to_receiver_vec.normalized()
        for _ in range(diff_samples):
            angle = diff_max_angle * random.random()
            sign = -1.0 if random.random() < 0.5 else 1.0
            rot = mathutils.Matrix.Rotation(sign * angle, 3, axis)
            sample_dir = (rot @ base_dir).normalized()
            far = hit_point + sample_dir * 100.0
            if not los_clear(hit_point + normal * eps, far, bvh, eps=eps):
                continue
            got, t_hit, _ = segment_hits_sphere(hit_point, far, receiver, recv_r)
            if not got:
                continue
            seg_dist = (far - hit_point).length * t_hit
            total_dist = path_len_hit + seg_dist
            if total_dist <= 0.0:
                continue
            angle_factor = np.exp(-angle / max(diff_max_angle, 1e-6))
            diff_gain = np.sqrt(np.clip(diff_frac_band, 0.0, 1.0)) * angle_factor
            band_amp = throughput_band * refl_amp_band * diff_gain
            if not np.any(band_amp > 1e-6):
                continue
            incoming = (hit_point - receiver).normalized()
            amp_scalar = 1.0 / max(total_dist, recv_r)
            _emit_impulse(band_amp, total_dist, incoming, amp_scalar)
            break

    if bvh is None:
        return ir

    for d in directions:
        dirn = mathutils.Vector(d)
        if dirn.length == 0.0:
            continue
        dirn.normalize()
        pos = source.copy()
        throughput = band_one.copy()
        path_len = 0.0
        bounce = 0
        while bounce <= max_bounces:
            hit, normal, index, dist = bvh.ray_cast(pos + dirn * eps, dirn)
            if hit is None or index is None:
                if seg_capture and not (omit_direct and bounce == 0):
                    far = pos + dirn * 100.0
                    got, t_hit, _ = segment_hits_sphere(pos, far, receiver, recv_r)
                    if got:
                        seg_len = (far - pos).length * t_hit
                        total_dist = path_len + seg_len
                        incoming = (-dirn).normalized()
                        area = pi * recv_r * recv_r
                        view = area / max(pi4 * total_dist * total_dist, 1e-9)
                        amp_scalar = sqrt(max(view, 0.0)) / max(total_dist, recv_r)
                        wrote_any = _emit_impulse(throughput * ray_weight, total_dist, incoming, amp_scalar) or wrote_any
                break

            seg_len = float(dist)
            total_dist_hit = path_len + seg_len
            hit_point = mathutils.Vector(hit)
            normal = mathutils.Vector(normal)
            if normal.dot(dirn) > 0.0:
                normal = -normal

            hit_obj = obj_map[index] if 0 <= index < len(obj_map) else None
            absorption = get_absorption_spectrum(hit_obj)
            scatter_spec = get_scatter_spectrum(hit_obj)
            transmission_coeff = get_transmission_coeff(hit_obj)
            transmission_spec = np.clip(np.full(_NUM_BANDS, transmission_coeff, dtype=np.float32), 0.0, 1.0)
            refl_spec = np.clip(1.0 - absorption - transmission_spec, 0.0, 1.0)
            if not np.any(refl_spec > 1e-6) and transmission_coeff <= 1e-6:
                break

            spec_frac = np.clip(1.0 - scatter_spec, 0.0, 1.0)
            diff_frac = np.clip(scatter_spec, 0.0, 1.0)
            refl_amp_band = np.sqrt(refl_spec)
            spec_amp_band = refl_amp_band * np.sqrt(np.maximum(spec_frac, 1e-6))
            diff_amp_band = refl_amp_band * np.sqrt(np.maximum(diff_frac, 1e-6))
            trans_amp_band = np.sqrt(transmission_spec)

            if seg_capture and not (omit_direct and bounce == 0):
                got, t_hit, _ = segment_hits_sphere(pos, hit_point, receiver, recv_r)
                if got:
                    partial_len = seg_len * t_hit
                    total_dist = path_len + partial_len
                    incoming = (-dirn).normalized()
                    area = pi * recv_r * recv_r
                    view = area / max(pi4 * total_dist * total_dist, 1e-9)
                    amp_scalar = sqrt(max(view, 0.0)) / max(total_dist, recv_r)
                    wrote_any = _emit_impulse(throughput * ray_weight, total_dist, incoming, amp_scalar) or wrote_any

            to_rcv = receiver - hit_point
            dist_rcv = to_rcv.length
            if dist_rcv > 0.0:
                has_los = los_clear(hit_point + normal * eps, receiver, bvh, eps=eps)
                if has_los:
                    to_rcv_dir = to_rcv.normalized()
                    req_out = reflect(dirn, normal)
                    cosang = max(-1.0, min(1.0, req_out.dot(to_rcv_dir)))
                    dtheta = acos(cosang)
                    spec_lobe = np.exp(-(dtheta / max(tol_rad, 1e-6)) ** 2)
                    cos_i = max(0.0, (-dirn).dot(normal))
                    diff_lobe = cos_i / pi
                    total_weight_band = np.clip(spec_frac * spec_lobe + diff_frac * diff_lobe, 0.0, 1.0)
                    if np.any(total_weight_band > 1e-6):
                        band_amp = throughput * refl_amp_band * np.sqrt(total_weight_band)
                        total_dist = total_dist_hit + dist_rcv
                        amp_scalar = 1.0 / max(total_dist, recv_r)
                        incoming = (hit_point - receiver).normalized()
                        wrote_any = _emit_impulse(band_amp * ray_weight, total_dist, incoming, amp_scalar) or wrote_any
                else:
                    _add_diffraction(hit_point, normal, dirn, to_rcv, throughput * ray_weight, refl_amp_band, diff_frac, total_dist_hit)

            path_len = total_dist_hit
            base_throughput = throughput.copy()
            trans_prob = min(0.999, max(0.0, transmission_coeff))
            remaining = max(0.0, 1.0 - trans_prob)
            diff_prob = remaining * float(np.clip(np.mean(diff_frac), 0.0, 1.0))
            spec_prob = max(remaining - diff_prob, 0.0)

            rnd = random.random()
            next_dir = None
            new_throughput = None
            offset = normal * eps

            if trans_prob > 0.0 and rnd < trans_prob:
                next_dir = dirn.normalized()
                new_throughput = base_throughput * trans_amp_band
                offset = -normal * eps
            else:
                rnd -= trans_prob
                if diff_prob > 0.0 and rnd < diff_prob:
                    candidate = None
                    cos_out = 0.0
                    for _ in range(8):
                        cand = cosine_weighted_hemisphere(normal)
                        cos_out = max(0.0, cand.dot(normal))
                        if cos_out > 1e-6:
                            candidate = cand
                            break
                    if candidate is None:
                        break
                    next_dir = candidate.normalized()
                    new_throughput = base_throughput * diff_amp_band
                else:
                    if spec_prob <= 0.0 or np.all(spec_amp_band < 1e-6):
                        break
                    spec_dir = reflect(dirn, normal)
                    spec_dir = _jitter_spec(spec_dir)
                    next_dir = spec_dir.normalized()
                    new_throughput = base_throughput * spec_amp_band

            if new_throughput is None or next_dir is None:
                break

            new_throughput = np.nan_to_num(new_throughput, nan=0.0, posinf=0.0, neginf=0.0)
            new_throughput = np.clip(new_throughput, 0.0, 1e6)
            throughput = new_throughput.astype(np.float32)

            if float(np.max(throughput)) < thr:
                break

            bounce += 1
            pos = hit_point + offset + next_dir * (eps * 0.5)
            dirn = next_dir

            if rr_enable and bounce >= rr_start:
                if random.random() > rr_survive:
                    break
                # Keep amplitude-domain IR stable by avoiding 1/p rescaling here.

    if not omit_direct and los_clear(source, receiver, bvh):
        dvec = receiver - source
        dist_direct = dvec.length
        if dist_direct > 0.0:
            incoming = (source - receiver).normalized()
            # Use the same 1/r geometric falloff as bounced arrivals
            amp_scalar = 1.0 / max(dist_direct, recv_r)
            wrote_any = _emit_impulse(band_one * ray_weight, dist_direct, incoming, amp_scalar) or wrote_any

    if not wrote_any:
        suffix = " (omit_direct active)" if omit_direct else ""
        print(f"[AIRT] forward trace produced no contributions (dirs={len(directions)}, bvh={'None' if bvh is None else 'BVH'}){suffix}")
    return ir
def _trace_ir_reverse(context, source, receiver, bvh, obj_map, directions):
    num_channels = 16
    sr = int(context.scene.airt_sr)
    ir_length = int(context.scene.airt_ir_seconds * sr)
    ir = np.zeros((num_channels, ir_length), dtype=np.float32)

    scene = context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    unit_scale = float(getattr(unit_settings, "scale_length", 1.0) or 1.0)
    c = _speed_of_sound_bu(context)
    tol_rad = context.scene.airt_angle_tol_deg * pi / 180.0
    max_order = int(context.scene.airt_max_order)
    rr_enable = bool(context.scene.airt_rr_enable)
    rr_start = int(context.scene.airt_rr_start)
    rr_survive = max(0.05, min(1.0, float(context.scene.airt_rr_p)))
    recv_r_m = max(1e-6, float(context.scene.airt_recv_radius))
    recv_r = recv_r_m / max(unit_scale, 1e-9)
    rough_rad = max(0.0, float(context.scene.airt_spec_rough_deg)) * pi / 180.0
    enable_diffraction = bool(getattr(context.scene, 'airt_enable_diffraction', False))
    diff_samples = int(getattr(context.scene, 'airt_diffraction_samples', 0))
    diff_max_angle = max(0.0, float(getattr(context.scene, 'airt_diffraction_max_deg', 40.0))) * pi / 180.0
    eps = 1e-4

    band_one = np.ones(_NUM_BANDS, dtype=np.float32)
    quick = bool(getattr(context.scene, 'airt_quick_broadband', False))
    num_dirs = max(1, len(directions))
    ray_weight = 1.0 / float(num_dirs)
    omit_direct = bool(getattr(context.scene, 'airt_omit_direct', False))
    thr = float(getattr(context.scene, 'airt_min_throughput', 1e-4))
    pi4 = 4.0 * pi

    def _path_band_profile(band_amp, distance_bu):
        air = _air_attenuation_bands(distance_bu * unit_scale, context)
        profile = np.array(band_amp, dtype=np.float32) * air
        return np.clip(profile, 0.0, 1e6)

    def _ambisonic(direction):
        return encode_ambisonics_3rd_order(_apply_orientation(direction, context))

    def _emit_impulse(band_amp, distance_bu, incoming_dir, amp_scalar):
        if distance_bu <= 0.0:
            return
        band_profile = _path_band_profile(band_amp, distance_bu)
        if quick:
            gain = float(np.mean(band_profile)) if isinstance(band_profile, np.ndarray) else float(band_profile)
            gain = max(gain, 0.0)
            if gain <= 0.0:
                return
            delay = (distance_bu / c) * sr
            ambi = _ambisonic(incoming_dir)
            ambi = apply_near_field_compensation(ambi, distance_bu * unit_scale, recv_r_m)
            if not np.any(np.abs(ambi) > 1e-8):
                ambi = np.zeros(16, dtype=np.float32)
                ambi[0] = 1.0
            add_impulse_simple(ir, ambi, delay, amp_scalar * gain)
            return
        if not np.any(band_profile > 1e-8):
            return
        delay = (distance_bu / c) * sr
        ambi = _ambisonic(incoming_dir)
        ambi = apply_near_field_compensation(ambi, distance_bu * unit_scale, recv_r_m)
        if not np.any(np.abs(ambi) > 1e-8):
            ambi = np.zeros(16, dtype=np.float32)
            ambi[0] = 1.0
        wrote = add_filtered_impulse(ir, ambi, delay, amp_scalar, band_profile, sr)
        if not wrote:
            add_impulse_simple(ir, ambi, delay, amp_scalar)

    def _jitter_spec(direction):
        v = direction.normalized()
        if rough_rad <= 1e-6:
            return v
        if abs(v.x) < 0.5:
            helper = mathutils.Vector((1.0, 0.0, 0.0))
        else:
            helper = mathutils.Vector((0.0, 1.0, 0.0))
        t = v.cross(helper).normalized()
        b = v.cross(t).normalized()
        u = random.random()
        vphi = 2.0 * pi * random.random()
        cos_a = 1.0 - u * (1.0 - cos(rough_rad))
        sin_a = sqrt(max(0.0, 1.0 - cos_a * cos_a))
        return (v * cos_a + t * (sin_a * cos(vphi)) + b * (sin_a * sin(vphi))).normalized()

    def _add_diffraction(hit_point, normal, dir_in, to_target_vec, throughput_band, refl_amp_band, diff_frac_band, path_len_hit, incoming_dir):
        if not enable_diffraction or diff_samples <= 0 or diff_max_angle <= 0.0:
            return
        axis = dir_in.cross(normal)
        if axis.length < 1e-6:
            axis = normal.cross(mathutils.Vector((0.0, 0.0, 1.0)))
            if axis.length < 1e-6:
                axis = mathutils.Vector((1.0, 0.0, 0.0))
        axis.normalize()
        base_dir = to_target_vec.normalized()
        src_radius = max(recv_r, 0.05)
        for _ in range(diff_samples):
            angle = diff_max_angle * random.random()
            sign = -1.0 if random.random() < 0.5 else 1.0
            rot = mathutils.Matrix.Rotation(sign * angle, 3, axis)
            sample_dir = (rot @ base_dir).normalized()
            far = hit_point + sample_dir * 100.0
            if not los_clear(hit_point + normal * eps, far, bvh, eps=eps):
                continue
            got, t_hit, _ = segment_hits_sphere(hit_point, far, source, src_radius)
            if not got:
                continue
            seg_dist = (far - hit_point).length * t_hit
            total_dist = path_len_hit + seg_dist
            if total_dist <= 0.0:
                continue
            angle_factor = np.exp(-angle / max(diff_max_angle, 1e-6))
            diff_gain = np.sqrt(np.clip(diff_frac_band, 0.0, 1.0)) * angle_factor
            band_amp = throughput_band * refl_amp_band * diff_gain
            if not np.any(band_amp > 1e-6):
                continue
            amp_scalar = 1.0 / max(total_dist, recv_r)
            _emit_impulse(band_amp, total_dist, incoming_dir, amp_scalar)
            break

    if bvh is None:
        return ir

    for d in directions:
        first_dir = mathutils.Vector(d)
        if first_dir.length == 0.0:
            continue
        first_dir.normalize()
        pos = receiver.copy()
        dirn = first_dir.copy()
        throughput = band_one.copy()
        path_len = 0.0
        bounce = 0
        incoming_dir = (-first_dir).normalized()

        while bounce < max_order:
            hit, normal, index, dist = bvh.ray_cast(pos + dirn * eps, dirn)
            if hit is None or index is None:
                break

            seg_len = float(dist)
            path_len += seg_len
            hit_point = mathutils.Vector(hit)
            normal = mathutils.Vector(normal)
            if normal.dot(dirn) > 0.0:
                normal = -normal

            hit_obj = obj_map[index] if 0 <= index < len(obj_map) else None
            absorption = get_absorption_spectrum(hit_obj)
            scatter_spec = get_scatter_spectrum(hit_obj)
            transmission_coeff = get_transmission_coeff(hit_obj)
            transmission_spec = np.clip(np.full(_NUM_BANDS, transmission_coeff, dtype=np.float32), 0.0, 1.0)
            refl_spec = np.clip(1.0 - absorption - transmission_spec, 0.0, 1.0)
            if not np.any(refl_spec > 1e-6) and transmission_coeff <= 1e-6:
                break

            spec_frac = np.clip(1.0 - scatter_spec, 0.0, 1.0)
            diff_frac = np.clip(scatter_spec, 0.0, 1.0)
            refl_amp_band = np.sqrt(refl_spec)
            spec_amp_band = refl_amp_band * np.sqrt(np.maximum(spec_frac, 1e-6))
            diff_amp_band = refl_amp_band * np.sqrt(np.maximum(diff_frac, 1e-6))
            trans_amp_band = np.sqrt(transmission_spec)

            to_src = source - hit_point
            dist_src = to_src.length
            if dist_src > 0.0:
                has_los = los_clear(hit_point + normal * eps, source, bvh, eps=eps)
                if has_los:
                    to_src_dir = to_src.normalized()
                    req_dir = reflect(to_src_dir, normal)
                    cosang = max(-1.0, min(1.0, req_dir.dot(dirn)))
                    dtheta = acos(cosang)
                    spec_lobe = np.exp(-(dtheta / max(tol_rad, 1e-6)) ** 2)
                    cos_i = max(0.0, (-dirn).dot(normal))
                    diff_lobe = cos_i / pi
                    total_weight_band = np.clip(spec_frac * spec_lobe + diff_frac * diff_lobe, 0.0, 1.0)
                    if np.any(total_weight_band > 1e-6):
                        band_amp = throughput * refl_amp_band * np.sqrt(total_weight_band)
                        total_dist = path_len + dist_src
                        amp_scalar = 1.0 / max(total_dist, recv_r)
                        _emit_impulse(band_amp, total_dist, incoming_dir, amp_scalar)
                else:
                    _add_diffraction(hit_point, normal, dirn, to_src, throughput * ray_weight, refl_amp_band, diff_frac, path_len, incoming_dir)

            base_throughput = throughput.copy()
            trans_prob = min(0.999, max(0.0, transmission_coeff))
            remaining = max(0.0, 1.0 - trans_prob)
            diff_prob = remaining * float(np.clip(np.mean(diff_frac), 0.0, 1.0))
            spec_prob = max(remaining - diff_prob, 0.0)

            rnd = random.random()
            next_dir = None
            new_throughput = None
            offset = normal * eps

            if trans_prob > 0.0 and rnd < trans_prob:
                next_dir = dirn.normalized()
                new_throughput = base_throughput * trans_amp_band
                offset = -normal * eps
            else:
                rnd -= trans_prob
                if diff_prob > 0.0 and rnd < diff_prob:
                    candidate = None
                    cos_out = 0.0
                    for _ in range(8):
                        cand = cosine_weighted_hemisphere(normal)
                        cos_out = max(0.0, cand.dot(normal))
                        if cos_out > 1e-6:
                            candidate = cand
                            break
                    if candidate is None:
                        break
                    next_dir = candidate.normalized()
                    new_throughput = base_throughput * diff_amp_band
                else:
                    if spec_prob <= 0.0 or np.all(spec_amp_band < 1e-6):
                        break
                    spec_dir = reflect(dirn, normal)
                    spec_dir = _jitter_spec(spec_dir)
                    next_dir = spec_dir.normalized()
                    new_throughput = base_throughput * spec_amp_band

            if new_throughput is None or next_dir is None:
                break

            new_throughput = np.nan_to_num(new_throughput, nan=0.0, posinf=0.0, neginf=0.0)
            new_throughput = np.clip(new_throughput, 0.0, 1e6)
            throughput = new_throughput.astype(np.float32)

            if float(np.max(throughput)) < thr:
                break

            bounce += 1
            pos = hit_point + offset + next_dir * (eps * 0.5)
            dirn = next_dir

            if rr_enable and bounce >= rr_start:
                if random.random() > rr_survive:
                    break
                # Keep amplitude-domain IR stable by avoiding 1/p rescaling here.

    if num_dirs > 0:
        ir /= float(num_dirs)

    if not omit_direct and los_clear(source, receiver, bvh):
        dvec = receiver - source
        dist_direct = dvec.length
        if dist_direct > 0.0:
            incoming = (source - receiver).normalized()
            # Use the same 1/r geometric falloff as bounced arrivals
            amp_scalar = 1.0 / max(dist_direct, recv_r)
            _emit_impulse(band_one, dist_direct, incoming, amp_scalar)

    return ir
def orthonormal_basis(n: mathutils.Vector):
    n = n.normalized()
    if abs(n.x) < 0.5:
        helper = mathutils.Vector((1.0, 0.0, 0.0))
    else:
        helper = mathutils.Vector((0.0, 1.0, 0.0))
    t = n.cross(helper).normalized()
    b = n.cross(t).normalized()
    return t, b, n


def cosine_weighted_hemisphere(normal: mathutils.Vector) -> mathutils.Vector:
    """Cosine-weighted sample about a given normal."""
    u1 = random.random()
    u2 = random.random()
    r = sqrt(u1)
    theta = 2.0 * pi * u2
    x = r * cos(theta)
    y = r * sin(theta)
    z = sqrt(max(0.0, 1.0 - u1))
    t, b, n = orthonormal_basis(normal)
    v = (t * x + b * y + n * z).normalized()
    return v


def fibonacci_sphere(samples):
    points = []
    phi = pi * (3.0 - sqrt(5.0))
    for i in range(samples):
        y = 1.0 - (i / float(samples - 1)) * 2.0
        radius = sqrt(max(0.0, 1.0 - y*y))
        theta = phi * i
        x = cos(theta) * radius
        z = sin(theta) * radius
        points.append((x, y, z))
    return points


def generate_ray_directions(samples):
    if samples <= 0:
        return []
    base = fibonacci_sphere(samples)
    axis = mathutils.Vector((np.random.normal(), np.random.normal(), np.random.normal()))
    if axis.length == 0.0:
        axis = mathutils.Vector((0.0, 0.0, 1.0))
    axis.normalize()
    angle = 2.0 * pi * np.random.random()
    rot = mathutils.Matrix.Rotation(angle, 3, axis)
    jitter_strength = 0.75 / float(max(samples, 1))
    dirs = []
    for p in base:
        v = rot @ mathutils.Vector(p)
        jitter = mathutils.Vector((
            np.random.uniform(-jitter_strength, jitter_strength),
            np.random.uniform(-jitter_strength, jitter_strength),
            np.random.uniform(-jitter_strength, jitter_strength)
        ))
        candidate = v + jitter
        if candidate.length == 0.0:
            candidate = v
        candidate = candidate.normalized()
        dirs.append((float(candidate.x), float(candidate.y), float(candidate.z)))
    random.shuffle(dirs)
    return dirs

# ----------------------------------------------------------------------------
# Ambisonic Encoding (ACN / SN3D)
# ----------------------------------------------------------------------------

def _apply_orientation(direction, context):
    """Map Blender world axes (X right, Y forward, Z up) to AmbiX axes (X front, Y left, Z up),
    then apply a yaw around +Z and optional Z flip.
    Blender -> AmbiX mapping (default): X_a = -Y_b, Y_a = +X_b, Z_a = +Z_b.
    """
    # Blender -> AmbiX basis mapping
    xb, yb, zb = float(direction.x), float(direction.y), float(direction.z)
    xa = -yb
    ya = xb
    za = zb

    # Yaw around +Z (AmbiX frame)
    yaw = float(getattr(context.scene, 'airt_yaw_offset_deg', 0.0)) * pi / 180.0
    cz, sz = cos(yaw), sin(yaw)
    xr = xa * cz - ya * sz
    yr = xa * sz + ya * cz
    zr = za

    # Optional Z flip
    if bool(getattr(context.scene, 'airt_invert_z', False)):
        zr = -zr

    import mathutils
    return mathutils.Vector((xr, yr, zr)).normalized()


def encode_ambisonics_3rd_order(direction):
    """Return 16-channel ACN/SN3D encoding for a unit direction vector.
    Uses elevation (el) where sin(el) = z; azimuth = atan2(y, x).
    SN3D normalization; output in ACN order.
    """
    if lpmv is None:
        return np.zeros(16, dtype=np.float32)
    x, y, z = float(direction.x), float(direction.y), float(direction.z)
    r = sqrt(x*x + y*y + z*z)
    if r == 0:
        return np.zeros(16, dtype=np.float32)
    az = atan2(y, x)
    el = asin(z / r)
    s = sin(el)

    def N_sn3d(l, m):
        m = abs(m)
        from math import factorial
        return sqrt(float(factorial(l - m)) / float(factorial(l + m)))

    sh = np.zeros(16, dtype=np.float32)
    idx = 0
    for l in range(0, 4):
        for m in range(-l, l+1):
            P = float(lpmv(abs(m), l, s))
            if m == 0:
                ylm = N_sn3d(l, 0) * P
            elif m > 0:
                ylm = sqrt(2.0) * N_sn3d(l, m) * cos(m * az) * P
            else:
                ylm = sqrt(2.0) * N_sn3d(l, -m) * sin(-m * az) * P
            sh[idx] = ylm
            idx += 1
    return sh


# ACN order mapping for 3rd order (16 channels)
_ACN_ORDERS = (0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3)


def apply_near_field_compensation(ambi_vec: np.ndarray, distance_m: float, reference_m: float) -> np.ndarray:
    """Apply a lightweight near-field compensation by boosting higher orders for close distances."""
    if reference_m <= 0.0:
        return ambi_vec
    ref = max(reference_m, 1e-3)
    dist = max(float(distance_m), 1e-3)
    if dist >= ref:
        return ambi_vec
    scale = ref / dist
    gains = np.ones_like(ambi_vec, dtype=np.float32)
    for idx, order in enumerate(_ACN_ORDERS):
        if order <= 0:
            continue
        gains[idx] = min(scale ** (0.5 * order), 8.0)
    return ambi_vec * gains

# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

def register():
    register_acoustic_props()
    bpy.utils.register_class(AIRT_PT_Panel)
    bpy.utils.register_class(AIRT_OT_RenderIR)


def unregister():
    bpy.utils.unregister_class(AIRT_OT_RenderIR)
    bpy.utils.unregister_class(AIRT_PT_Panel)
    unregister_acoustic_props()

if __name__ == "__main__":
    register()
