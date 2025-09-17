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

# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------

def register_acoustic_props():
    bpy.types.Object.absorption = bpy.props.FloatProperty(
        name="Absorption",
        description="Wideband absorption coefficient (0 = reflective, 1 = fully absorbent)",
        default=0.2,
        min=0.0,
        max=1.0
    )
    bpy.types.Object.scatter = bpy.props.FloatProperty(
        name="Scatter",
        description="Surface scattering (0 = purely specular, 1 = fully diffuse/cosine)",
        default=0.35,
        min=0.0,
        max=1.0
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
    # Orientation controls (to match downstream decoder conventions)
    scene.airt_yaw_offset_deg = bpy.props.FloatProperty(name="Yaw offset (deg)", default=0.0, min=-180.0, max=180.0)
    scene.airt_invert_z = bpy.props.BoolProperty(name="Flip Z (up/down)", default=False)
    scene.airt_calibrate_direct = bpy.props.BoolProperty(name="Calibrate direct (1/r)", default=True)
    # Air absorption (frequency-dependent)
    scene.airt_air_enable = bpy.props.BoolProperty(name="Air absorption (freq)", default=True)
    scene.airt_air_temp_c = bpy.props.FloatProperty(name="Air temp (deg C)", default=20.0, min=-30.0, max=50.0)
    scene.airt_air_humidity = bpy.props.FloatProperty(name="Rel humidity (%)", default=50.0, min=0.0, max=100.0)
    scene.airt_air_pressure_kpa = bpy.props.FloatProperty(name="Air pressure (kPa)", default=101.325, min=80.0, max=110.0)


def unregister_acoustic_props():
    for attr in ("absorption", "scatter", "is_acoustic_source", "is_acoustic_receiver"):
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    scene = bpy.types.Scene
    for k in (
        "airt_num_rays","airt_passes","airt_max_order","airt_sr","airt_ir_seconds",
        "airt_angle_tol_deg","airt_wav_subtype","airt_seed","airt_recv_radius",
        "airt_trace_mode","airt_rr_enable","airt_rr_start","airt_rr_p",
        "airt_spec_rough_deg","airt_enable_seg_capture",
        "airt_yaw_offset_deg","airt_invert_z","airt_calibrate_direct",
        "airt_air_enable","airt_air_temp_c","airt_air_humidity","airt_air_pressure_kpa"
    ):
        if hasattr(scene, k):
            delattr(scene, k)

# ----------------------------------------------------------------------------
# UI Panel
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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
            col.prop(obj, "absorption")
            col.prop(obj, "scatter")
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
        col.prop(context.scene, "airt_yaw_offset_deg")
        col.prop(context.scene, "airt_invert_z")
        col.prop(context.scene, "airt_calibrate_direct")
        col.prop(context.scene, "airt_air_enable")
        if context.scene.airt_air_enable:
            col.prop(context.scene, "airt_air_temp_c")
            col.prop(context.scene, "airt_air_humidity")
            col.prop(context.scene, "airt_air_pressure_kpa")
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
        directions = fibonacci_sphere(num_rays)
        bvh, obj_map = build_bvh(context)

        ir = None
        for i in range(passes):
            if scene.airt_seed:
                seed = int(scene.airt_seed) + i
                random.seed(seed)
                np.random.seed(seed)
            ir_i = trace_ir(context, source, receiver, bvh=bvh, obj_map=obj_map, directions=directions)
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


def _air_kernel_for_distance(distance: float, sr: int, context) -> np.ndarray:
    """Design a short low-pass kernel that approximates frequency-dependent air loss
    for a given path length. We match the magnitude at 8 kHz using a biquad low-pass
    (Q~0.707) and return its impulse response (length 8)."""
    import numpy as _np
    if distance <= 0.0 or not bool(getattr(context.scene, 'airt_air_enable', True)):
        return _np.array([1.0], dtype=_np.float32)
    T = float(getattr(context.scene, 'airt_air_temp_c', 20.0))
    RH = float(getattr(context.scene, 'airt_air_humidity', 50.0))
    Pk = float(getattr(context.scene, 'airt_air_pressure_kpa', 101.325))
    f_ref = 8000.0
    alpha_dbpm = iso9613_alpha_dbpm(f_ref, T, RH, Pk)
    # Target gain at f_ref for this path length
    target_gain = 10.0 ** (-(alpha_dbpm * distance) / 20.0)
    target_gain = float(max(1e-6, min(1.0, target_gain)))

    # RBJ biquad low-pass, solve for fc s.t. |H(f_ref)|~target_gain
    def biquad_coeffs(fc, fs, Q=0.707):
        from math import sin, cos, pi
        w0 = 2.0 * pi * (fc / fs)
        alpha = sin(w0) / (2.0 * Q)
        cw = cos(w0)
        b0 = (1.0 - cw) * 0.5
        b1 = 1.0 - cw
        b2 = (1.0 - cw) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
        return b0, b1, b2, a1, a2

    def biquad_mag(b0,b1,b2,a1,a2,f,fs):
        from math import cos, sin, pi
        w = 2.0 * pi * (f / fs)
        cw = cos(w); sw = sin(w)
        # H(e^jw) magnitude
        num_r = b0 + b1 * cos(w) + b2 * cos(2*w)
        num_i = b1 * sin(w) + b2 * sin(2*w)
        den_r = 1.0 + a1 * cos(w) + a2 * cos(2*w)
        den_i = a1 * sin(w) + a2 * sin(2*w)
        num = num_r*num_r + num_i*num_i
        den = den_r*den_r + den_i*den_i
        return _np.sqrt(max(1e-20, num / max(1e-20, den)))

    # Binary search fc
    fs = float(sr)
    lo, hi = 50.0, min(f_ref, 0.45*fs)
    b0=b1=b2=a1=a2=0.0
    for _ in range(30):
        fc = 0.5 * (lo + hi)
        b0,b1,b2,a1,a2 = biquad_coeffs(fc, fs)
        mag = biquad_mag(b0,b1,b2,a1,a2, f_ref, fs)
        if mag > target_gain:
            # Need lower cutoff to reduce HF more
            hi = fc
        else:
            lo = fc
    # Generate short impulse response (length 8)
    L = 8
    x0 = 1.0
    y = _np.zeros(L, dtype=_np.float32)
    x1=x2=y1=y2=0.0
    for n in range(L):
        x = x0 if n == 0 else 0.0
        y_n = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[n] = y_n
        x2, x1 = x1, x
        y2, y1 = y1, y_n
    # Normalize DC (sum of impulse) to 1 so overall gain at low freq is unchanged
    s = float(_np.sum(y))
    if s > 1e-9:
        y /= s
    return y


# ----------------------------------------------------------------------------

def _speed_of_sound_bu(context):
    """Convert physical speed of sound to Blender units using the scene scale."""
    scene = getattr(context, "scene", None) or bpy.context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    scale_length = getattr(unit_settings, "scale_length", 1.0) if unit_settings else 1.0
    unit_scale = float(scale_length or 1.0)
    return 343.0 / max(unit_scale, 1e-9)


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
    # Original 2-tap fractional delay (kept for fallback)
    n = int(np.floor(delay_samples))
    frac = float(delay_samples - n)
    if 0 <= n < ir.shape[1]:
        ir[:, n] += ambi_vec * amp * (1.0 - frac)
    if 0 <= n + 1 < ir.shape[1]:
        ir[:, n + 1] += ambi_vec * amp * frac


def add_impulse_air(ir, ambi_vec, delay_samples, amp, distance, context, sr):
    """Fractional-delay impulse with optional air-absorption kernel per path length.
    Uses a short low-pass kernel to approximate frequency-dependent air loss."""
    kern = _air_kernel_for_distance(float(distance), int(sr), context)
    n0 = int(np.floor(delay_samples))
    frac = float(delay_samples - n0)
    # Split fractional delay into two starting indices, each convolved with kernel
    for base, w in ((n0, 1.0 - frac), (n0 + 1, frac)):
        if w <= 0.0:
            continue
        start = base
        for k, kv in enumerate(kern):
            idx = start + k
            if 0 <= idx < ir.shape[1]:
                ir[:, idx] += ambi_vec * (amp * w * kv)
    return


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
        directions = fibonacci_sphere(num_rays)
    mode = context.scene.airt_trace_mode
    if mode == 'FORWARD':
        return _trace_ir_forward(context, source, receiver, bvh, obj_map, directions)
    else:
        return _trace_ir_reverse(context, source, receiver, bvh, obj_map, directions)


def _trace_ir_forward(context, source, receiver, bvh, obj_map, directions):
    num_channels = 16
    sr = context.scene.airt_sr
    ir_length = int(context.scene.airt_ir_seconds * sr)
    ir = np.zeros((num_channels, ir_length), dtype=np.float32)

    scene = context.scene
    unit_settings = getattr(scene, "unit_settings", None)
    unit_scale = float(getattr(unit_settings, "scale_length", 1.0) or 1.0)
    c = _speed_of_sound_bu(context)
    max_bounces = context.scene.airt_max_order
    tol_rad = context.scene.airt_angle_tol_deg * pi / 180.0
    recv_r_m = max(1e-6, float(context.scene.airt_recv_radius))
    recv_r = recv_r_m / max(unit_scale, 1e-9)
    rr_enable = bool(context.scene.airt_rr_enable)
    rr_start = int(context.scene.airt_rr_start)
    rr_p = float(context.scene.airt_rr_p)
    seg_capture = bool(context.scene.airt_enable_seg_capture)
    rough_rad = max(0.0, float(context.scene.airt_spec_rough_deg)) * pi / 180.0
    eps = 1e-4

    def refl_amp(absorb):
        # Absorption is an ENERGY coefficient; convert to AMPLITUDE reflectivity
        a = max(0.0, min(1.0, float(absorb)))
        return sqrt(1.0 - a)

    # Direct path (source -> receiver)
    if los_clear(source, receiver, bvh):
        dvec = receiver - source
        dist = dvec.length
        if dist > 0:
            incoming = (source - receiver).normalized()  # DoA at receiver
            ambi = encode_ambisonics_3rd_order(_apply_orientation(incoming, context))
            delay = (dist / c) * sr
            amp = 1.0 / max(dist, recv_r)
            add_impulse_air(ir, ambi, delay, amp, dist, context, sr)

    for d in directions:
        dirn = mathutils.Vector(d).normalized()
        pos = source.copy()
        path_len = 0.0
        refl_prod = 1.0  # product of per-bounce reflection amplitude (sqrt(1-alpha))
        bounce = 0
        while True:
            if bvh is None or bounce > max_bounces:
                break

            # Cast to next surface
            hit, normal, index, dist = bvh.ray_cast(pos + dirn * eps, dirn)
            if hit is None or index is None:
                # Optional capture along a far segment to catch near misses
                if seg_capture:
                    far = pos + dirn * 100.0
                    got, t_hit, _ = segment_hits_sphere(pos, far, receiver, recv_r)
                    if got:
                        seg_len = (far - pos).length * t_hit
                        total_dist = path_len + seg_len
                        incoming = (-dirn).normalized()
                        ambi = encode_ambisonics_3rd_order(_apply_orientation(incoming, context))
                        # Geometric acceptance ~ (pi R^2) / (4 pi r^2) and pressure 1/r => ~ R^2 / (4 r^3)
                        amp = refl_prod * (recv_r * recv_r) / (4.0 * max(total_dist, recv_r)**3)
                        delay = (total_dist / c) * sr
                        add_impulse_air(ir, ambi, delay, amp, total_dist, context, sr)
                break

            if normal.dot(dirn) > 0.0:
                normal = -normal

            seg_len = float(dist)
            d_to_hit = path_len + seg_len
            next_pos = hit

            # Segment capture along pos->hit (improves early density)
            if seg_capture:
                got, t_hit, _ = segment_hits_sphere(pos, next_pos, receiver, recv_r)
                if got:
                    partial_len = seg_len * t_hit
                    total_dist = path_len + partial_len
                    incoming = (-dirn).normalized()
                    ambi = encode_ambisonics_3rd_order(_apply_orientation(incoming, context))
                    # Geometric acceptance ~ (pi R^2) / (4 pi r^2) and pressure 1/r => ~ R^2 / (4 r^3)
                    amp = refl_prod * (recv_r * recv_r) / (4.0 * max(total_dist, recv_r)**3)
                    delay = (total_dist / c) * sr
                    add_impulse_air(ir, ambi, delay, amp, total_dist, context, sr)

            # Path connection from hit to receiver (LOS)
            hit_obj = obj_map[index] if 0 <= index < len(obj_map) else None
            absorb = float(getattr(hit_obj, 'absorption', 0.2)) if hit_obj else 0.2
            scatter = float(getattr(hit_obj, 'scatter', 0.0)) if hit_obj else 0.0
            r_amp = refl_amp(absorb)

            to_rcv = (receiver - hit)
            dist_rcv = to_rcv.length
            if dist_rcv > 0 and los_clear(hit + normal * eps, receiver, bvh, eps=eps):
                to_rcv_dir = to_rcv.normalized()
                # Specular lobe weight (dimensionless gain)
                req_out = reflect(dirn, normal)
                cosang = max(-1.0, min(1.0, req_out.dot(to_rcv_dir)))
                dtheta = acos(cosang)
                weight_spec = (1.0 - scatter) * np.exp(-(dtheta / max(tol_rad, 1e-6))**2)
                # Diffuse lobe
                cos_i = max(0.0, (-dirn).dot(normal))
                cos_o = max(0.0, to_rcv_dir.dot(normal))
                weight_diff = scatter * (cos_i * cos_o / pi)
                total_weight = max(0.0, min(1.0, weight_spec + weight_diff))
                if total_weight > 1e-6:
                    total_dist = d_to_hit + dist_rcv
                    incoming = (hit - receiver).normalized()
                    ambi = encode_ambisonics_3rd_order(_apply_orientation(incoming, context))
                    # Treat lobe weight as a POWER factor; convert to amplitude via sqrt
                    amp = (refl_prod * r_amp * sqrt(total_weight)) / max(total_dist, recv_r)
                    delay = (total_dist / c) * sr
                    add_impulse_air(ir, ambi, delay, amp, total_dist, context, sr)

            # Update accumulators after the hit for the next bounce
            path_len = d_to_hit
            refl_prod *= r_amp
            if abs(refl_prod) < 1e-8:
                break

            # Russian roulette termination to allow large bounce counts (unbiased via 1/p)
            if rr_enable and bounce >= rr_start:
                if random.random() > rr_p:
                    break
                # NOTE: no 1/p compensation in amplitude-domain IR to avoid large late spikes

            # Sample outgoing direction
            spec_dir = reflect(dirn, normal)
            # Apply micro-roughness jitter to specular direction
            if rough_rad > 1e-6:
                v = spec_dir.normalized()
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
                jittered = (v * cos_a + t * (sin_a * cos(vphi)) + b * (sin_a * sin(vphi))).normalized()
                spec_dir = jittered

            if scatter <= 1e-6:
                dirn = spec_dir
            elif scatter >= 1.0 - 1e-6:
                dirn = cosine_weighted_hemisphere(normal)
            else:
                if random.random() < scatter:
                    dirn = cosine_weighted_hemisphere(normal)
                else:
                    dirn = spec_dir

            pos = hit + normal * eps
            bounce += 1

    # Keep absolute scale; downstream convolvers can normalize if needed.
    return ir


def _trace_ir_reverse(context, source, receiver, bvh, obj_map, directions):
    # Existing reverse implementation (specular/diffuse with LOS-to-source)
    num_channels = 16  # 3rd order HOA (ACN)
    sr = context.scene.airt_sr
    ir_length = int(context.scene.airt_ir_seconds * sr)
    ir = np.zeros((num_channels, ir_length), dtype=np.float32)

    c = _speed_of_sound_bu(context)
    tol_rad = context.scene.airt_angle_tol_deg * pi / 180.0
    max_order = context.scene.airt_max_order
    eps = 1e-4

    # Direct path (order 0)
    if los_clear(source, receiver, bvh):
        dvec = source - receiver
        dist = dvec.length
        if dist > 0:
            incoming_dir = dvec.normalized()
            ambi = encode_ambisonics_3rd_order(_apply_orientation(incoming_dir, context))
            delay = (dist / c) * sr
            amp = 1.0 / max(dist, 1e-6)
            add_impulse_air(ir, ambi, delay, amp, dist, context, sr)

    for d in directions:
        first_dir = mathutils.Vector(d).normalized()  # from receiver outward
        pos = receiver.copy()
        dirn = first_dir.copy()
        path_len = 0.0
        energy = 1.0

        for order in range(1, max_order + 1):
            if bvh is None:
                break
            hit, normal, index, dist = bvh.ray_cast(pos + dirn * eps, dirn)
            if hit is None or index is None:
                break
            if normal.dot(dirn) > 0.0:
                normal = -normal

            path_len += float(dist)
            hit_obj = obj_map[index] if 0 <= index < len(obj_map) else None
            absorb = float(getattr(hit_obj, 'absorption', 0.2)) if hit_obj else 0.2
            scatter = float(getattr(hit_obj, 'scatter', 0.0)) if hit_obj else 0.0
            energy *= max(0.0, 1.0 - absorb)

            # Contribution to source from this bounce (specular and diffuse)
            to_src = (source - hit)
            dist_src = to_src.length
            if dist_src > 0 and los_clear(hit + normal * eps, source, bvh, eps=eps):
                to_src_dir = to_src.normalized()
                req_dir = reflect(to_src_dir, normal)
                cosang = max(-1.0, min(1.0, req_dir.dot(dirn)))
                dtheta = acos(cosang)
                weight_spec = (1.0 - scatter) * np.exp(-(dtheta / max(tol_rad, 1e-6))**2)
                cos_i = max(0.0, (-dirn).dot(normal))
                cos_o = max(0.0, to_src_dir.dot(normal))
                weight_diff = scatter * (cos_i * cos_o / pi)

                total_weight = weight_spec + weight_diff
                if total_weight > 1e-6:
                    total_dist = path_len + dist_src
                    incoming_dir = -first_dir
                    ambi = encode_ambisonics_3rd_order(_apply_orientation(incoming_dir, context))
                    delay = (total_dist / c) * sr
                    amp = (energy * total_weight) / max(total_dist, 1e-6)
                    add_impulse_air(ir, ambi, delay, amp, total_dist, context, sr)

            # Continue with a scattered/specular mixture direction
            spec_dir = reflect(dirn, normal)
            if scatter <= 1e-6:
                dirn = spec_dir
            elif scatter >= 1.0 - 1e-6:
                dirn = cosine_weighted_hemisphere(normal)
            else:
                rnd_dir = cosine_weighted_hemisphere(normal)
                blended = (spec_dir * (1.0 - scatter) + rnd_dir * scatter)
                dirn = blended.normalized()

            pos = hit + normal * eps
            if energy < 1e-6:
                break

    # Do not normalize here; preserve direct-to-reverb ratio
    return ir



# ----------------------------------------------------------------------------
# Sampling on the sphere
# ----------------------------------------------------------------------------

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
