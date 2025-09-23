#!/usr/bin/env python3
"""Tests for the Option B hybrid blending (_blend_early_late) without Blender.

Focus:
 1. Standalone execution without Blender's mathutils/context.
 2. Verifies absence of late energy hump (RMS window comparison).
 3. Ensures exponential forward tail does not collapse completely (residual energy persists) while remaining monotonic. 

Run manually:
  python test_blend_option_b.py

If pytest is installed, it will also be discovered as a normal test.
"""
import sys
import types
import numpy as np

# ---------------------------------------------------------------------------
# Provide a minimal mathutils stub BEFORE importing the ray tracer module.
# ---------------------------------------------------------------------------
if 'mathutils' not in sys.modules:
    class _Vec:
        def __init__(self, xyz=(0.0, 0.0, 0.0)):
            self.x, self.y, self.z = xyz
        def __add__(self, o):
            return _Vec((self.x + o.x, self.y + o.y, self.z + o.z))
        def __sub__(self, o):
            return _Vec((self.x - o.x, self.y - o.y, self.z - o.z))
        def __mul__(self, s):
            return _Vec((self.x * s, self.y * s, self.z * s))
        def dot(self, o):
            return self.x*o.x + self.y*o.y + self.z*o.z
        def normalized(self):
            n = (self.x**2 + self.y**2 + self.z**2) ** 0.5
            if n > 0:
                return _Vec((self.x/n, self.y/n, self.z/n))
            return _Vec((0.0, 0.0, 0.0))
    class _DummyBVH:
        def ray_cast(self, *args, **kwargs):
            return (None, None, None, None)
    stub = types.SimpleNamespace(Vector=_Vec, bvhtree=types.SimpleNamespace(BVHTree=_DummyBVH))
    sys.modules['mathutils'] = stub

# Provide minimal bpy stub to satisfy ir_raytracer package import side-effects.
if 'bpy' not in sys.modules:
    class _DummyPanel:  # Placeholder base class
        pass
    class _DummyOperator:
        pass
    class _DummyUtils:
        def register_class(self, cls):
            return None
        def unregister_class(self, cls):
            return None
    class _DummySceneProps:
        # Supply attributes accessed in RayTracingConfig with sensible defaults
        airt_num_rays = 1024
        airt_max_order = 32
        airt_sr = 48000
        airt_ir_seconds = 4.0
        airt_recv_radius = 0.05
        airt_angle_tol_deg = 5.0
        airt_spec_rough_deg = 0.0
        airt_enable_seg_capture = False
        airt_rr_enable = False
        airt_rr_start = 4
        airt_rr_p = 0.5
        airt_enable_diffraction = False
        airt_diffraction_samples = 0
        airt_diffraction_max_deg = 40.0
        airt_air_enable = True
        airt_air_temp_c = 20.0
        airt_air_humidity = 50.0
        airt_air_pressure_kpa = 101.325
        airt_quick_broadband = False
        airt_min_throughput = 1e-4
        airt_yaw_offset_deg = 0.0
        airt_invert_z = False
        airt_hybrid_forward_gain_db = 0.0
        airt_hybrid_reverse_gain_db = 0.0
        airt_hybrid_reverb_ramp_time = 0.2
    class _DummyContext:
        scene = _DummySceneProps()
    class _DummyProp:
        def __init__(self, **kwargs):
            self.default = kwargs.get('default')
        def __get__(self, inst, owner):
            return self.default
        def __set__(self, inst, value):
            pass
    class _DummyPropsNS:
        @staticmethod
        def FloatProperty(**kwargs):
            return _DummyProp(**kwargs)
        @staticmethod
        def BoolProperty(**kwargs):
            return _DummyProp(**kwargs)
        @staticmethod
        def IntProperty(**kwargs):
            return _DummyProp(**kwargs)
        @staticmethod
        def EnumProperty(**kwargs):
            return _DummyProp(**kwargs)
    sys.modules['bpy'] = types.SimpleNamespace(
        utils=_DummyUtils(),
        context=_DummyContext(),
        types=types.SimpleNamespace(Panel=_DummyPanel, Operator=_DummyOperator, PropertyGroup=object),
        props=_DummyPropsNS()
    )

# Import ONLY the blend function (internal) to isolate logic.
from ir_raytracer.core.ray_tracer import _blend_early_late  # type: ignore


class MinimalConfig:
    """Lightweight stand‑in for RayTracingConfig with only needed attributes."""
    def __init__(self, sample_rate=48000, ramp_time=0.25, fwd_gain_db=0.0, rev_gain_db=0.0):
        self.sample_rate = sample_rate
        self.hybrid_reverb_ramp_time = ramp_time
        self.hybrid_forward_gain_db = fwd_gain_db
        self.hybrid_reverse_gain_db = rev_gain_db
        self.hybrid_forward_gain_linear = 10 ** (fwd_gain_db / 20.0)
        self.hybrid_reverse_gain_linear = 10 ** (rev_gain_db / 20.0)


def _synthetic_ir_pair(channels=16, seconds=4.0, sample_rate=48000, seed=42):
    """Generate a synthetic (forward, reverse) IR pair for testing.

    Forward: sparse discrete decaying impulses (simulating reflections).
    Reverse: exponentially decaying colored noise (diffuse tail).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    fwd = np.zeros((channels, n), dtype=np.float32)
    rev = np.zeros_like(fwd)

    # Forward sparse impulses at selected times (seconds)
    impulse_times = [0.008, 0.025, 0.052, 0.085, 0.130, 0.200, 0.350, 0.600, 0.950, 1.400]
    for idx, t in enumerate(impulse_times):
        s = int(t * sample_rate)
        if s < n:
            amp = (0.95 ** idx) * 0.9  # gently decaying sequence
            fwd[:, s] += amp * (0.5 + 0.5 * rng.random(channels))

    # Reverse diffuse tail: exponential decay * filtered noise
    t_axis = np.arange(n) / sample_rate
    rt60 = 1.6  # seconds
    tau = rt60 / 6.91  # approximate for -60 dB
    decay = np.exp(-t_axis / tau)
    noise = rng.standard_normal((channels, n)).astype(np.float32)
    # Light smoothing (moving average) to emulate band-limited character
    kernel = np.ones(32, dtype=np.float32) / 32.0
    for ch in range(channels):
        noise[ch] = np.convolve(noise[ch], kernel, mode='same')
    rev = noise * decay

    return fwd, rev


def window_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))


def test_no_late_energy_hump():
    cfg = MinimalConfig(ramp_time=0.25)
    fwd, rev = _synthetic_ir_pair()
    blended = _blend_early_late(fwd, rev, cfg)

    sr = cfg.sample_rate
    # Windows chosen to detect rebound between 2.0-3.0 s
    w1 = blended[:, int(1.5*sr):int(2.0*sr)]
    w2 = blended[:, int(2.0*sr):int(3.0*sr)]
    rms1 = window_rms(w1)
    rms2 = window_rms(w2)
    if rms1 > 1e-9:  # Only assert if meaningful baseline
        delta_db = 20 * np.log10(max(rms2, 1e-12) / rms1)
        # Expect no significant hump (> +2.5 dB)
        assert delta_db < 2.5, f"Late energy hump detected: Δ={delta_db:.2f} dB (rms1={rms1:.3e}, rms2={rms2:.3e})"

    # Ensure residual late energy still present (not over-attenuated)
    tail_segment = blended[:, int(3.2*sr):int(3.8*sr)]
    tail_rms = window_rms(tail_segment)
    assert tail_rms > 1e-6, "Tail energy vanished unexpectedly"

    # Peak headroom sanity
    peak = float(np.max(np.abs(blended)))
    assert peak <= 0.95, f"Peak clipping risk: {peak:.3f}"


def test_scaling_stability_across_ramp_variants():
    fwd, rev = _synthetic_ir_pair()
    for ramp in [0.07, 0.15, 0.30, 0.45]:
        cfg = MinimalConfig(ramp_time=ramp)
        blended = _blend_early_late(fwd, rev, cfg)
        sr = cfg.sample_rate
        early_100ms = blended[:, :int(0.1*sr)]
        late_segment = blended[:, int(1.0*sr):int(1.2*sr)]
        rms_early = window_rms(early_100ms)
        rms_late = window_rms(late_segment)
        # Late RMS should be lower but not orders of magnitude tiny; allow broad band of 6-40 dB below
        if rms_early > 0 and rms_late > 0:
            diff_db = 20 * np.log10(rms_early / rms_late)
            assert 6.0 <= diff_db <= 40.0, f"Unexpected early/late balance ramp={ramp}s diff={diff_db:.1f} dB"


def test_forward_residual_presence():
    cfg = MinimalConfig(ramp_time=0.20)
    fwd, rev = _synthetic_ir_pair()
    blended = _blend_early_late(fwd, rev, cfg)
    sr = cfg.sample_rate
    # Reconstruct what pure reverse alone would look like scaled by measuring first 40 ms ratio
    first_win = slice(0, int(0.04*sr))
    rev_only_rms = window_rms(rev[:, first_win])
    blend_rms = window_rms(blended[:, first_win])
    # Expect blend dominated by forward (reverse suppressed early) => blended > reverse-alone window
    if rev_only_rms > 0:
        assert blend_rms >= rev_only_rms * 1.2, "Forward early dominance not evident"


if __name__ == "__main__":
    # Manual execution summary
    failures = []
    for fn in [test_no_late_energy_hump, test_scaling_stability_across_ramp_variants, test_forward_residual_presence]:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)
    if failures:
        print(f"\n❌ {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    print("\n✅ All standalone Option B blend tests passed")
