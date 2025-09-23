#!/usr/bin/env python3
"""
Extract acoustic analysis data from impulse response WAV file.
Run this script and paste the output for analysis.
"""
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import hilbert
import sys


def _to_float(sig: np.ndarray) -> np.ndarray:
    """Convert integer PCM arrays to float in [-1,1]."""
    if sig.dtype.kind in ('i', 'u'):
        # Normalize by max possible of dtype
        info = np.iinfo(sig.dtype)
        sig = sig.astype(np.float64) / max(abs(info.min), info.max)
    else:
        sig = sig.astype(np.float64)
    return sig


def _schroeder_energy_db(power: np.ndarray) -> np.ndarray:
    """Compute Schroeder decay (energy) from per-sample power (already squared, non-negative).

    power: 1D array of instantaneous energy proxy (e.g. sum of channel^2 per sample).
    Returns dB curve normalized to 0 dB at start (peak integrated energy).
    """
    if power.ndim != 1:
        raise ValueError("power must be 1D array")
    if not np.any(power > 0):
        return np.full_like(power, -120.0, dtype=np.float64)
    integ = np.cumsum(power[::-1])[::-1]
    integ /= np.max(integ)
    return 10 * np.log10(np.maximum(integ, 1e-12))


def _linear_fit_rt(db_curve: np.ndarray, sr: int, t0: float, t1: float) -> float:
    """Fit a straight line to dB curve between t0, t1 (seconds) and extrapolate RT60.
    Returns np.nan if slope not negative enough."""
    n0 = int(t0 * sr)
    n1 = int(t1 * sr)
    n1 = min(len(db_curve)-1, max(n0+10, n1))
    seg = db_curve[n0:n1]
    if seg.size < 32:
        return np.nan
    t = np.linspace(t0, t0 + (seg.size-1)/sr, seg.size)
    A = np.vstack([t, np.ones_like(t)]).T
    m, c = np.linalg.lstsq(A, seg, rcond=None)[0]
    if m >= -1e-3:  # too flat or rising
        return np.nan
    rt60 = -60.0 / m
    return rt60

def analyze_ir(wav_file):
    """Extract key acoustic metrics from IR WAV file."""
    
    # Read WAV file
    try:
        sr, data = wavfile.read(wav_file)
        print(f"Sample Rate: {sr} Hz")
        print(f"Duration: {len(data)/sr:.2f} seconds")
        print(f"Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
    except Exception as e:
        print(f"Error reading WAV: {e}")
        return
    
    # Multi-channel handling: interpret shape (samples, channels)
    if data.ndim == 1:
        channels = 1
        arr = _to_float(data)
    else:
        channels = data.shape[1]
        arr = _to_float(data)

    # Ambisonic handling: treat channels as higher-order components. For analysis:
    #  - Use W (channel 0) for Hilbert-style amplitude envelope (pressure-like).
    #  - Use per-sample energy proxy: sum(channel^2) (no destructive cancellation).
    if channels == 1:
        w_channel = arr
        channel_power = arr**2
    else:
        w_channel = arr[:, 0]
        channel_power = np.sum(arr**2, axis=1)  # energy proxy

    # Global peak across all channels (amplitude) for normalization reference
    global_peak = np.max(np.abs(arr)) or 1.0
    w_norm = w_channel / global_peak
    # Amplitude RMS for diagnostic (not used directly in decay integration)
    rms_amplitude = np.sqrt(np.mean(arr**2))
    
    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(w_norm)
    envelope = np.abs(analytic_signal)
    
    # Convert to dB
    envelope_db = 20 * np.log10(np.maximum(envelope, 1e-6))
    
    # Time axis
    time_axis = np.arange(len(w_norm)) / sr
    
    # Find peak (direct path)
    peak_idx = np.argmax(envelope)
    peak_time = peak_idx / sr
    
    print(f"\n--- ACOUSTIC ANALYSIS ---")
    print(f"Peak at: {peak_time*1000:.1f} ms")
    print(f"Peak level: {envelope_db[peak_idx]:.1f} dB (relative to global amplitude peak)")
    
    # RT60 calculation with enhanced diagnostics
    peak_level = envelope_db[peak_idx]
    rt60_target = peak_level - 60
    
    print(f"RT60 Diagnostics:")
    print(f"  Peak level: {peak_level:.1f} dB")
    print(f"  Target level (-60dB): {rt60_target:.1f} dB")
    
    # Find minimum level reached
    min_level = np.min(envelope_db[peak_idx:])
    print(f"  Minimum level reached: {min_level:.1f} dB")
    print(f"  Decay range: {peak_level - min_level:.1f} dB")
    
    # Analyze early decay characteristics
    early_samples = [
        int(0.05 * sr), int(0.1 * sr), int(0.2 * sr), 
        int(0.3 * sr), int(0.4 * sr), int(0.5 * sr)
    ]
    
    early_levels = []
    for sample_idx in early_samples:
        if sample_idx < len(envelope_db):
            early_levels.append(envelope_db[sample_idx])
    
    if len(early_levels) >= 3:
        early_decay_rate = (early_levels[0] - early_levels[-1]) / 0.45  # dB per second over 0.45s
        print(f"  Early decay rate: {early_decay_rate:.1f} dB/s")
        
        # Estimate RT60 from early decay rate if steady
        if early_decay_rate > 10:  # At least 10 dB/s decay
            estimated_rt60 = 60 / early_decay_rate
            print(f"  RT60 from early decay: {estimated_rt60:.2f} seconds")
    
    # Find RT60 point
    decay_start_idx = peak_idx
    rt60_idx = None
    for i in range(decay_start_idx, len(envelope_db)):
        if envelope_db[i] <= rt60_target:
            rt60_idx = i
            break
    
    if rt60_idx:
        rt60_time = (rt60_idx - peak_idx) / sr
        print(f"RT60: {rt60_time:.2f} seconds")
    else:
        # Try alternative RT60 methods if direct -60dB not found
        print(f"RT60: Did not reach -60dB")
        
        # Try RT30 extrapolation (common method)
        rt30_target = peak_level - 30
        rt30_idx = None
        for i in range(decay_start_idx, len(envelope_db)):
            if envelope_db[i] <= rt30_target:
                rt30_idx = i
                break
        
        if rt30_idx:
            rt30_time = (rt30_idx - peak_idx) / sr
            rt60_estimated = rt30_time * 2  # RT60 = 2 * RT30
            print(f"RT60 (from RT30): {rt60_estimated:.2f} seconds")
        else:
            # Try RT20 extrapolation
            rt20_target = peak_level - 20
            rt20_idx = None
            for i in range(decay_start_idx, len(envelope_db)):
                if envelope_db[i] <= rt20_target:
                    rt20_idx = i
                    break
            
            if rt20_idx:
                rt20_time = (rt20_idx - peak_idx) / sr
                rt60_estimated = rt20_time * 3  # RT60 = 3 * RT20
                print(f"RT60 (from RT20): {rt60_estimated:.2f} seconds")
            else:
                # Check if decay is simply too slow/flat
                level_at_1s = envelope_db[min(peak_idx + sr, len(envelope_db) - 1)]
                decay_1s = peak_level - level_at_1s
                print(f"  Decay in first 1s: {decay_1s:.1f} dB")
                
                if decay_1s < 20:
                    print(f"RT60: >3 seconds (very slow decay - may need material adjustment)")
                else:
                    print(f"RT60: Cannot calculate (insufficient decay)")
    
    # Sample points for decay curve analysis
    print(f"\n--- DECAY CURVE DATA ---")
    sample_times = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0]
    
    for t in sample_times:
        if t < len(w_norm)/sr:
            idx = int(t * sr)
            if idx < len(envelope_db):
                level = envelope_db[idx]
                print(f"t={t:.2f}s: {level:.1f} dB (Hilbert W)")

    # --- Improved decay using energy-based Schroeder integration (sum of channel powers) ---
    sch_db = _schroeder_energy_db(channel_power)
    print(f"\n--- IMPROVED DECAY (Schroeder, channel energy sum) ---")
    for t in sample_times:
        if t < len(channel_power)/sr:
            idx = int(t * sr)
            print(f"t={t:.2f}s: {sch_db[idx]:.1f} dB (Schroeder)")

    # Late rebound diagnostic (1.5-2.0 vs 2.0-3.0 s RMS)
    def window_rms(sig, a, b):
        a_i = int(a*sr); b_i = int(min(b*sr, len(sig)))
        if b_i <= a_i:
            return 0.0
        seg = sig[a_i:b_i]
        return float(np.sqrt(np.mean(seg**2) + 1e-18))
    # Use energy proxy for late window comparison: RMS over sqrt(power)
    amp_equiv = np.sqrt(np.maximum(channel_power, 0.0))
    r1 = window_rms(amp_equiv, 1.5, 2.0)
    r2 = window_rms(amp_equiv, 2.0, 3.0)
    if r1 > 0 and r2 > 0:
        late_delta_db = 20*np.log10(r2/r1)
        print(f"Late window Δ (2.0-3.0 vs 1.5-2.0): {late_delta_db:+.2f} dB")
        if late_delta_db > 2.5:
            print("  WARNING: Late energy increase detected (possible hump)")
    else:
        print("Late window Δ: insufficient energy to evaluate")

    # Schroeder-based RT60 estimates (two segments)
    rt60_est1 = _linear_fit_rt(sch_db, sr, 0.1, 0.4)
    rt60_est2 = _linear_fit_rt(sch_db, sr, 0.4, 1.0)
    if not np.isnan(rt60_est1):
        print(f"RT60 (Schroeder 0.1-0.4s): {rt60_est1:.2f} s")
    if not np.isnan(rt60_est2):
        print(f"RT60 (Schroeder 0.4-1.0s): {rt60_est2:.2f} s")
    if (not np.isnan(rt60_est1)) and (not np.isnan(rt60_est2)):
        diff = abs(rt60_est1 - rt60_est2)
        avg = 0.5*(rt60_est1 + rt60_est2)
        if avg > 0:
            var_pct = 100*diff/avg
            print(f"RT60 segment variance: {var_pct:.1f}%")
    
    # Energy decay analysis (channel energy sum consistent with Schroeder)
    print(f"\n--- ENERGY ANALYSIS (Channel Energy Sum) ---")
    windows = [(0, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    for start_t, end_t in windows:
        a = int(start_t * sr); b = int(min(end_t * sr, len(channel_power)))
        if b <= a:
            continue
        e_sum = np.sum(channel_power[a:b])
        e_db = 10*np.log10(max(e_sum, 1e-18))
        print(f"{start_t:.2f}-{end_t:.1f}s: {e_db:.1f} dB energy")

    # Consistency note (Hilbert W may differ due to directional nulls)
    print("Note: Hilbert(W) vs Schroeder(energy) divergence > 6-10 dB at late times can be normal for HOA; rely on Schroeder for RT metrics.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_ir.py <ir_file.wav>")
        print("Then copy/paste the output for analysis")
    else:
        analyze_ir(sys.argv[1])