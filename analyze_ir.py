#!/usr/bin/env python3
"""
Extract acoustic analysis data from impulse response WAV file.
Run this script and paste the output for analysis.
"""
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import hilbert
import sys

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
    
    # Use first channel (W component for ambisonic)
    if len(data.shape) > 1:
        ir = data[:, 0].astype(np.float64)
    else:
        ir = data.astype(np.float64)
    
    # Normalize
    ir = ir / np.max(np.abs(ir))
    
    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(ir)
    envelope = np.abs(analytic_signal)
    
    # Convert to dB
    envelope_db = 20 * np.log10(np.maximum(envelope, 1e-6))
    
    # Time axis
    time_axis = np.arange(len(ir)) / sr
    
    # Find peak (direct path)
    peak_idx = np.argmax(envelope)
    peak_time = peak_idx / sr
    
    print(f"\n--- ACOUSTIC ANALYSIS ---")
    print(f"Peak at: {peak_time*1000:.1f} ms")
    print(f"Peak level: {envelope_db[peak_idx]:.1f} dB")
    
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
        if t < len(ir)/sr:
            idx = int(t * sr)
            if idx < len(envelope_db):
                level = envelope_db[idx]
                print(f"t={t:.2f}s: {level:.1f} dB")
    
    # Energy decay analysis
    print(f"\n--- ENERGY ANALYSIS ---")
    
    # Calculate energy in different time windows
    windows = [(0, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    
    for start_t, end_t in windows:
        start_idx = int(start_t * sr)
        end_idx = int(min(end_t * sr, len(ir)))
        
        if end_idx > start_idx:
            window_energy = np.sum(ir[start_idx:end_idx]**2)
            window_energy_db = 10 * np.log10(max(window_energy, 1e-12))
            print(f"{start_t:.2f}-{end_t:.1f}s: {window_energy_db:.1f} dB energy")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_ir.py <ir_file.wav>")
        print("Then copy/paste the output for analysis")
    else:
        analyze_ir(sys.argv[1])