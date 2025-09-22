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
    
    # RT60 calculation (find -60dB point)
    peak_level = envelope_db[peak_idx]
    rt60_target = peak_level - 60
    
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
        print("RT60: >4 seconds (did not reach -60dB)")
    
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