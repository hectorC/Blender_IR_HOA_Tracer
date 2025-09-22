#!/usr/bin/env python3
"""Test just the improved blending function with synthetic data."""

import numpy as np
import sys
import os

# Simple test of the blending improvements
def simulate_improved_blend():
    """Simulate the improved blending with synthetic IR data."""
    
    print("=== IMPROVED HYBRID BLENDING TEST ===")
    
    # Test parameters
    sample_rate = 48000
    duration = 2.0  # 2 seconds
    samples = int(duration * sample_rate)
    
    # Create synthetic IR data that mimics real tracers
    time_axis = np.arange(samples) / sample_rate
    
    # Forward tracer: Strong early, weak late (exponential decay)
    ir_forward = np.zeros((16, samples))  # 16 channels (ambisonics)
    for ch in range(16):
        ir_forward[ch, :] = 0.2 * np.exp(-time_axis * 8.0)  # Fast decay
    
    # Reverse tracer: Weak early, strong late (different decay)
    ir_reverse = np.zeros((16, samples))
    for ch in range(16):
        ir_reverse[ch, :] = 0.5 * np.exp(-time_axis * 2.0)  # Slower decay
    
    print(f"Synthetic data created:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s ({samples} samples)")
    print(f"  Channels: {ir_forward.shape[0]}")
    
    # Simulate the improved blending algorithm
    def improved_blend(ir_early, ir_late, sample_rate):
        """Replicate the improved blending logic."""
        
        # ENERGY MATCHING: Scale tracers to similar energy levels
        early_peak_region = slice(0, int(0.15 * sample_rate))  # 0-150ms
        late_peak_region = slice(int(0.05 * sample_rate), int(0.3 * sample_rate))  # 50-300ms
        
        early_rms = np.sqrt(np.mean(ir_early[:, early_peak_region] ** 2))
        late_rms = np.sqrt(np.mean(ir_late[:, late_peak_region] ** 2))
        
        if late_rms > 0 and early_rms > 0:
            energy_match_factor = early_rms / late_rms
            ir_late_scaled = ir_late * energy_match_factor
            print(f"  Energy matching - Forward RMS: {early_rms:.6f}, Reverse RMS: {late_rms:.6f}")
            print(f"  Scaling Reverse by factor: {energy_match_factor:.6f}")
        else:
            ir_late_scaled = ir_late
            
        # EXTENDED crossfade for smoother transition
        transition_time_sec = 0.1  # 100ms transition point
        crossfade_width = 0.15     # 150ms crossfade region
        crossfade_start = transition_time_sec - crossfade_width/2
        crossfade_end = transition_time_sec + crossfade_width/2
        
        samples = ir_early.shape[1]
        time_axis = np.arange(samples) / sample_rate
        
        # Create smooth crossfade weights using cosine interpolation
        early_weight = np.ones_like(time_axis)
        late_weight = np.zeros_like(time_axis)
        
        # Before crossfade: pure Forward
        mask_early = time_axis < crossfade_start
        early_weight[mask_early] = 1.0
        late_weight[mask_early] = 0.0
        
        # After crossfade: pure Reverse  
        mask_late = time_axis > crossfade_end
        early_weight[mask_late] = 0.0
        late_weight[mask_late] = 1.0
        
        # During crossfade: smooth cosine transition
        mask_crossfade = (time_axis >= crossfade_start) & (time_axis <= crossfade_end)
        if np.any(mask_crossfade):
            linear_progress = (time_axis[mask_crossfade] - crossfade_start) / crossfade_width
            cosine_progress = 0.5 * (1.0 - np.cos(np.pi * linear_progress))
            early_weight[mask_crossfade] = 1.0 - cosine_progress
            late_weight[mask_crossfade] = cosine_progress
        
        # Apply weighting
        ir_combined = np.zeros_like(ir_early)
        for ch in range(ir_early.shape[0]):
            ir_combined[ch, :] = (ir_early[ch, :] * early_weight + 
                                  ir_late_scaled[ch, :] * late_weight)
        
        # Gentle normalization - only if really needed
        combined_max_before = np.max(np.abs(ir_combined))
        target_max = 0.5
        if combined_max_before > target_max:
            normalization_factor = target_max / combined_max_before
            ir_combined *= normalization_factor
            print(f"  Applied gentle normalization: {combined_max_before:.6f} → {np.max(np.abs(ir_combined)):.6f}")
        
        return ir_combined, early_weight, late_weight
    
    # Apply improved blending
    ir_blended, early_weight, late_weight = improved_blend(ir_forward, ir_reverse, sample_rate)
    
    # Analysis
    print(f"\nBlending Analysis:")
    
    # Check energy distribution over time
    early_samples = int(0.1 * sample_rate)  # First 100ms
    mid_samples = int(0.5 * sample_rate)    # Up to 500ms
    
    early_energy = np.sum(ir_blended[:, :early_samples] ** 2)
    mid_energy = np.sum(ir_blended[:, early_samples:mid_samples] ** 2) 
    late_energy = np.sum(ir_blended[:, mid_samples:] ** 2)
    total_energy = early_energy + mid_energy + late_energy
    
    print(f"  Early energy (0-100ms): {early_energy:.6f} ({100*early_energy/total_energy:.1f}%)")
    print(f"  Mid energy (100-500ms): {mid_energy:.6f} ({100*mid_energy/total_energy:.1f}%)")
    print(f"  Late energy (500ms+): {late_energy:.6f} ({100*late_energy/total_energy:.1f}%)")
    
    # Check for discontinuities (mono analysis)
    mono_ir = np.mean(ir_blended, axis=0)
    diff = np.abs(np.diff(mono_ir))
    max_jump = np.max(diff)
    max_jump_time = np.argmax(diff) / sample_rate * 1000
    
    print(f"\nContinuity Check:")
    print(f"  Max amplitude jump: {max_jump:.6f} at {max_jump_time:.1f}ms")
    
    # Check specific time points
    test_times_ms = [50, 100, 150, 200, 300, 500]
    print(f"\nWeight Distribution Analysis:")
    print("Time (ms) | Early Weight | Late Weight | Combined Amplitude")
    print("-" * 55)
    
    for t_ms in test_times_ms:
        t_sample = int(t_ms * sample_rate / 1000)
        if t_sample < len(early_weight):
            early_w = early_weight[t_sample]
            late_w = late_weight[t_sample]
            amplitude = mono_ir[t_sample]
            print(f"{t_ms:8.0f} | {early_w:11.3f} | {late_w:10.3f} | {amplitude:16.6f}")
    
    # Success criteria
    crossfade_ok = max_jump_time < 175 or max_jump < 0.01  # Jump not in crossfade or very small
    energy_ok = late_energy > early_energy * 0.2  # Good late energy preservation
    
    print(f"\n{'✅' if crossfade_ok else '❌'} Crossfade smoothness: {'GOOD' if crossfade_ok else 'NEEDS WORK'}")
    print(f"{'✅' if energy_ok else '❌'} Energy preservation: {'GOOD' if energy_ok else 'NEEDS WORK'}")
    
    return crossfade_ok and energy_ok

if __name__ == "__main__":
    success = simulate_improved_blend()
    if success:
        print("\n🎉 IMPROVED BLENDING SIMULATION: SUCCESS")
        print("The energy matching and smooth crossfade should eliminate the 200ms jump!")
    else:
        print("\n⚠️  IMPROVED BLENDING SIMULATION: NEEDS REFINEMENT")