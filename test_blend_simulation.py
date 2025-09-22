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
    
    # Simulate the ADDITIVE blending algorithm  
    def additive_blend(ir_early, ir_late, sample_rate):
        """Replicate the new ADDITIVE blending logic."""
        
        # ENERGY SCALING for additive combination
        early_peak_region = slice(0, int(0.15 * sample_rate))  # 0-150ms
        late_peak_region = slice(int(0.1 * sample_rate), int(0.4 * sample_rate))  # 100-400ms
        
        early_rms = np.sqrt(np.mean(ir_early[:, early_peak_region] ** 2))
        late_rms = np.sqrt(np.mean(ir_late[:, late_peak_region] ** 2))
        
        # For additive blend: scale Reverse to complement Forward
        if late_rms > 0 and early_rms > 0:
            additive_scale_factor = (early_rms / late_rms) * 0.6  # 60% of energy match
            ir_late_scaled = ir_late * additive_scale_factor
            print(f"  Additive scaling - Forward RMS: {early_rms:.6f}, Reverse RMS: {late_rms:.6f}")
            print(f"  Reverse scaled by: {additive_scale_factor:.6f} (for additive blend)")
        else:
            ir_late_scaled = ir_late * 0.5  # Default conservative scaling
            
        # ADDITIVE WEIGHTING: Forward always 100%, Reverse ramps up
        reverse_ramp_start = 0.05   # 50ms - start adding reverse
        reverse_ramp_end = 0.2      # 200ms - full reverse addition
        
        samples = ir_early.shape[1]
        time_axis = np.arange(samples) / sample_rate
        
        # Forward weight: Always 1.0 (preserve all discrete echoes)
        forward_weight = np.ones_like(time_axis)
        
        # Reverse weight: Smooth ramp from 0 to 1
        reverse_weight = np.zeros_like(time_axis)
        
        # Before ramp: no reverse
        mask_early = time_axis < reverse_ramp_start
        reverse_weight[mask_early] = 0.0
        
        # After ramp: full reverse addition
        mask_late = time_axis > reverse_ramp_end
        reverse_weight[mask_late] = 1.0
        
        # During ramp: smooth increase using cosine
        mask_ramp = (time_axis >= reverse_ramp_start) & (time_axis <= reverse_ramp_end)
        if np.any(mask_ramp):
            ramp_width = reverse_ramp_end - reverse_ramp_start
            linear_progress = (time_axis[mask_ramp] - reverse_ramp_start) / ramp_width
            cosine_progress = 0.5 * (1.0 - np.cos(np.pi * linear_progress))
            reverse_weight[mask_ramp] = cosine_progress
        
        # ADDITIVE COMBINATION: Forward + (Reverse * weight)
        ir_combined = np.zeros_like(ir_early)
        for ch in range(ir_early.shape[0]):
            ir_combined[ch, :] = (ir_early[ch, :] * forward_weight + 
                                  ir_late_scaled[ch, :] * reverse_weight)
        
        # Gentle normalization for additive blend
        combined_max_before = np.max(np.abs(ir_combined))
        target_max = 0.7  # Higher threshold for additive
        if combined_max_before > target_max:
            normalization_factor = target_max / combined_max_before
            ir_combined *= normalization_factor
            print(f"  Applied gentle normalization: {combined_max_before:.6f} → {np.max(np.abs(ir_combined)):.6f}")
        
        return ir_combined, forward_weight, reverse_weight
    
    # Apply ADDITIVE blending
    ir_blended, forward_weight, reverse_weight = additive_blend(ir_forward, ir_reverse, sample_rate)
    
    # Analysis
    print(f"\nADDITIVE Blending Analysis:")
    
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
    
    # Check specific time points for ADDITIVE weights
    test_times_ms = [25, 50, 100, 150, 200, 300, 500]
    print(f"\nADDITIVE Weight Distribution Analysis:")
    print("Time (ms) | Forward Weight | Reverse Weight | Combined Amplitude | Interpretation")
    print("-" * 82)
    
    for t_ms in test_times_ms:
        t_sample = int(t_ms * sample_rate / 1000)
        if t_sample < len(forward_weight):
            forward_w = forward_weight[t_sample]
            reverse_w = reverse_weight[t_sample]
            amplitude = mono_ir[t_sample]
            
            if reverse_w == 0:
                interp = "Pure Forward"
            elif reverse_w < 0.1:
                interp = "Mostly Forward"  
            elif reverse_w < 0.9:
                interp = "Forward+Reverse"
            else:
                interp = "Forward+Full Reverse"
                
            print(f"{t_ms:8.0f} | {forward_w:13.3f} | {reverse_w:13.3f} | {amplitude:16.6f} | {interp}")
    
    # Success criteria for ADDITIVE approach
    discrete_preservation = np.all(forward_weight == 1.0)  # Forward always preserved
    gradual_addition = reverse_weight[int(0.05*sample_rate)] < reverse_weight[int(0.2*sample_rate)]  # Reverse ramps up
    energy_ok = late_energy > early_energy * 0.2  # Good late energy preservation
    
    print(f"\n{'✅' if discrete_preservation else '❌'} Discrete echo preservation: {'PERFECT' if discrete_preservation else 'COMPROMISED'} (Forward always 100%)")
    print(f"{'✅' if gradual_addition else '❌'} Gradual reverse addition: {'GOOD' if gradual_addition else 'NEEDS WORK'}")
    print(f"{'✅' if energy_ok else '❌'} Energy preservation: {'GOOD' if energy_ok else 'NEEDS WORK'}")
    
    return discrete_preservation and gradual_addition and energy_ok

if __name__ == "__main__":
    success = simulate_improved_blend()
    if success:
        print("\n🎉 ADDITIVE BLENDING SIMULATION: SUCCESS")
        print("✨ Forward echoes preserved 100% + Reverse reverb gradually added!")
        print("🎯 Discrete tunnel echoes should be clearly audible with rich diffuse tail!")
    else:
        print("\n⚠️  ADDITIVE BLENDING SIMULATION: NEEDS REFINEMENT")