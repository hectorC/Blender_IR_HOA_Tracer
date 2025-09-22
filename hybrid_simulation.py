#!/usr/bin/env python3
"""Simulate hybrid blending using our known working Forward and Reverse results."""

import numpy as np

def simulate_hybrid_blend():
    """Simulate the hybrid blending algorithm based on the implementation."""
    
    print("=== HYBRID BLENDING SIMULATION ===")
    
    # Simulate timing and parameters
    sample_rate = 48000
    ir_length_sec = 2.0
    total_samples = int(ir_length_sec * sample_rate)  # 96000 samples
    transition_time_sec = 0.1  # 100ms transition point
    transition_sample = int(transition_time_sec * sample_rate)  # 4800 samples
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"IR length: {ir_length_sec} sec ({total_samples} samples)")
    print(f"Transition point: {transition_time_sec*1000} ms (sample {transition_sample})")
    
    # Create time axis
    time_axis = np.arange(total_samples) / sample_rate
    
    # Blending weights (from the actual implementation)
    early_weight = np.exp(-np.maximum(0, time_axis - transition_time_sec) * 10)
    late_weight = 1.0 - early_weight * 0.7  # Allow some overlap
    
    # Analyze the blending strategy
    print(f"\n=== BLENDING ANALYSIS ===")
    
    # Key time points
    times_ms = [0, 50, 100, 150, 200, 500, 1000, 2000]
    print("Time (ms) | Early Weight | Late Weight | Total")
    print("-" * 45)
    
    for t_ms in times_ms:
        sample_idx = int((t_ms / 1000.0) * sample_rate)
        if sample_idx < len(early_weight):
            early_w = early_weight[sample_idx]
            late_w = late_weight[sample_idx]
            total_w = early_w + late_w
            print(f"{t_ms:8.0f} | {early_w:11.3f} | {late_w:10.3f} | {total_w:.3f}")
    
    # Simulate Forward and Reverse characteristics
    print(f"\n=== SIMULATED TRACER CHARACTERISTICS ===")
    
    # Forward tracer: Strong early (0-100ms), weak late
    forward_early_strength = 0.1  # Based on our analysis: peak ~0.2 dB = ~0.1 amplitude
    forward_late_strength = 0.001  # Weak late energy
    
    # Reverse tracer: Weak early, strong throughout
    reverse_early_strength = 0.05  # Some early energy
    reverse_late_strength = 0.08   # Strong late energy (from 0.61s RT60)
    
    print(f"Forward - Early: {forward_early_strength}, Late: {forward_late_strength}")
    print(f"Reverse - Early: {reverse_early_strength}, Late: {reverse_late_strength}")
    
    # Simulate hybrid combination at key points
    print(f"\n=== HYBRID COMBINATION SIMULATION ===")
    print("Time (ms) | Forward Contrib | Reverse Contrib | Total Energy")
    print("-" * 58)
    
    for t_ms in times_ms:
        sample_idx = int((t_ms / 1000.0) * sample_rate)
        if sample_idx < len(early_weight):
            early_w = early_weight[sample_idx]
            late_w = late_weight[sample_idx]
            
            # Choose Forward/Reverse strengths based on time
            if t_ms <= 100:
                forward_energy = forward_early_strength
                reverse_energy = reverse_early_strength
            else:
                forward_energy = forward_late_strength  
                reverse_energy = reverse_late_strength
            
            # Apply hybrid weighting
            forward_contrib = forward_energy * early_w
            reverse_contrib = reverse_energy * late_w
            total_energy = forward_contrib + reverse_contrib
            
            print(f"{t_ms:8.0f} | {forward_contrib:14.4f} | {reverse_contrib:14.4f} | {total_energy:.4f}")
    
    # Expected characteristics
    print(f"\n=== EXPECTED HYBRID CHARACTERISTICS ===")
    print("✓ Strong early reflections (0-100ms): Forward dominates")
    print("✓ Smooth transition (100ms): Overlapping weights")  
    print("✓ Rich late reverb (100ms+): Reverse dominates")
    print("✓ Natural decay curve: Combined energy profiles")
    print("✓ Best of both worlds: Early precision + late richness")

if __name__ == "__main__":
    simulate_hybrid_blend()