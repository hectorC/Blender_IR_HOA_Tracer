#!/usr/bin/env python3
"""Test the new hybrid workflow implementation."""

import numpy as np
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_crossfade_logic():
    """Test the crossfading logic without Blender dependency."""
    
    # Mock the crossfade method logic
    def crossfade_hybrid_irs(forward_ir, reverse_ir, sample_rate):
        """Crossfade forward and reverse IRs with time-based weighting."""
        min_length = min(forward_ir.shape[1], reverse_ir.shape[1])
        forward_ir = forward_ir[:, :min_length]
        reverse_ir = reverse_ir[:, :min_length]
        
        early_time_ms = 50.0
        late_time_ms = 200.0
        
        early_samples = int((early_time_ms / 1000.0) * sample_rate)
        late_samples = int((late_time_ms / 1000.0) * sample_rate)
        
        forward_weight = np.ones(min_length)
        reverse_weight = np.ones(min_length)
        
        # Early period: 100% forward, 0% reverse
        forward_weight[:early_samples] = 1.0
        reverse_weight[:early_samples] = 0.0
        
        # Transition period: smooth crossfade
        if late_samples > early_samples:
            transition_length = late_samples - early_samples
            transition_ramp = np.linspace(0, 1, transition_length)
            
            forward_weight[early_samples:late_samples] = 1.0 - transition_ramp
            reverse_weight[early_samples:late_samples] = transition_ramp
        
        # Late period: 0% forward, 100% reverse
        forward_weight[late_samples:] = 0.0
        reverse_weight[late_samples:] = 1.0
        
        # Apply weights and combine
        hybrid_ir = (forward_ir * forward_weight[None, :] + 
                    reverse_ir * reverse_weight[None, :])
        
        return hybrid_ir, forward_weight, reverse_weight
    
    # Test parameters
    sample_rate = 44100
    duration_s = 2.0
    num_samples = int(duration_s * sample_rate)
    num_channels = 4  # Ambisonic
    
    # Create test IRs
    # Forward: strong early energy, weak late energy
    forward_ir = np.zeros((num_channels, num_samples))
    forward_ir[:, 100] = 1.0  # Strong direct at ~2.3ms
    forward_ir[:, 1000:3000] = np.random.normal(0, 0.1, (num_channels, 2000))  # Early reflections
    forward_ir[:, 10000:] = np.random.normal(0, 0.01, (num_channels, num_samples - 10000))  # Weak late
    
    # Reverse: weak early energy, strong late energy  
    reverse_ir = np.zeros((num_channels, num_samples))
    reverse_ir[:, :5000] = np.random.normal(0, 0.02, (num_channels, 5000))  # Weak early
    reverse_ir[:, 10000:] = np.random.normal(0, 0.3, (num_channels, num_samples - 10000))  # Strong late
    
    # Apply crossfade
    hybrid_ir, fw_weight, rv_weight = crossfade_hybrid_irs(forward_ir, reverse_ir, sample_rate)
    
    # Analyze results
    early_samples = int(0.05 * sample_rate)  # 50ms
    late_start = int(0.2 * sample_rate)      # 200ms
    
    print("=== New Hybrid Workflow Crossfade Test ===")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration_s} s ({num_samples} samples)")
    print(f"Early period: 0-{early_samples} samples (0-50ms)")
    print(f"Transition: {early_samples}-{late_start} samples (50-200ms)")
    print(f"Late period: {late_start}+ samples (200ms+)")
    print()
    
    # Check weights at key points
    print("Weight verification:")
    print(f"At 0ms: forward={fw_weight[0]:.3f}, reverse={rv_weight[0]:.3f}")
    print(f"At 25ms: forward={fw_weight[int(0.025*sample_rate)]:.3f}, reverse={rv_weight[int(0.025*sample_rate)]:.3f}")
    print(f"At 50ms: forward={fw_weight[early_samples]:.3f}, reverse={rv_weight[early_samples]:.3f}")
    print(f"At 125ms: forward={fw_weight[int(0.125*sample_rate)]:.3f}, reverse={rv_weight[int(0.125*sample_rate)]:.3f}")
    print(f"At 200ms: forward={fw_weight[late_start]:.3f}, reverse={rv_weight[late_start]:.3f}")
    print(f"At 500ms: forward={fw_weight[int(0.5*sample_rate)]:.3f}, reverse={rv_weight[int(0.5*sample_rate)]:.3f}")
    print()
    
    # Analyze energy distribution
    early_energy = np.sum(hybrid_ir[:, :early_samples]**2)
    mid_energy = np.sum(hybrid_ir[:, early_samples:late_start]**2) 
    late_energy = np.sum(hybrid_ir[:, late_start:]**2)
    total_energy = early_energy + mid_energy + late_energy
    
    print("Energy distribution in hybrid result:")
    print(f"Early (0-50ms): {early_energy/total_energy*100:.1f}%")
    print(f"Transition (50-200ms): {mid_energy/total_energy*100:.1f}%")
    print(f"Late (200ms+): {late_energy/total_energy*100:.1f}%")
    print()
    
    # Verify crossfade completeness
    weight_sum = fw_weight + rv_weight
    weight_error = np.max(np.abs(weight_sum - 1.0))
    print(f"Weight sum verification (should be 1.0): max error = {weight_error:.6f}")
    
    if weight_error < 1e-10:
        print("✓ Crossfade weights are properly normalized")
    else:
        print("✗ Crossfade weight normalization issue")
    
    # Check for smooth transitions
    weight_gradient = np.diff(fw_weight)
    max_gradient = np.max(np.abs(weight_gradient))
    print(f"Maximum weight gradient: {max_gradient:.6f}")
    print(f"Transition smoothness: {'✓ Smooth' if max_gradient < 0.01 else '✗ Rough'}")
    
    print("\n=== Test Complete ===")
    
    return hybrid_ir, fw_weight, rv_weight

if __name__ == "__main__":
    test_crossfade_logic()