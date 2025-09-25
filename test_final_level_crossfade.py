#!/usr/bin/env python3
"""Test the enhanced crossfade logic with final forward level control."""

import numpy as np
import sys
import os

def test_enhanced_crossfade():
    """Test crossfade with final forward level control."""
    
    def crossfade_hybrid_irs(forward_ir, reverse_ir, sample_rate, 
                           crossfade_start_ms, crossfade_length_ms, forward_final_level):
        """Enhanced crossfade with final forward level control."""
        min_length = min(forward_ir.shape[1], reverse_ir.shape[1])
        forward_ir = forward_ir[:, :min_length]
        reverse_ir = reverse_ir[:, :min_length]
        
        crossfade_start_samples = int((crossfade_start_ms / 1000.0) * sample_rate)
        crossfade_end_samples = int(((crossfade_start_ms + crossfade_length_ms) / 1000.0) * sample_rate)
        
        crossfade_start_samples = max(0, min(crossfade_start_samples, min_length - 1))
        crossfade_end_samples = max(crossfade_start_samples + 1, min(crossfade_end_samples, min_length))
        
        forward_weight = np.ones(min_length)
        reverse_weight = np.ones(min_length)
        
        # Early period: 100% forward, 0% reverse
        forward_weight[:crossfade_start_samples] = 1.0
        reverse_weight[:crossfade_start_samples] = 0.0
        
        # Transition period
        if crossfade_end_samples > crossfade_start_samples:
            transition_length = crossfade_end_samples - crossfade_start_samples
            forward_transition = np.linspace(1.0, forward_final_level, transition_length)
            reverse_transition = np.linspace(0.0, 1.0 - forward_final_level, transition_length)
            
            forward_weight[crossfade_start_samples:crossfade_end_samples] = forward_transition
            reverse_weight[crossfade_start_samples:crossfade_end_samples] = reverse_transition
        
        # Late period
        forward_weight[crossfade_end_samples:] = forward_final_level
        reverse_weight[crossfade_end_samples:] = 1.0 - forward_final_level
        
        hybrid_ir = (forward_ir * forward_weight[None, :] + 
                    reverse_ir * reverse_weight[None, :])
        
        return hybrid_ir, forward_weight, reverse_weight

    print("=== Enhanced Crossfade Test with Final Forward Level ===")
    
    # Test parameters
    sample_rate = 44100
    duration_s = 2.0
    num_samples = int(duration_s * sample_rate)
    num_channels = 4
    
    # Create test IRs
    forward_ir = np.random.normal(0, 0.1, (num_channels, num_samples))
    reverse_ir = np.random.normal(0, 0.1, (num_channels, num_samples))
    
    # Test different final forward levels
    test_cases = [
        {"final_level": 0.0, "name": "Complete fadeout (original behavior)"},
        {"final_level": 0.2, "name": "20% forward preserved"},
        {"final_level": 0.5, "name": "50% forward preserved"},
        {"final_level": 1.0, "name": "No forward fadeout"},
    ]
    
    crossfade_start_ms = 50.0
    crossfade_length_ms = 150.0
    
    for test_case in test_cases:
        final_level = test_case["final_level"]
        name = test_case["name"]
        
        print(f"\n--- {name} (final level: {final_level*100:.0f}%) ---")
        
        hybrid_ir, fw_weight, rv_weight = crossfade_hybrid_irs(
            forward_ir, reverse_ir, sample_rate, 
            crossfade_start_ms, crossfade_length_ms, final_level
        )
        
        # Check key points
        early_idx = int(0.025 * sample_rate)  # 25ms (before crossfade)
        mid_idx = int(0.125 * sample_rate)    # 125ms (during crossfade)  
        late_idx = int(0.5 * sample_rate)     # 500ms (after crossfade)
        
        print(f"At 25ms (early):  forward={fw_weight[early_idx]:.3f}, reverse={rv_weight[early_idx]:.3f}")
        print(f"At 125ms (mid):   forward={fw_weight[mid_idx]:.3f}, reverse={rv_weight[mid_idx]:.3f}")
        print(f"At 500ms (late):  forward={fw_weight[late_idx]:.3f}, reverse={rv_weight[late_idx]:.3f}")
        
        # Verify weight sum normalization
        weight_sum_early = fw_weight[early_idx] + rv_weight[early_idx]
        weight_sum_late = fw_weight[late_idx] + rv_weight[late_idx]
        print(f"Weight sum early: {weight_sum_early:.6f}, late: {weight_sum_late:.6f}")
        
        # Check final forward level matches expectation
        actual_final = fw_weight[late_idx]
        print(f"Expected final forward: {final_level:.3f}, Actual: {actual_final:.3f}")
        
        if abs(actual_final - final_level) < 0.001:
            print("✓ Final forward level correct")
        else:
            print("✗ Final forward level mismatch")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_enhanced_crossfade()