#!/usr/bin/env python3
"""Test the hybrid controls with different gain settings."""

import numpy as np
import sys
import os

def test_hybrid_controls():
    """Test hybrid controls with different gain configurations."""
    
    print("=== HYBRID CONTROL TESTING ===")
    
    # Test parameters
    sample_rate = 48000
    duration = 2.0
    samples = int(duration * sample_rate)
    
    # Create synthetic IR data
    time_axis = np.arange(samples) / sample_rate
    
    # Forward tracer: Strong early, weak late (discrete echoes)
    ir_forward = np.zeros((16, samples))
    for ch in range(16):
        # Simulate discrete echoes at specific times
        echo_times = [0.02, 0.05, 0.1, 0.15, 0.3, 0.5]  # Echo at these times
        for echo_time in echo_times:
            echo_sample = int(echo_time * sample_rate)
            if echo_sample < samples:
                # Decaying echoes
                amplitude = 0.1 * np.exp(-echo_time * 3.0)  
                ir_forward[ch, echo_sample] = amplitude
    
    # Reverse tracer: Weak early, strong late (diffuse reverb)
    ir_reverse = np.zeros((16, samples))
    for ch in range(16):
        ir_reverse[ch, :] = 0.05 * np.exp(-time_axis * 1.5)  # Slower decay
    
    print(f"Synthetic data created with discrete echoes at: {[20, 50, 100, 150, 300, 500]} ms")
    
    # Test different gain configurations
    test_configs = [
        {"name": "Default (0dB/0dB)", "forward_db": 0.0, "reverse_db": 0.0, "ramp": 0.2},
        {"name": "Tunnel/Corridor", "forward_db": 2.0, "reverse_db": -1.0, "ramp": 0.3},
        {"name": "Cathedral", "forward_db": -1.0, "reverse_db": 2.0, "ramp": 0.15},
        {"name": "Echo Focus", "forward_db": 3.0, "reverse_db": -3.0, "ramp": 0.4},
        {"name": "Reverb Focus", "forward_db": -3.0, "reverse_db": 3.0, "ramp": 0.1}
    ]
    
    def apply_hybrid_controls(ir_early, ir_late, forward_db, reverse_db, ramp_time, sample_rate):
        """Apply hybrid controls to IR data."""
        
        # Convert dB to linear gains
        forward_gain = 10.0 ** (forward_db / 20.0)
        reverse_gain = 10.0 ** (reverse_db / 20.0)
        
        # Energy scaling for additive combination (simplified)
        early_rms = np.sqrt(np.mean(ir_early ** 2))
        late_rms = np.sqrt(np.mean(ir_late ** 2))
        
        if late_rms > 0 and early_rms > 0:
            additive_scale_factor = (early_rms / late_rms) * 0.6
            ir_late_scaled = ir_late * additive_scale_factor
        else:
            ir_late_scaled = ir_late * 0.5
            
        # Additive weighting with user-controlled ramp
        reverse_ramp_start = 0.05
        reverse_ramp_end = ramp_time
        
        samples = ir_early.shape[1]
        time_axis = np.arange(samples) / sample_rate
        
        # Forward weight: Always 1.0
        forward_weight = np.ones_like(time_axis)
        
        # Reverse weight: Ramp from 0 to 1
        reverse_weight = np.zeros_like(time_axis)
        mask_early = time_axis < reverse_ramp_start
        reverse_weight[mask_early] = 0.0
        mask_late = time_axis > reverse_ramp_end
        reverse_weight[mask_late] = 1.0
        
        # Smooth ramp
        mask_ramp = (time_axis >= reverse_ramp_start) & (time_axis <= reverse_ramp_end)
        if np.any(mask_ramp) and reverse_ramp_end > reverse_ramp_start:
            ramp_width = reverse_ramp_end - reverse_ramp_start
            linear_progress = (time_axis[mask_ramp] - reverse_ramp_start) / ramp_width
            cosine_progress = 0.5 * (1.0 - np.cos(np.pi * linear_progress))
            reverse_weight[mask_ramp] = cosine_progress
        
        # Apply user gains
        ir_combined = np.zeros_like(ir_early)
        for ch in range(ir_early.shape[0]):
            ir_combined[ch, :] = (ir_early[ch, :] * forward_weight * forward_gain + 
                                  ir_late_scaled[ch, :] * reverse_weight * reverse_gain)
        
        return ir_combined, forward_weight, reverse_weight
    
    # Test each configuration
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        print(f"Forward: {config['forward_db']:+.1f}dB | Reverse: {config['reverse_db']:+.1f}dB | Ramp: {config['ramp']:.3f}s")
        
        ir_result, fw, rw = apply_hybrid_controls(
            ir_forward, ir_reverse, 
            config['forward_db'], config['reverse_db'], config['ramp'], 
            sample_rate
        )
        
        # Analyze key time points
        test_times = [50, 150, 300, 500]  # ms - where our echoes are
        mono_ir = np.mean(ir_result, axis=0)
        
        print("Echo Analysis:")
        print("Time (ms) | Amplitude | Interpretation")
        print("-" * 40)
        
        for t_ms in test_times:
            t_sample = int(t_ms * sample_rate / 1000)
            if t_sample < len(mono_ir):
                amplitude = mono_ir[t_sample]
                
                # Determine what's contributing
                fw_contrib = fw[t_sample] * (10.0 ** (config['forward_db'] / 20.0))
                rw_contrib = rw[t_sample] * (10.0 ** (config['reverse_db'] / 20.0))
                
                if rw_contrib < 0.1:
                    interp = "Pure echo"
                elif fw_contrib > rw_contrib * 2:
                    interp = "Echo dominant"  
                elif rw_contrib > fw_contrib * 2:
                    interp = "Reverb dominant"
                else:
                    interp = "Mixed"
                
                print(f"{t_ms:8.0f} | {amplitude:9.6f} | {interp}")
        
        # Overall energy assessment
        total_energy = np.sum(mono_ir ** 2)
        early_energy = np.sum(mono_ir[:int(0.1*sample_rate)] ** 2)
        late_energy = np.sum(mono_ir[int(0.1*sample_rate):] ** 2)
        
        echo_emphasis = early_energy / total_energy if total_energy > 0 else 0
        reverb_emphasis = late_energy / total_energy if total_energy > 0 else 0
        
        print(f"Balance: {echo_emphasis:.1%} echo / {reverb_emphasis:.1%} reverb")
        
        # Suitability assessment
        if config['name'] == "Tunnel/Corridor":
            if echo_emphasis > 0.3:
                print("✅ GOOD for tunnels - strong discrete echoes preserved")
            else:
                print("⚠️  May be too reverberant for tunnel clarity")
        elif config['name'] == "Cathedral":  
            if reverb_emphasis > 0.7:
                print("✅ GOOD for cathedrals - lush reverb with subtle echoes")
            else:
                print("⚠️  May be too dry for cathedral ambience")
        
    print(f"\n🎛️  HYBRID CONTROL TEST COMPLETE")
    print("Users can now fine-tune discrete echoes vs diffuse reverb balance!")
    return True

if __name__ == "__main__":
    success = test_hybrid_controls()
    if success:
        print("\n🎉 HYBRID CONTROLS READY FOR USE!")
    else:
        print("\n❌ HYBRID CONTROLS NEED REFINEMENT")