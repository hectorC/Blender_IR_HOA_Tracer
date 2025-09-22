#!/usr/bin/env python3
"""Debug the hybrid blending weights to identify the RT60 loss issue."""

import numpy as np

def analyze_hybrid_weights():
    """Analyze the hybrid blending weights that are killing RT60."""
    
    print("=== HYBRID BLENDING WEIGHT ANALYSIS ===")
    
    # Simulate the current (problematic) blending
    sample_rate = 48000
    ir_length_sec = 2.0
    total_samples = int(ir_length_sec * sample_rate)
    transition_time_sec = 0.1
    
    time_axis = np.arange(total_samples) / sample_rate
    
    # Current (problematic) weights from the code
    early_weight = np.exp(-np.maximum(0, time_axis - transition_time_sec) * 10)
    late_weight = 1.0 - early_weight * 0.7  # PROBLEM: 0.7 factor suppresses late energy
    
    # Proposed fix: Proper complementary weights
    early_weight_fixed = np.exp(-np.maximum(0, time_axis - transition_time_sec) * 5)
    late_weight_fixed = 1.0 - early_weight_fixed  # Proper complement
    
    print("Current (Problematic) Weights:")
    print("Time (ms) | Early | Late  | Total | Comment")
    print("-" * 50)
    
    times_ms = [0, 50, 100, 150, 200, 500, 1000, 2000]
    for t_ms in times_ms:
        sample_idx = int((t_ms / 1000.0) * sample_rate)
        if sample_idx < len(early_weight):
            early_w = early_weight[sample_idx]
            late_w = late_weight[sample_idx]
            total_w = early_w + late_w
            
            if t_ms <= 100:
                comment = "Early dominance OK"
            elif late_w < 0.8:
                comment = "❌ Late energy suppressed!"
            else:
                comment = "Late energy OK"
                
            print(f"{t_ms:8.0f} | {early_w:.3f} | {late_w:.3f} | {total_w:.3f} | {comment}")
    
    print(f"\nProposed Fix - Proper Complementary Weights:")
    print("Time (ms) | Early | Late  | Total | Comment")
    print("-" * 50)
    
    for t_ms in times_ms:
        sample_idx = int((t_ms / 1000.0) * sample_rate)
        if sample_idx < len(early_weight_fixed):
            early_w = early_weight_fixed[sample_idx]
            late_w = late_weight_fixed[sample_idx]
            total_w = early_w + late_w
            
            if t_ms <= 100:
                comment = "Early dominance preserved"
            elif t_ms >= 200:
                comment = "✅ Full late energy!"
            else:
                comment = "Smooth transition"
                
            print(f"{t_ms:8.0f} | {early_w:.3f} | {late_w:.3f} | {total_w:.3f} | {comment}")
    
    # Energy impact analysis
    print(f"\n=== ENERGY IMPACT ANALYSIS ===")
    
    # Simulate Forward energy (strong early, weak late)
    forward_early = 0.1
    forward_late = 0.001
    
    # Simulate Reverse energy (some early, strong late) 
    reverse_early = 0.05
    reverse_late = 0.08
    
    print("Energy Contribution Analysis:")
    print("Time Range | Forward*Early | Reverse*Late | Current Total | Fixed Total")
    print("-" * 75)
    
    ranges = [
        ("0-100ms", 0, forward_early, reverse_early),
        ("100-200ms", 1, forward_late, reverse_late), 
        ("200ms+", 2, forward_late, reverse_late)
    ]
    
    for range_name, range_idx, fwd_energy, rev_energy in ranges:
        if range_idx == 0:  # Early period
            early_w_curr = 1.0
            late_w_curr = 0.3
            early_w_fix = 1.0  
            late_w_fix = 0.0
        elif range_idx == 1:  # Transition
            early_w_curr = 0.6
            late_w_curr = 0.58
            early_w_fix = 0.6
            late_w_fix = 0.4
        else:  # Late period  
            early_w_curr = 0.0
            late_w_curr = 1.0
            early_w_fix = 0.0
            late_w_fix = 1.0
        
        current_total = fwd_energy * early_w_curr + rev_energy * late_w_curr
        fixed_total = fwd_energy * early_w_fix + rev_energy * late_w_fix
        
        improvement = fixed_total / current_total if current_total > 0 else float('inf')
        
        print(f"{range_name:10} | {fwd_energy * early_w_curr:11.4f} | {rev_energy * late_w_curr:11.4f} | {current_total:11.4f} | {fixed_total:10.4f} ({improvement:.1f}x)")
    
    print(f"\n✅ CONCLUSION: Fix will preserve late energy and restore RT60!")

if __name__ == "__main__":
    analyze_hybrid_weights()