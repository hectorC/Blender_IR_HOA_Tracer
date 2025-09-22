#!/usr/bin/env python3
"""Analyze what went wrong with the hybrid fix."""

import numpy as np

def analyze_overcorrection():
    """Analyze why the hybrid fix caused energy saturation."""
    
    print("=== HYBRID OVERCORRECTION ANALYSIS ===")
    
    # The problem: We changed from suppressed weights to full complementary weights
    # But this might be doubling the energy contribution
    
    sample_rate = 48000
    time_axis = np.arange(96000) / sample_rate  # 2 seconds
    transition_time_sec = 0.1
    
    # Old problematic weights
    early_old = np.exp(-np.maximum(0, time_axis - transition_time_sec) * 10)
    late_old = 1.0 - early_old * 0.7
    
    # New fix (might be too aggressive)
    early_new = np.exp(-np.maximum(0, time_axis - transition_time_sec) * 5)
    late_new = 1.0 - early_new
    
    # Analysis at key times
    times_ms = [0, 50, 100, 200, 500, 1000]
    
    print("Energy Amplification Analysis:")
    print("Time (ms) | Old Total | New Total | Amplification")
    print("-" * 50)
    
    for t_ms in times_ms:
        idx = int((t_ms / 1000.0) * sample_rate)
        if idx < len(early_old):
            old_total = early_old[idx] + late_old[idx]
            new_total = early_new[idx] + late_new[idx]
            amplification = new_total / old_total if old_total > 0 else float('inf')
            
            print(f"{t_ms:8.0f} | {old_total:8.3f} | {new_total:8.3f} | {amplification:11.2f}x")
    
    # The issue: We're now always summing to 1.0, but before we were sometimes < 1.0
    # This means we're amplifying the overall energy level
    
    print(f"\n🚨 PROBLEM IDENTIFIED:")
    print(f"- Old weights: Sum varied from 1.0 to 1.3 (inconsistent)")
    print(f"- New weights: Always sum to 1.0 (consistent but might amplify)")
    print(f"- Result: Energy levels became too high, causing saturation")
    
    print(f"\n💡 SOLUTION:")
    print(f"- Need to normalize overall energy level")
    print(f"- Or reduce the contribution factors")
    print(f"- Or use a scaling factor < 1.0")

if __name__ == "__main__":
    analyze_overcorrection()