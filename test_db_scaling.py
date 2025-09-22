#!/usr/bin/env python3
"""
Quick test to verify dB gain scaling implementation
"""
import numpy as np

def db_to_linear(db):
    """Convert dB to linear gain factor"""
    return 10.0 ** (db / 20.0)

def test_db_conversion():
    """Test dB to linear conversion"""
    test_cases = [
        (+6.0, 1.995),  # Should be ~2x
        (+3.0, 1.413),  # Should be ~√2
        ( 0.0, 1.000),  # Unity
        (-3.0, 0.708),  # Should be ~1/√2
        (-6.0, 0.501),  # Should be ~0.5x
    ]
    
    print("dB to Linear Conversion Test:")
    print("=" * 40)
    for db, expected in test_cases:
        actual = db_to_linear(db)
        error = abs(actual - expected)
        print(f"{db:+4.1f}dB → {actual:.3f}x (expected {expected:.3f}x, error: {error:.3f})")
    print()

def test_additive_blend():
    """Test additive blending with gain controls"""
    # Simulate simple signals
    forward_signal = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Discrete echoes
    reverse_signal = np.array([0.1, 0.2, 0.4, 0.6, 0.8])  # Building reverb
    
    # Time weights (reverse builds up)
    forward_weight = np.ones(5)  # Always 1.0
    reverse_weight = np.array([0.0, 0.2, 0.5, 0.8, 1.0])  # Ramp up
    
    # User gains
    forward_gain_db = +3.0  # Boost echoes by 3dB
    reverse_gain_db = -2.0  # Reduce reverb by 2dB
    
    forward_gain_linear = db_to_linear(forward_gain_db)
    reverse_gain_linear = db_to_linear(reverse_gain_db)
    
    # Apply blending (same formula as ray_tracer.py)
    combined = (forward_signal * forward_weight * forward_gain_linear + 
                reverse_signal * reverse_weight * reverse_gain_linear)
    
    print("Additive Blend Test:")
    print("=" * 50)
    print(f"Forward gain: {forward_gain_db:+.1f}dB ({forward_gain_linear:.3f}x)")
    print(f"Reverse gain: {reverse_gain_db:+.1f}dB ({reverse_gain_linear:.3f}x)")
    print()
    print("Sample | Forward | Reverse | FwdWgt | RevWgt | Combined")
    print("-------|---------|---------|--------|--------|----------")
    for i in range(5):
        fwd_contrib = forward_signal[i] * forward_weight[i] * forward_gain_linear
        rev_contrib = reverse_signal[i] * reverse_weight[i] * reverse_gain_linear
        print(f"   {i}   |  {forward_signal[i]:.3f}  |  {reverse_signal[i]:.3f}  | {forward_weight[i]:.3f}  | {reverse_weight[i]:.3f}  |  {combined[i]:.3f}")
        print(f"       | ({fwd_contrib:.3f}) | ({rev_contrib:.3f}) |        |        | = {fwd_contrib + rev_contrib:.3f}")
    print()

if __name__ == "__main__":
    test_db_conversion()
    test_additive_blend()