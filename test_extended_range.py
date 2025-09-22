#!/usr/bin/env python3
"""
Test extended dB range from -24dB to +24dB
"""
import numpy as np

def db_to_linear(db):
    """Convert dB to linear gain factor"""
    return 10.0 ** (db / 20.0)

def test_extended_db_range():
    """Test extended dB range conversion"""
    print("Extended dB Range Test (-24dB to +24dB):")
    print("=" * 50)
    
    test_cases = [
        +24.0,  # Maximum boost
        +12.0,  # Moderate boost
        +6.0,   # Double amplitude
        +3.0,   # √2 boost
        0.0,    # Unity gain
        -3.0,   # -√2 cut
        -6.0,   # Half amplitude
        -12.0,  # Moderate cut
        -24.0,  # Maximum cut
    ]
    
    print("dB     | Linear | Effect")
    print("-------|--------|------------------")
    for db in test_cases:
        linear = db_to_linear(db)
        if db > 0:
            effect = f"{linear:.1f}x boost"
        elif db < 0:
            effect = f"{1/linear:.1f}x reduction"
        else:
            effect = "unity gain"
        print(f"{db:+5.1f}  | {linear:6.3f} | {effect}")
    
    print("\nPractical Applications:")
    print("=" * 30)
    print("+24dB: Extreme echo boost (16x amplification)")
    print("+12dB: Strong echo emphasis (4x amplification)")  
    print("+6dB:  Moderate echo boost (2x amplification)")
    print("0dB:   Balanced echoes and reverb")
    print("-6dB:  Subtle echo reduction (0.5x)")
    print("-12dB: Strong echo reduction (0.25x)")
    print("-24dB: Echo almost muted (0.06x)")

if __name__ == "__main__":
    test_extended_db_range()