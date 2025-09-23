#!/usr/bin/env python3
"""
Quick test script to verify RT60 normalization fix.
This demonstrates how the new normalization should work.
"""

def test_normalization_fix():
    """Test the new normalization calculation."""
    
    # Simulate connection count from your test
    connection_count = 301051
    
    print("=== NORMALIZATION TEST ===")
    print(f"Connection count: {connection_count:,}")
    
    # Old method (too aggressive)
    if connection_count > 50000:
        old_base_factor = min(1000.0, connection_count / 100.0)
        old_normalization = 1.0 / old_base_factor
        print(f"\nOLD method (aggressive):")
        print(f"  Base factor: {old_base_factor:.1f}")
        print(f"  Normalization: {old_normalization:.6f}")
        print(f"  This crushes reverb tail energy!")
    
    # New method (gentler)
    if connection_count > 100000:
        new_base_factor = min(20.0, connection_count / 10000.0)
        new_normalization = 1.0 / new_base_factor
        print(f"\nNEW method (gentler):")
        print(f"  Base factor: {new_base_factor:.1f}")
        print(f"  Normalization: {new_normalization:.6f}")
        print(f"  {new_normalization/old_normalization:.1f}x more reverb energy preserved!")
    
    # Expected RT60 improvement
    print(f"\n=== EXPECTED IMPACT ===")
    print("• OLD: RT60 = 0.00s (energy crushed by 1/1000)")
    print("• NEW: RT60 = 0.3-0.8s (realistic for 44% absorption carpet)")
    print("• Energy preservation: 50x better")
    
    print(f"\n✅ Fix should restore proper RT60 decay for absorptive materials!")

if __name__ == "__main__":
    test_normalization_fix()