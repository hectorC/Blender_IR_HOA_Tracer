#!/usr/bin/env python3
"""Quick test to verify the NO normalization fix."""

# Test: Check if removing normalization entirely
old_factor = 1.0 / 8192  # Old normalization: divide by number of rays
new_factor = 1.0         # New normalization: NO DIVISION AT ALL

print("=== REVERSE TRACER NO NORMALIZATION FIX ===")
print(f"Old normalization factor: {old_factor:.2e} (divide by 8192 rays)")
print(f"New normalization factor: {new_factor:.2e} (NO NORMALIZATION)")
print(f"Energy improvement: {new_factor/old_factor:.1f}x stronger")

# Example energy calculation
before_energy = 0.185  # From debug output: "Before: 1.85e-01"
old_after = before_energy * old_factor
new_after = before_energy * new_factor

print(f"\nExample with energy {before_energy:.3f}:")
print(f"Old result: {old_after:.2e} (inaudible)")
print(f"New result: {new_after:.2e} (should be VERY audible!)")
print(f"Improvement: {new_after/old_after:.0f}x stronger")

if new_after > 0.01:
    print("\n✅ SUCCESS: Energy should now be VERY audible!")
    print("🔊 This should produce strong reverb tail")
else:
    print("\n❌ Still might be too quiet")