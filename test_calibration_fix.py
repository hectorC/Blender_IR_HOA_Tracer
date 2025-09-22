#!/usr/bin/env python3
"""
Test the new calibration fallback strategy for occluded scenarios
"""
import numpy as np

def simulate_occluded_ir(sample_rate=48000, duration=1.0):
    """Create a simulated IR with blocked direct path but early reflections"""
    samples = int(sample_rate * duration)
    ir = np.zeros((16, samples))  # 16-channel ambisonic
    
    # No direct path (blocked by wall)
    # But add some early reflections
    
    # First reflection at 70ms (wall bounce)
    t1 = int(0.070 * sample_rate)
    if t1 < samples:
        ir[0, t1] = 0.3  # Strong first reflection
    
    # Second reflection at 85ms
    t2 = int(0.085 * sample_rate)
    if t2 < samples:
        ir[0, t2] = 0.2  # Secondary reflection
        
    # Some later reflections
    t3 = int(0.120 * sample_rate)
    if t3 < samples:
        ir[0, t3] = 0.15
        
    # Add some reverb tail
    for i in range(int(0.2 * sample_rate), int(0.8 * sample_rate), 100):
        if i < samples:
            decay = np.exp(-(i - 0.2 * sample_rate) / (0.3 * sample_rate))
            ir[0, i] = 0.1 * decay * np.random.random()
    
    return ir

def test_calibration_fallback():
    """Test the new calibration strategy"""
    print("Testing Calibration Fallback Strategy")
    print("=" * 50)
    
    # Simulate scenario
    sample_rate = 48000
    source_to_receiver_dist = 11.5  # meters (from your log)
    
    # Create occluded IR
    ir = simulate_occluded_ir(sample_rate)
    
    # Test original calibration (should fail)
    print("1. Testing Original Calibration Logic:")
    c = 343.0  # speed of sound
    delay = (source_to_receiver_dist / c) * sample_rate
    n = int(round(delay))
    n0 = max(0, n - 10)
    n1 = min(ir.shape[1], n + 11)
    
    a_meas_direct = float(np.max(np.abs(ir[0, n0:n1])))
    print(f"   Direct path energy at {n} samples: {a_meas_direct:.6f}")
    print(f"   Result: {'PASS' if a_meas_direct > 1e-9 else 'FAIL - would skip calibration'}")
    
    # Test new fallback calibration
    print("\n2. Testing Fallback Calibration Logic:")
    early_limit_ms = 200.0
    early_samples = int((early_limit_ms / 1000.0) * sample_rate)
    early_samples = min(early_samples, ir.shape[1])
    
    early_peak = float(np.max(np.abs(ir[0, :early_samples])))
    print(f"   Early reflection peak (0-200ms): {early_peak:.6f}")
    
    if early_peak > 1e-9:
        peak_idx = int(np.argmax(np.abs(ir[0, :early_samples])))
        peak_time_s = peak_idx / sample_rate
        peak_time_ms = peak_time_s * 1000
        
        estimated_reflection_dist = source_to_receiver_dist * 1.7
        a_exp = 1.0 / max(estimated_reflection_dist, 1e-9)
        k = a_exp / early_peak
        
        print(f"   Peak found at: {peak_time_ms:.1f}ms")
        print(f"   Estimated reflection distance: {estimated_reflection_dist:.2f}m")
        print(f"   Calibration factor: {k:.6f}")
        print(f"   Result: PASS - fallback calibration successful")
        
        # Show before/after
        calibrated_peak = early_peak * k
        print(f"   Before calibration: {early_peak:.6f}")
        print(f"   After calibration: {calibrated_peak:.6f}")
        print(f"   Expected amplitude: {a_exp:.6f}")
        
    else:
        print(f"   Result: FAIL - no early reflections found")
    
    # Visual verification
    print("\n3. IR Timeline Analysis:")
    time_axis = np.arange(ir.shape[1]) / sample_rate * 1000  # Convert to ms
    
    # Find all significant peaks
    threshold = 0.01
    peaks = []
    for i in range(int(0.5 * sample_rate)):  # First 500ms
        if np.abs(ir[0, i]) > threshold:
            peaks.append((i / sample_rate * 1000, ir[0, i]))
    
    print(f"   Significant peaks (>{threshold:.2f}) in first 500ms:")
    for time_ms, amplitude in peaks[:5]:  # Show first 5
        print(f"     {time_ms:6.1f}ms: {amplitude:+.3f}")

if __name__ == "__main__":
    test_calibration_fallback()