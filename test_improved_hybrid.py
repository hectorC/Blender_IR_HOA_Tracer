#!/usr/bin/env python3
"""Test the improved hybrid ray tracer with energy matching and smooth crossfade."""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock Blender modules for standalone testing
class MockVector:
    def __init__(self, coords=(0, 0, 0)):
        if hasattr(coords, '__iter__'):
            self.x, self.y, self.z = coords
        else:
            self.x = self.y = self.z = coords
    
    def __sub__(self, other):
        return MockVector((self.x - other.x, self.y - other.y, self.z - other.z))
    
    def __add__(self, other):
        return MockVector((self.x + other.x, self.y + other.y, self.z + other.z))
    
    def normalized(self):
        length = (self.x**2 + self.y**2 + self.z**2)**0.5
        if length > 0:
            return MockVector((self.x/length, self.y/length, self.z/length))
        return MockVector((0, 0, 0))
    
    @property
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

class MockMathutils:
    Vector = MockVector

sys.modules['mathutils'] = MockMathutils()
sys.modules['bpy'] = type('MockBpy', (), {})()

try:
    from ir_raytracer.core.ray_tracer import (
        RayTracingConfig, 
        ForwardRayTracer, 
        ReverseRayTracer, 
        create_hybrid_ir
    )
    print("✅ Successfully imported ray tracer modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_improved_hybrid():
    """Test the improved hybrid with energy matching and smooth crossfade."""
    
    print("\n=== TESTING IMPROVED HYBRID RAY TRACER ===")
    print("Features: Energy matching + smooth cosine crossfade + gentler normalization")
    
    # Create test configuration
    config = RayTracingConfig()
    config.num_rays = 8192
    config.max_bounces = 64
    config.sample_rate = 48000
    config.ir_duration = 4.0
    config.enable_specular = True
    config.enable_diffuse = True
    
    # Test points
    source = MockVector((5, 5, 5))
    receiver = MockVector((15, 15, 15))
    
    print(f"\nConfiguration:")
    print(f"  Rays: {config.num_rays}")
    print(f"  Bounces: {config.max_bounces}")
    print(f"  Duration: {config.ir_duration}s")
    print(f"  Sample rate: {config.sample_rate} Hz")
    
    try:
        # Create hybrid impulse response
        print(f"\n🎯 Running hybrid tracer...")
        ir_result = create_hybrid_ir(source, receiver, None, [], config)
        
        if ir_result is not None and ir_result.size > 0:
            print(f"✅ Generated hybrid IR: {ir_result.shape}")
            
            # Basic analysis
            max_amplitude = np.max(np.abs(ir_result))
            rms_amplitude = np.sqrt(np.mean(ir_result**2))
            print(f"  Max amplitude: {max_amplitude:.6f}")
            print(f"  RMS amplitude: {rms_amplitude:.6f}")
            
            # Check for different time regions
            sample_rate = config.sample_rate
            early_samples = int(0.1 * sample_rate)  # First 100ms
            mid_samples = int(0.5 * sample_rate)    # Up to 500ms
            
            early_energy = np.sum(ir_result[:, :early_samples] ** 2) if ir_result.ndim > 1 else np.sum(ir_result[:early_samples] ** 2)
            mid_energy = np.sum(ir_result[:, early_samples:mid_samples] ** 2) if ir_result.ndim > 1 else np.sum(ir_result[early_samples:mid_samples] ** 2)
            late_energy = np.sum(ir_result[:, mid_samples:] ** 2) if ir_result.ndim > 1 else np.sum(ir_result[mid_samples:] ** 2)
            
            total_energy = early_energy + mid_energy + late_energy
            
            print(f"\n🔊 Energy Distribution:")
            print(f"  Early (0-100ms): {early_energy:.6f} ({100*early_energy/total_energy:.1f}%)")
            print(f"  Mid (100-500ms): {mid_energy:.6f} ({100*mid_energy/total_energy:.1f}%)")
            print(f"  Late (500ms+): {late_energy:.6f} ({100*late_energy/total_energy:.1f}%)")
            
            # Check for energy discontinuities
            if ir_result.ndim > 1:
                mono_ir = np.mean(ir_result, axis=0)
            else:
                mono_ir = ir_result
                
            # Look for sudden jumps (derivative analysis)
            diff = np.abs(np.diff(mono_ir))
            max_jump = np.max(diff)
            max_jump_time = np.argmax(diff) / sample_rate * 1000
            
            print(f"\n📊 Continuity Check:")
            print(f"  Max jump: {max_jump:.6f} at {max_jump_time:.1f}ms")
            
            if max_jump_time > 150 and max_jump_time < 250:
                print(f"  ⚠️  Jump in crossfade region - may need adjustment")
            else:
                print(f"  ✅ No major jumps in crossfade region")
            
            # Success indicators
            if max_amplitude > 1e-6 and late_energy > early_energy * 0.1:
                print(f"\n🎉 SUCCESS: Hybrid tracer produced audible output with good energy distribution!")
                return True
            else:
                print(f"\n❌ Issues: max_amp={max_amplitude:.2e}, late_energy_ratio={late_energy/early_energy:.3f}")
                return False
                
        else:
            print("❌ Failed to generate hybrid IR")
            return False
            
    except Exception as e:
        print(f"❌ Error during hybrid tracing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_hybrid()
    if success:
        print("\n✅ IMPROVED HYBRID TEST PASSED")
    else:
        print("\n❌ IMPROVED HYBRID TEST FAILED")