#!/usr/bin/env python3
"""Test the hybrid ray tracer implementation."""

import sys
import os
import numpy as np

# Simulate Blender context for testing
class MockScene:
    def __init__(self):
        self.airt_trace_mode = 'HYBRID'  # Test hybrid mode
        # Add other required properties
        self.airt_num_rays = 8192
        self.airt_max_bounces = 64
        self.airt_rr_enable = True
        self.airt_rr_start = 3
        self.airt_rr_p = 0.5
        self.airt_ir_length = 2.0
        self.airt_sample_rate = 48000
        self.airt_hoa_order = 3
        self.airt_use_lf = True
        self.airt_air_absorption = True
        self.airt_omit_direct = False

class MockContext:
    def __init__(self):
        self.scene = MockScene()

# Add paths for testing
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Mock mathutils for testing
    import mathutils
    print("✓ Using real mathutils")
except ImportError:
    # Create mock mathutils if not available
    class MockVector:
        def __init__(self, coords):
            self.x, self.y, self.z = coords[:3]
        def normalized(self):
            return self
        def __getitem__(self, i):
            return [self.x, self.y, self.z][i]
    
    class MockMathutils:
        Vector = MockVector
    
    sys.modules['mathutils'] = MockMathutils()
    mathutils = MockMathutils()
    print("✓ Using mock mathutils")

try:
    from ir_raytracer.core.ray_tracer import trace_impulse_response, generate_ray_directions
    from ir_raytracer.utils.scene_utils import create_shoebox_room, create_bvh
    
    def test_hybrid_tracer():
        """Test the hybrid ray tracer."""
        print("=== TESTING HYBRID RAY TRACER ===")
        
        # Setup test parameters
        context = MockContext()
        
        # Test room (same 30m cube as before)
        room_size = 30.0
        source_pos = mathutils.Vector((5.0, 5.0, 1.5))
        receiver_pos = mathutils.Vector((15.0, 10.0, 1.8))
        
        print(f"Room: {room_size}m cube")
        print(f"Source: {source_pos.x}, {source_pos.y}, {source_pos.z}")
        print(f"Receiver: {receiver_pos.x}, {receiver_pos.y}, {receiver_pos.z}")
        
        # Create room
        print("\nCreating test room...")
        mesh_data, obj_map = create_shoebox_room(room_size, room_size, room_size,
                                               wall_absorption=0.02)
        
        # Build BVH
        print("Building BVH...")
        bvh = create_bvh([mesh_data[0]])
        
        # Generate directions
        print(f"Generating {context.scene.airt_num_rays} ray directions...")
        directions = generate_ray_directions(context.scene.airt_num_rays)
        
        # Trace impulse response with HYBRID mode
        print(f"\n🚀 Starting HYBRID tracing...")
        ir_result = trace_impulse_response(context, source_pos, receiver_pos, 
                                         bvh, obj_map, directions)
        
        # Analyze results
        max_energy = np.max(np.abs(ir_result))
        total_energy = np.sum(ir_result ** 2)
        
        # Energy distribution analysis
        sample_rate = 48000
        early_samples = int(0.1 * sample_rate)  # 0-100ms
        mid_samples = int(0.5 * sample_rate)    # 100-500ms
        late_samples = len(ir_result[0])        # 500ms-end
        
        early_energy = np.sum(ir_result[:, :early_samples] ** 2)
        mid_energy = np.sum(ir_result[:, early_samples:mid_samples] ** 2)
        late_energy = np.sum(ir_result[:, mid_samples:] ** 2)
        
        print(f"\n=== HYBRID TRACER RESULTS ===")
        print(f"Max energy: {max_energy:.6f}")
        print(f"Total energy: {total_energy:.6f}")
        print(f"Early energy (0-100ms): {early_energy:.6f}")
        print(f"Mid energy (100-500ms): {mid_energy:.6f}")
        print(f"Late energy (500ms+): {late_energy:.6f}")
        
        # Success criteria
        if max_energy > 0.01:
            print("✅ SUCCESS: Strong energy levels")
        else:
            print("❌ Energy still too low")
            
        if early_energy > mid_energy > late_energy:
            print("✅ SUCCESS: Proper energy decay profile")
        else:
            print("⚠️  WARNING: Unusual energy distribution")
        
        # Save result
        try:
            import scipy.io.wavfile as wav
            ir_mono = ir_result[0]  # Use first channel
            ir_16bit = np.clip(ir_mono * 32767, -32767, 32767).astype(np.int16)
            wav.write("hybrid_tracer_result.wav", 48000, ir_16bit)
            print("💾 Saved to hybrid_tracer_result.wav")
        except ImportError:
            print("💾 scipy not available, skipping save")
        
        return ir_result
    
    if __name__ == "__main__":
        test_hybrid_tracer()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the source directory")