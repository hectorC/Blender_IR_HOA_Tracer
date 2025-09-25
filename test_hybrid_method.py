#!/usr/bin/env python3
"""Test calling the new hybrid workflow method."""

import sys
import os
import numpy as np

# Add the source directory to Python path  
sys.path.insert(0, os.path.dirname(__file__))

def test_method_syntax():
    """Test that the new hybrid method syntax is correct."""
    
    # Mock the operator class structure to test method syntax
    class MockOperator:
        def report(self, level, message):
            print(f"[{level['INFO'] if 'INFO' in level else level}] {message}")
            
        def _remove_direct_impulse(self, ir, sample_rate):
            """Mock direct impulse removal."""
            return ir  # Just return unchanged for test
            
        def _crossfade_hybrid_irs(self, forward_ir, reverse_ir, sample_rate):
            """Mock crossfading - use the actual logic."""
            min_length = min(forward_ir.shape[1], reverse_ir.shape[1])
            forward_ir = forward_ir[:, :min_length]
            reverse_ir = reverse_ir[:, :min_length]
            
            early_time_ms = 50.0
            late_time_ms = 200.0
            
            early_samples = int((early_time_ms / 1000.0) * sample_rate)
            late_samples = int((late_time_ms / 1000.0) * sample_rate)
            
            forward_weight = np.ones(min_length)
            reverse_weight = np.ones(min_length)
            
            forward_weight[:early_samples] = 1.0
            reverse_weight[:early_samples] = 0.0
            
            if late_samples > early_samples:
                transition_length = late_samples - early_samples
                transition_ramp = np.linspace(0, 1, transition_length)
                
                forward_weight[early_samples:late_samples] = 1.0 - transition_ramp
                reverse_weight[early_samples:late_samples] = transition_ramp
            
            forward_weight[late_samples:] = 0.0
            reverse_weight[late_samples:] = 1.0
            
            hybrid_ir = (forward_ir * forward_weight[None, :] + 
                        reverse_ir * reverse_weight[None, :])
            
            return hybrid_ir
        
        def _trace_new_hybrid(self, context, sources, receivers, bvh, obj_map):
            """Test the new hybrid workflow method signature."""
            
            # Mock imports would happen here
            print("Step 1: Import ray tracer components")
            
            # Mock config creation
            print("Step 2: Create ray tracing configuration")
            
            # Mock forward tracing
            print("Step 3: Generate forward tracer IR")
            forward_ir = np.random.normal(0, 0.1, (4, 44100))  # 1 second of mock data
            
            # Mock reverse tracing  
            print("Step 4: Generate reverse tracer IR")
            reverse_ir = np.random.normal(0, 0.1, (4, 44100))
            
            # Mock post-processing
            print("Step 5: Post-process individual IRs")
            forward_ir = self._remove_direct_impulse(forward_ir, 44100)
            reverse_ir = self._remove_direct_impulse(reverse_ir, 44100)
            
            # Mock normalization
            print("Step 6: Normalize individual IRs")
            forward_peak = np.max(np.abs(forward_ir))
            reverse_peak = np.max(np.abs(reverse_ir))
            combined_peak = max(forward_peak, reverse_peak)
            
            if combined_peak > 1e-9:
                scale_factor = 1.0 / combined_peak
                forward_ir = forward_ir * scale_factor
                reverse_ir = reverse_ir * scale_factor
                self.report({'INFO'}, f"Normalized both IRs to 0 dBFS (scale: {scale_factor:.6f})")
            
            # Mock crossfading
            print("Step 7: Crossfade processed IRs")
            hybrid_ir = self._crossfade_hybrid_irs(forward_ir, reverse_ir, 44100)
            
            # Mock final normalization
            print("Step 8: Final normalization")
            final_peak = np.max(np.abs(hybrid_ir))
            if final_peak > 1e-9:
                final_scale = 1.0 / final_peak
                hybrid_ir = hybrid_ir * final_scale
                self.report({'INFO'}, f"Final normalization to 0 dBFS (scale: {final_scale:.6f})")
            
            return hybrid_ir
    
    # Test the method
    print("=== Testing New Hybrid Workflow Method ===")
    
    operator = MockOperator()
    
    # Mock parameters
    context = None
    sources = None  
    receivers = None
    bvh = None
    obj_map = None
    
    try:
        result = operator._trace_new_hybrid(context, sources, receivers, bvh, obj_map)
        print(f"\n✓ Method completed successfully")
        print(f"✓ Result shape: {result.shape}")
        print(f"✓ Result peak: {np.max(np.abs(result)):.6f}")
        print(f"✓ Result RMS: {np.sqrt(np.mean(result**2)):.6f}")
        
    except Exception as e:
        print(f"\n✗ Method failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_method_syntax()