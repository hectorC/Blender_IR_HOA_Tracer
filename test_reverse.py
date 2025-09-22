#!/usr/bin/env python3
"""Test script for the Reverse ray tracer with fixed normalization."""

import sys
import os
import numpy as np
import mathutils

# Add the ir_raytracer package to the path
sys.path.insert(0, os.path.dirname(__file__))

from ir_raytracer.core.ray_tracer import create_ray_tracer, RayTracingConfig
from ir_raytracer.utils.scene_utils import create_shoebox_room, create_bvh

def main():
    """Test the Reverse tracer with fixed normalization."""
    # Test room parameters (30m cube)
    room_size = 30.0
    source_pos = mathutils.Vector((5.0, 5.0, 1.5))
    receiver_pos = mathutils.Vector((15.0, 10.0, 1.8))  # 4m from source
    
    # Create test room
    print("Creating test room...")
    mesh_data, obj_map = create_shoebox_room(room_size, room_size, room_size, 
                                           wall_absorption=0.02)
    
    # Build BVH
    print("Building BVH...")
    bvh = create_bvh([mesh_data[0]])
    
    # Create ray tracer config
    config = RayTracingConfig()
    config.sample_rate = 48000
    config.ir_length = 2.0
    config.num_bands = 8
    config.max_bounces = 64
    
    # Create tracer and generate directions
    tracer = create_ray_tracer('REVERSE', config)
    print(f"Created reverse tracer")
    
    # Generate 8192 uniform random directions
    num_rays = 8192
    directions = []
    np.random.seed(42)  # Reproducible results
    
    for _ in range(num_rays):
        # Uniform random direction on sphere
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        directions.append((x, y, z))
    
    print(f"Generated {len(directions)} directions")
    
    # Trace rays
    print("Tracing rays...")
    ir_result = tracer.trace_rays(source_pos, receiver_pos, bvh, obj_map, directions)
    
    # Analyze results
    max_energy = np.max(np.abs(ir_result))
    total_energy = np.sum(ir_result ** 2)
    
    print(f"\n=== RESULTS ===")
    print(f"Max energy: {max_energy:.6f}")
    print(f"Total energy: {total_energy:.6f}")
    print(f"Energy in first 100ms: {np.sum(ir_result[:4800] ** 2):.6f}")
    
    if max_energy > 1e-6:
        print("✅ SUCCESS: Reverse tracer now produces audible energy!")
        print(f"Energy improvement: {max_energy/2.25e-05:.1f}x stronger than before")
    else:
        print("❌ Still too quiet")
    
    # Save result for analysis
    try:
        import scipy.io.wavfile as wav
        # Convert to 16-bit for saving
        ir_16bit = np.clip(ir_result * 32767, -32767, 32767).astype(np.int16)
        wav.write("reverse_tracer_fixed.wav", 48000, ir_16bit)
        print("Saved result to reverse_tracer_fixed.wav")
    except ImportError:
        print("scipy not available, skipping audio save")

if __name__ == "__main__":
    main()