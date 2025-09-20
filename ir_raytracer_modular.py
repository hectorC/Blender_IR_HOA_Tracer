# -*- coding: utf-8 -*-
"""
Modular Ambisonic IR Tracer for Blender

This is the new modular version of the Ambisonic IR Tracer. 
To use this version, rename this file to replace ir_raytracer.py and 
ensure the ir_raytracer/ package directory exists alongside it.

The modular structure provides:
- Better code organization and maintainability
- Separation of concerns between UI, physics, and utilities  
- Easier testing and debugging
- Cleaner import structure
"""

bl_info = {
    "name": "Ambisonic IR Tracer (Modular)",
    "blender": (4, 5, 0),
    "category": "Object",
    "author": "ChatGPT + Hector Centeno",
    "description": "Modular version: Trace impulse responses with 3rd-order ambisonic encoding (ACN/SN3D)",
    "version": (1, 0, 1),
    "location": "3D Viewport > Sidebar > IR Tracer",
}

# Import the modular addon
from .ir_raytracer import register, unregister

if __name__ == "__main__":
    register()