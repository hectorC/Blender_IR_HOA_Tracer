# -*- coding: utf-8 -*-
"""
Ambisonic IR Tracer for Blender

A Blender add-on that renders third-order ambisonic (ACN/SN3D) impulse responses 
directly from scene geometry using forward and reverse ray-tracing strategies.

Features:
- Third-order ambisonic encoding with configurable orientation offsets
- Forward (stochastic) and reverse (specular) tracing modes
- Per-object acoustic materials with frequency-dependent properties
- Russian roulette termination and micro-roughness controls
- Simple diffraction sampling and frequency-dependent air absorption
- Batch averaging across multiple randomized passes
"""

bl_info = {
    "name": "Ambisonic IR Tracer",
    "blender": (4, 5, 0),
    "category": "Object",
    "author": "ChatGPT + Hector Centeno",
    "description": "Trace impulse responses with 3rd-order ambisonic encoding (ACN/SN3D) using reverse ray tracing with specular and diffuse reflections",
    "version": (1, 0, 1),
    "location": "3D Viewport > Sidebar > IR Tracer",
    "doc_url": "",
    "tracker_url": "",
}

import bpy
from typing import List


# Import UI components
from .ui.properties import register_acoustic_props, unregister_acoustic_props
from .ui.panels import (
    AIRT_PT_Panel, 
    AIRT_PT_MaterialPanel, 
    AIRT_PT_AdvancedPanel
)
from .ui.operators import (
    AIRT_OT_RenderIR,
    AIRT_OT_ValidateScene,
    AIRT_OT_ResetMaterial,
    AIRT_OT_CopyMaterial,
    AIRT_OT_DiagnoseScene
)


# List of classes to register
classes = [
    # Panels
    AIRT_PT_Panel,
    AIRT_PT_MaterialPanel,
    AIRT_PT_AdvancedPanel,
    
    # Operators
    AIRT_OT_RenderIR,
    AIRT_OT_ValidateScene,
    AIRT_OT_ResetMaterial,
    AIRT_OT_CopyMaterial,
    AIRT_OT_DiagnoseScene,
]


def register():
    """Register all addon classes and properties."""
    # Register properties first
    register_acoustic_props()
    
    # Register UI classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            print(f"Warning: Failed to register {cls.__name__}: {e}")
    
    print("Ambisonic IR Tracer: Registered successfully")


def unregister():
    """Unregister all addon classes and properties."""
    # Unregister UI classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except ValueError as e:
            print(f"Warning: Failed to unregister {cls.__name__}: {e}")
    
    # Unregister properties
    unregister_acoustic_props()
    
    print("Ambisonic IR Tracer: Unregistered successfully")


if __name__ == "__main__":
    register()