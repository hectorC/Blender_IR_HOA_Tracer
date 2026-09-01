# -*- coding: utf-8 -*-
"""
Ambisonic IR Tracer for Blender

A Blender add-on that renders third-order ambisonic (ACN/SN3D) impulse responses
from scene geometry using receiver-centric acoustic energy transport.

Features:
- Third-order ambisonic encoding with configurable orientation offsets
- Deterministic direct/early paths and stochastic diffuse transport
- Per-object acoustic materials with frequency-dependent properties
- Russian roulette termination and frequency-dependent air absorption
- Optional bounded edge diffraction
"""

bl_info = {
    "name": "Ambisonic IR Tracer",
    "blender": (5, 2, 1),
    "category": "Object",
    "author": "ChatGPT + Hector Centeno",
    "description": "Create artistic 3rd-order ambisonic IRs from Blender geometry and acoustic materials",
    "version": (2, 0, 0),
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
    AIRT_PT_AudioPanel,
    AIRT_PT_AdvancedPanel,
    AIRT_PT_DiagnosticsPanel
)
from .ui.operators import (
    AIRT_OT_RenderIR,
    AIRT_OT_AssignSource,
    AIRT_OT_AssignReceiver,
    AIRT_OT_ValidateScene,
    AIRT_OT_ResetMaterial,
    AIRT_OT_CopyMaterial,
    AIRT_OT_CheckDependencies,
)


# List of classes to register
classes = [
    # Panels
    AIRT_PT_Panel,
    AIRT_PT_MaterialPanel,
    AIRT_PT_AudioPanel,
    AIRT_PT_AdvancedPanel,
    AIRT_PT_DiagnosticsPanel,
    
    # Operators
    AIRT_OT_RenderIR,
    AIRT_OT_AssignSource,
    AIRT_OT_AssignReceiver,
    AIRT_OT_ValidateScene,
    AIRT_OT_ResetMaterial,
    AIRT_OT_CopyMaterial,
    AIRT_OT_CheckDependencies,
]


def register():
    """Register all addon classes and properties."""
    # Check critical dependencies and warn user
    try:
        import soundfile as sf
        print("Ambisonic IR Tracer: soundfile dependency found")
    except ImportError:
        print("WARNING: Ambisonic IR Tracer - soundfile not found!")
        print("Install with: python -m pip install soundfile")
        print("Or use Blender's Python: [Blender]/python/bin/python.exe -m pip install soundfile")
    
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
