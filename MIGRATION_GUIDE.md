# Migration Guide: Monolithic to Modular Structure

## Overview
The Ambisonic IR Tracer has been restructured from a single 2,500-line file into a well-organized modular package. This improves maintainability, readability, and extensibility.

## New Structure
```
ir_raytracer/                    # Main package
├── __init__.py                  # Registration and main entry point
├── core/                        # Core algorithms
│   ├── __init__.py
│   ├── acoustics.py            # Material properties, air absorption
│   ├── ambisonic.py            # Ambisonic encoding, orientation
│   └── ray_tracer.py           # Ray tracing engine
├── ui/                         # User interface
│   ├── __init__.py
│   ├── properties.py           # Blender property definitions
│   ├── panels.py               # UI panels
│   └── operators.py            # Blender operators
└── utils/                      # Utilities
    ├── __init__.py
    ├── math_utils.py           # Mathematical utilities
    └── scene_utils.py          # Scene management utilities
```

## Key Improvements

### 1. Separation of Concerns
- **Core algorithms** isolated from UI code
- **Material properties** centralized in acoustics module
- **Mathematical utilities** extracted to reusable functions

### 2. Better Error Handling
- Added scene validation operator
- Improved dependency checking
- More informative error messages

### 3. Enhanced UI Organization  
- Split complex panel into logical sections
- Added material-specific panel
- Advanced settings in collapsible sections

### 4. Code Quality Improvements
- Type hints throughout (where possible with Blender)
- Consistent naming conventions
- Reduced code duplication
- Better documentation

## Migration Steps

### For Users
1. **Backup** your current .blend files
2. **Uninstall** the old addon version
3. **Install** the new modular version
4. **Test** your existing scenes - all parameters should be preserved

### For Developers
1. **Core algorithms** are now in `ir_raytracer.core.*`
2. **UI components** are in `ir_raytracer.ui.*`
3. **Utilities** are in `ir_raytracer.utils.*`
4. **Import paths** have changed - update any custom scripts

## Backwards Compatibility
- All Blender properties remain the same
- Existing .blend files will work without changes
- Same output file format and naming

## New Features
- **Scene validation** operator (`airt.validate_scene`)
- **Material copying** between objects (`airt.copy_material`)
- **Reset materials** to defaults (`airt.reset_material`)
- **Better progress reporting** during renders
- **Improved error messages** with specific suggestions

## Configuration Classes
The new structure introduces configuration classes for better parameter management:

```python
# Example: accessing configuration programmatically
from ir_raytracer.core.ray_tracer import RayTracingConfig

config = RayTracingConfig(context)
print(f"Ray count: {config.num_rays}")
print(f"Max bounces: {config.max_bounces}")
```

## Performance Notes
- **Same ray tracing algorithms** - no performance regression
- **Better memory management** through class structure
- **Preparation** for future optimizations (ray packets, SIMD)

## Testing
After migration, test these key features:
1. Material presets still work
2. Forward/reverse tracing modes function correctly  
3. Frequency-dependent absorption/scattering
4. Air absorption model
5. Direct path calibration
6. Multi-pass averaging

## Future Roadmap
The modular structure enables:
- **Ray packet tracing** for SIMD performance
- **GPU acceleration** through compute shaders
- **Real-time preview** mode
- **Plugin architecture** for custom materials
- **Advanced diffraction** models
- **Room acoustics** analysis tools

## Support
If you encounter issues:
1. Check the **Console** for detailed error messages
2. Use **Scene Validation** operator to identify problems
3. Compare with **working .blend** files from previous version
4. Report issues with **specific steps to reproduce**