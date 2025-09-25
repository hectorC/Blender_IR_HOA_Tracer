## Final Parameter Fix for New Hybrid Workflow

### Root Cause Identified

The error `AttributeError: 'float' object has no attribute 'location'` was caused by incorrect parameter handling in the new hybrid method.

### Key Understanding

1. **`get_scene_sources(context)`** returns `List[mathutils.Vector]` (positions, not objects)
2. **`get_scene_receivers(context)`** returns `List[mathutils.Vector]` (positions, not objects)  
3. **`trace_impulse_response()`** signature expects `source: mathutils.Vector, receiver: mathutils.Vector`

### The Fix

**Before (incorrect):**
```python
def _trace_new_hybrid(self, context, sources, receivers, bvh, obj_map):
    source_pos = sources[0].location.copy()  # ❌ sources[0] is already a Vector, not an object
    receiver_pos = receivers[0].location.copy()  # ❌ receivers[0] is already a Vector, not an object
```

**After (correct):**
```python  
def _trace_new_hybrid(self, context, source_pos, receiver_pos, bvh, obj_map):
    # source_pos and receiver_pos are already mathutils.Vector objects ✅
```

### Integration Consistency

The new method now has the **exact same signature** as the existing `trace_impulse_response()`:

```python
# Existing function signature:
def trace_impulse_response(context, source: mathutils.Vector, receiver: mathutils.Vector, bvh, obj_map, directions=None)

# New method signature (now matches):
def _trace_new_hybrid(self, context, source_pos, receiver_pos, bvh, obj_map)
```

### Calling Pattern

Both functions are called identically from the main execute loop:

```python
if scene.airt_trace_mode == 'HYBRID':
    ir_pass = self._trace_new_hybrid(context, source, receiver, bvh, obj_map)  # ✅ source/receiver are Vectors
else:
    ir_pass = trace_impulse_response(context, source, receiver, bvh, obj_map)  # ✅ source/receiver are Vectors
```

### Verification

- ✅ **Parameter types**: Both methods expect `mathutils.Vector` positions
- ✅ **Calling convention**: Both called with identical parameters 
- ✅ **Ray tracer usage**: Both use `create_ray_tracer()` factory and `trace_rays()` method
- ✅ **Configuration**: Both use `RayTracingConfig(context)` constructor
- ✅ **Syntax validation**: No compilation errors

The implementation is now **correctly integrated** and should work without parameter errors in Blender.