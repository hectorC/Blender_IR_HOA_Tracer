# Hybrid Tracer Blending Fix

## Problem Identified

The hybrid tracer was producing a "hump" in the decay curve around 2-3 seconds, where energy **increased** instead of decreasing naturally. Analysis showed:

- **At t=2.00s:** -68.8 dB (good decay)  
- **At t=3.00s:** -47.1 dB (energy INCREASED by 21.7 dB!)

## Root Cause

The original blending algorithm was **purely additive**:

```
Forward Weight:  1.0 (always, throughout entire IR)
Reverse Weight:  0.0 → 1.0 (ramps up after 200ms)
Total Energy:    1.0 + 1.0 = 2.0x (energy doubling!)
```

This caused the reverse tracer to **add** energy on top of the forward tracer instead of **replacing** it in the late field.

## Solution Implemented

Changed to an **energy-conserving crossfade** strategy:

```
Forward Weight:  1.0 → 0.4 (fades to make room for reverb)
Reverse Weight:  0.0 → 0.8 (builds up but balanced) 
Total Energy:    Max 1.2x (40% improvement!)
```

### Key Changes:

1. **Intelligent Energy Analysis**: Measures actual energy levels of both tracers
2. **Conservative Scaling**: Scales reverse tracer based on measured energy ratios  
3. **Complementary Weights**: Forward fades as reverse builds up
4. **Energy Conservation**: Total weight stays ≤ 1.2 instead of 2.0

## Expected Results

After applying this fix, you should see:

✅ **No more "hump"** - Energy should decrease smoothly from 2s to 3s  
✅ **Better RT60 accuracy** - Decay curve will be more natural  
✅ **Preserved echoes** - Forward tracer still contributes to late field (0.4x)  
✅ **Rich reverb** - Reverse tracer provides diffuse tail (0.8x)  
✅ **Balanced sound** - Neither tracer overpowers the other  

## Technical Details

The fix modifies `_blend_early_late()` in `ray_tracer.py`:

- **Energy Analysis**: Measures forward/reverse energy in early/mid/late windows
- **Intelligent Scaling**: Scales reverse tracer conservatively based on energy ratio
- **Crossfade Strategy**: Forward weight fades from 1.0 → 0.4, Reverse builds 0.0 → 0.8
- **Conservation Check**: Monitors total weight to ensure ≤ 1.2x energy

## User Controls Unchanged

All existing user controls remain functional:
- **Forward Gain**: -24dB to +24dB
- **Reverse Gain**: -24dB to +24dB  
- **Reverb Ramp Time**: 0.05s to 0.5s

## Test Your Results

After running the hybrid tracer with this fix:

1. Check decay curve for smooth energy decrease
2. Verify RT60 measurements are reasonable
3. Listen for balanced early reflections + reverb tail
4. No artificial energy "humps" should be present

The 40% energy reduction should eliminate the problematic hump while maintaining the best qualities of both tracers.