## Hybrid Crossfade UI Controls Implementation

### New UI Properties Added

1. **`airt_hybrid_crossfade_start_ms`** (Float, default: 50.0ms)
   - Range: 0.0 to 500.0ms
   - Controls when the crossfade from forward to reverse begins
   - Earlier = more diffuse sound, Later = more discrete echoes

2. **`airt_hybrid_crossfade_length_ms`** (Float, default: 150.0ms)
   - Range: 10.0 to 1000.0ms  
   - Controls the duration of the crossfade transition
   - Shorter = abrupt change, Longer = smoother blend

3. **Existing Gain Controls Repurposed:**
   - `airt_hybrid_forward_gain_db` (-24 to +24 dB, default: 0.0)
   - `airt_hybrid_reverse_gain_db` (-24 to +24 dB, default: 0.0)

### Implementation Flow

```
Forward IR → Remove Direct → Normalize → Apply Forward Gain → \
                                                                → Crossfade → Final Normalize → Output
Reverse IR → Remove Direct → Normalize → Apply Reverse Gain → /
```

### Gain Application Timing

The gain adjustments are applied **after individual post-processing** (direct removal + normalization) but **before crossfading** and **before final normalization**. This allows:

- Clean boost/cut of normalized individual components
- Maintains proper balance between forward and reverse
- Final normalization ensures 0 dBFS output regardless of gain settings

### UI Panel Layout

The hybrid controls panel now shows:

```
🎛️ Hybrid Crossfade Controls
├── Gain Adjustments:
│   ├── Early: [Forward Gain dB slider]
│   └── Reverb: [Reverse Gain dB slider]
├── Crossfade Timing:
│   ├── Start Time: [Start ms slider]  
│   └── Fade Length: [Length ms slider]
├── Legacy (unused):
│   └── Reverb Onset: [Old parameter, for compatibility]
└── [Preset buttons: Tunnel/Corridor | Cathedral | Reset]
```

### Default Values Match Current Behavior

- **Crossfade Start**: 50ms (matches old early_time_ms)
- **Crossfade Length**: 150ms (matches old late_time_ms - early_time_ms = 200-50)
- **Forward Gain**: 0dB (no change)
- **Reverse Gain**: 0dB (no change)

### User Control Benefits

1. **Timing Flexibility**: Adjust when discrete echoes give way to diffuse reverb
2. **Balance Control**: Boost/cut individual components without affecting normalization
3. **Room Adaptation**: Different spaces may need different crossfade timing
4. **Creative Control**: Fine-tune the hybrid character for musical/artistic purposes

### Technical Notes

- Gain is applied in linear domain after dB-to-linear conversion: `10^(dB/20)`
- Crossfade uses smooth linear interpolation between start and end times
- All timing is sample-accurate based on the project sample rate
- Bounds checking ensures crossfade parameters stay within valid ranges

The implementation is fully backwards compatible and ready for testing!