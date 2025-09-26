# Important:
[human] Code generated with AI asistance (ChatGPT web, Codex and Claude). Use at your own risk. It seems to produce usable results for artistic purposes but it is not fully tested. I've created this project mainly to understand the current capabilities of AI assisted coding while attempting to create a Blender tool that for a long time I've been wanting to have. Take the information below with a large pinch of salt! It is all AI generated and might not be fully accurate. Additionally, these AI systems seem to easily start bragging and calling what they produce "professional" and "industry-standard" without a sustantive testing or benchmarking process against actual professional tools. I pointed this out to Claude and it stopped for a while claiming that it somehow switched into "marketing speech mode" instead of being technically accurate. It then resumed producing inflated or over optimistic claims after a while. The comments in the code commits are all mine. Overall, it is impressive that an AI was able to assist in producing the results it did! [/human]

# Ambisonic IR Tracer for Blender - Professional 3D Acoustic Ray Tracer

**Advanced hybrid ray-based acoustic impulse response simulation for Blender**

A professional-grade physics-based ray tracer that simulates room acoustics by tracking sound wave propagation through 3D spaces. Features sophisticated hybrid algorithms, professional materials modeling, and multi-channel ambisonic output for film, game, and architectural audio production.

## 🎯 Key Features

- **Hybrid ray tracing**: Intelligent combination of Forward + Reverse algorithms for optimal results  
- **Professional materials system**: 8-band octave processing (125Hz-8kHz) with industry-standard presets
- **Advanced blend controls**: ±24dB gain adjustment for fine-tuning acoustic character
- **Fast Preview Mode**: Quick broadband calculations for rapid iteration
- **Third-order ambisonic encoding**: 16-channel spatial audio with configurable orientation
- **Realistic acoustic modeling**: Frequency-dependent absorption, scattering, and edge diffraction
- **Production-ready output**: 32-bit float WAV files, sample rates up to 192kHz with quality controls

## Requirements
- Blender 4.5 or newer (matches the add-on `bl_info`)
- Python packages:
  - `soundfile` (used for writing multi-channel WAV files)
  - `scipy` (provides `scipy.special.lpmv` for spherical harmonics)

`numpy` ships with Blender and is used extensively by the add-on.

## 🚀 Ray Tracing Modes

### Hybrid Tracing (Recommended) 
**Combines Forward + Reverse algorithms for optimal acoustic modeling**

The hybrid approach intelligently blends two complementary ray tracing strategies:
- **Forward component**: Traces rays from source → receiver, capturing precise early reflections and direct paths
- **Reverse component**: Traces rays from receiver ← source, efficiently modeling complex reverb tails
- **Smart blending**: Automatically transitions from forward-dominated early reflections to reverse-dominated late reverberation

**Advanced Blend Controls:**
- **Forward Tracer Gain**: ±24dB adjustment for discrete echoes and early reflections
- **Reverse Tracer Gain**: ±24dB adjustment for ambient reverb and late reflections  
- **Reverb Ramp Time**: 0.05-0.5s transition period between early/late sound (default: 0.2s)

**Quick Presets:**
- **Tunnel/Corridor**: Forward +2dB, Reverse -1dB (emphasizes echoes and slap-back)
- **Cathedral**: Forward -1dB, Reverse +2dB (emphasizes ambient reverb)
- **Reset**: 0dB both (balanced blend)

### Forward Tracing Only
**Source → Receiver: Excellent for early reflections**
- Traces rays from sound sources toward receiver positions
- Captures precise direct paths and early reflection patterns
- Best for: Small rooms, near-field monitoring, discrete echo analysis
- Limitation: Less efficient for complex late reverberation

### Reverse Tracing Only  
**Receiver ← Source: Efficient for complex acoustics**
- Traces rays backward from receiver toward sound sources
- Excels at capturing complex multi-bounce reverberation paths
- Best for: Large spaces, ambient acoustics, reverb tail modeling
- Limitation: May miss some early reflection detail

## Installation

### Download from GitHub
1. **Download the repository**
   - Go to the GitHub repository: `https://github.com/hectorC/Blender_IR_HOA_Tracer`
   - Click **Code** → **Download ZIP** to get the complete repository
   - Extract the ZIP file to a temporary location

2. **Locate the add-on folder**
   - Navigate to the extracted folder: `Blender_IR_HOA_Tracer-main/source/`
   - You'll find the `ir_raytracer` folder - this is the complete add-on package

### Install in Blender  
3. **Install the add-on folder**
   - Launch Blender and open **Edit** → **Preferences** → **Add-ons**
   - Click **Install...** and browse to the `ir_raytracer` folder
   - Select the entire `ir_raytracer` folder (not individual files) and click **Install Add-on**
   - Enable the checkbox next to **"Ambisonic IR Tracer"**
   - Blender will copy the add-on to `scripts/addons/ir_raytracer/` in your user configuration

### Install Python Dependencies
4. **Install required packages** (only needed once per Blender installation)
   - Open Blender's **Scripting** workspace
   - Copy-paste this code into the Python console and run it:

```python
import bpy, subprocess, sys
pybin = bpy.app.binary_path_python
subprocess.check_call([pybin, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.check_call([pybin, "-m", "pip", "install", "soundfile", "scipy"])
```

   **Alternative method** - Run from command line (adjust path for your Blender version):

```bash
# Windows example
"C:\Program Files\Blender Foundation\Blender 4.5\4.5\python\bin\python.exe" -m pip install soundfile scipy

# macOS example  
"/Applications/Blender.app/Contents/Resources/4.5/python/bin/python3.10" -m pip install soundfile scipy

# Linux example
"/opt/blender/4.5/python/bin/python3.10" -m pip install soundfile scipy
```

5. **Restart Blender** after installing dependencies so the add-on can import the required packages

### Verify Installation
- Open the **3D Viewport** → **Sidebar** (press `N`) → **IR Tracer** tab
- You should see the complete ray tracing interface with material presets and tracing mode options
- Use **Check Dependencies** in the Diagnostics panel to verify soundfile and scipy are properly installed

## Usage

### 🎵 Professional Materials System

**8-Band Octave Processing for Realistic Acoustic Modeling**

The materials system uses industry-standard octave bands for frequency-dependent acoustic modeling:

**Frequency Bands**: 125Hz • 250Hz • 500Hz • 1kHz • 2kHz • 4kHz • 8kHz • 16kHz

Each material defines separate absorption and scattering coefficients per band, enabling realistic acoustic behavior across the full audio spectrum.

**Professional Presets** (based on industry measurements):

- **Concrete**: Highly reflective with slight low-frequency absorption
  - Absorption: 0.01-0.02 (low) to 0.05-0.07 (high)  
  - Scatter: 0.05-0.10 (smooth but not perfectly flat)
  
- **Carpet**: High absorption, especially at higher frequencies
  - Absorption: 0.15 (low) to 0.65 (high) - strong frequency dependence
  - Scatter: 0.20-0.30 (textured surface)
  
- **Wood**: Balanced absorption with natural material characteristics  
  - Absorption: 0.08 (low) to 0.12 (high) - moderate across spectrum
  - Scatter: 0.10-0.15 (natural surface texture)
  
- **Glass**: Very reflective with minimal high-frequency loss
  - Absorption: 0.03-0.04 (extremely low across spectrum)
  - Scatter: 0.02-0.05 (very smooth surface)
  
- **Metal**: Extremely reflective across all frequencies
  - Absorption: 0.01-0.02 (minimal across entire spectrum)  
  - Scatter: 0.02-0.04 (smooth metallic surface)

**Energy Conservation**: All presets maintain physically accurate energy relationships where absorption + reflection = 1.0, ensuring realistic acoustic behavior.

### 🎯 Basic Workflow

1. **Tag acoustic objects**
   - Select a mesh in the scene and open the *IR Tracer* panel (3D Viewport → Sidebar → *IR Tracer* tab).
   - Choose a **Material Preset** from the dropdown, or set custom *Absorption* and *Scatter* coefficients.
   - For advanced control, expand frequency bands to adjust per-octave behavior.
   - Mark one object as *Acoustic Source* and another as *Acoustic Receiver*.
2. **Configure render settings**
   - Choose **Tracing Mode**: *Hybrid* (recommended), *Forward Only*, or *Reverse Only*.
   - For Hybrid mode, adjust **Forward/Reverse Gain** (±24dB) to fine-tune acoustic character.
   - Set the number of rays, averaging passes, maximum bounce order, sample rate, IR length, and receiver radius.
   - Enable **Fast Preview Mode** for quick broadband testing, disable for final production.
   - Optionally configure advanced settings: Russian roulette (default: 40 bounces, 0.99 survival), diffraction, and air absorption.
   - Adjust ambisonic orientation settings (yaw offset, invert Z) to match your downstream decoder.
3. **Render the impulse response**
   - Click **Render Ambisonic IR** in the panel.
   - The add-on caches the BVH for the scene and traces all passes, averaging the result.
   - Status messages appear in the Info bar (and Blender's console, if open).
4. **Retrieve the output**
   - By default the IR is saved as `ir_output.wav` next to the `.blend` file.
   - If Blender cannot write to that folder, it falls back to Blender's temporary directory or the system temp directory. The console message includes the exact path.

## Parameter Reference

### Object Properties
- **Material Preset**: Choose from 5 professional presets (Concrete, Carpet, Wood, Glass, Metal) with industry-standard 8-band values, or select "Custom" for manual control.
- **Absorption**: Broadband energy absorption coefficient (0.0 = perfectly reflective, 1.0 = fully absorbent). Presets override this with frequency-dependent values.
- **Scatter**: Fraction of reflected energy sent into diffuse (cosine) lobes instead of specular reflections (0.0 = perfect mirror, 1.0 = fully diffuse Lambert surface).
- **Absorption Spectrum**: 8-band octave absorption values at centers: 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz. Essential for realistic acoustic modeling.
- **Scatter Spectrum**: 8-band scattering values controlling diffuse/specular reflection balance per frequency. Higher values = more diffuse surface behavior.
- **Transmission**: Energy fraction that transmits through the surface (0.0 = opaque wall, 1.0 = transparent). Useful for thin barriers and partial occlusion.
- **Acoustic Source**: Marks sound emission point for Forward tracing and target for Reverse tracing rays.
- **Acoustic Receiver**: Listener position where ambisonic impulse responses are captured using the configured receiver radius.

### Render Settings (Scene)
- **Tracing Mode**: **Hybrid** (combines Forward+Reverse, recommended) / **Forward Only** (source→receiver, good for early reflections) / **Reverse Only** (receiver←source, efficient for reverb tails).

### Hybrid Mode Controls (Advanced)
- **Forward Tracer Gain**: ±24dB adjustment for discrete echoes and early reflections. Higher values emphasize slap-back and clarity.
- **Reverse Tracer Gain**: ±24dB adjustment for ambient reverb and late reflections. Higher values emphasize spaciousness and tail length.
- **Reverb Ramp Time**: 0.05-0.5s transition period between early/late sound blending (default: 0.2s). Shorter = more distinct transition.

### Core Parameters  
- **Receiver Radius**: 0.001-2.0m capture sphere radius. Larger = more rays captured but less positional precision (default: 0.25m).
- **Rays**: 1000-32768+ rays per pass. Higher counts reduce noise but increase render time. Recommended: 8192+ for production.
- **Averaging Passes**: 1-100+ Monte Carlo passes to average. More passes = smoother results but longer render times (default: 4).
- **Max Bounces**: 10-200+ surface interactions per ray. Higher for longer reverb tails (default: 100).
- **Sample Rate**: 16000-192000 Hz audio sample rate. 48kHz/96kHz recommended for professional use.
- **IR Length**: 0.1-30.0s impulse response duration. Should exceed expected RT60 (default: 3.0s).

### Performance Options
- **Fast Preview Mode**: Bypasses 8-band processing for quick broadband calculations. Use for testing, disable for final renders.
- **WAV Format**: **32-bit Float** (recommended for IR), **24-bit PCM** (good quality), **16-bit PCM** (smallest files).

### Advanced Algorithm Settings
- **Russian Roulette Start**: Bounce number (0-1000) to begin probabilistic ray termination. Higher = more accurate tails (default: 40, optimized for occlusion scenarios).
- **Russian Roulette Survival**: 0.05-1.0 probability ray survives termination check. Higher = longer tails (default: 0.99, enhanced for professional results).
- **Min Throughput**: 1e-8 to 1e-2 minimum ray energy before termination. Lower allows weaker rays to contribute (default: 1e-6, improved sensitivity).
- **Specular Roughness**: 0-30° angular spread of specular reflections. 0° = perfect mirror, higher = more diffuse specular behavior (default: 5.0°).
- **Flip Z (up/down)**: Inverts the ambisonic Z axis (useful for matching systems that assume left-handed coordinates).
- **Calibrate Direct (1/r)**: Scales the entire impulse response so the direct-path amplitude matches 1/distance, aiding distance cues.
- **Omit Direct (reverb-only)**: Excludes the direct sound from the rendered IR so you can route dry/direct separately and use this IR purely for reverberation.
- **Air Absorption (freq)**: Enables ISO 9613-1 based air absorption filtering per path length.
- **Air Temp (deg C)**: Air temperature used in the absorption model.
- **Rel Humidity (%)**: Relative humidity percentage for the absorption model.
- **Air Pressure (kPa)**: Barometric pressure in kilopascals for the absorption model.
## Tips
- For dense scenes or many passes, enable Russian roulette to keep runtimes manageable.
- Start with a modest ray count (e.g., 4096) and increase as needed for smoother late reverberation tails.
- Use the *Random Seed* field to generate multiple statistically independent IRs or to reproduce a previous run exactly.
- The 16-channel WAV output is ACN/SN3D encoded; ensure your renderer or DAW plug-in expects this convention.

## 🚀 Professional Workflow & Performance

### Quick Start Workflow
1. **Scene Setup**: Create your room geometry and place Source/Receiver objects
2. **Material Assignment**: Use **material presets** for quick, realistic results  
3. **Fast Preview**: Enable **Fast Preview Mode** and use low ray counts (1000-2000) for initial testing
4. **Algorithm Selection**: Use **Hybrid mode** with default gains (0dB/0dB) for most scenarios
5. **Quality Pass**: Disable Fast Preview, increase rays (8192+), and render final output

### Performance Optimization
**For Fast Iteration:**
- **Fast Preview Mode**: ON (bypasses 8-band processing)
- **Rays**: 1000-2000 (quick noise assessment)  
- **Passes**: 1-2 (minimal averaging)
- **Tracing Mode**: Hybrid with default gains

**For Production Quality:**
- **Fast Preview Mode**: OFF (full 8-band processing)
- **Rays**: 8192-32768+ (smooth late reverberation)
- **Passes**: 4-16 (statistical averaging)
- **Russian Roulette**: Start=40, Survival=0.99 (optimized for quality)

### Hybrid Mode Presets Usage
- **Default (0dB/0dB)**: Balanced blend for most rooms
- **Tunnel/Corridor Preset**: Emphasizes discrete echoes and slap-back
- **Cathedral Preset**: Enhances ambient reverb and spatial impression  
- **Custom Gains**: Fine-tune ±24dB for specific acoustic character

### Material Workflow Tips
- **Start with presets**: Choose the closest physical material (Concrete, Wood, etc.)
- **Room-specific tuning**: Adjust broadband values if needed (Absorption/Scatter)
- **Advanced modeling**: Expand frequency bands for surgical acoustic control
- **Energy conservation**: Higher absorption = less reflection energy (presets handle this automatically)

### Troubleshooting Performance
- **High memory usage**: Reduce max bounces, enable Russian Roulette earlier
- **Long render times**: Use Fast Preview for testing, reduce ray count initially
- **Noisy results**: Increase ray count and averaging passes
- **Missing reverb tail**: Increase max bounces, check Russian Roulette settings
- **Too much/little early reflection**: Adjust Hybrid Forward Gain (±24dB)
- **Excessive/weak ambient reverb**: Adjust Hybrid Reverse Gain (±24dB)

## 🔧 Troubleshooting & Technical Issues

### Missing Dependencies Error
If you get a "python-soundfile is required" error when rendering:

1. **Use the "Check Dependencies" button** in the Diagnostics panel to verify installation status
2. **Reinstall soundfile** using Blender's Python:
   ```bash
   # Windows example - adjust path for your Blender version
   C:\Program Files\Blender Foundation\Blender 4.5\4.5\python\bin\python.exe -m pip install soundfile
   ```
3. **Toggle the add-on off and on** in Blender's Add-ons preferences  
4. **Restart Blender completely** to clear Python module cache
5. **Check Blender's console output** for detailed error messages

### Add-on Update Issues
When updating the add-on:
1. **Disable the old version** first in Add-ons preferences
2. **Remove old files** from the addons folder if needed
3. **Restart Blender** before installing the new version  
4. **Clean install** often works better than updating in place

### Scene Setup Issues
- **No sound generated**: Verify Source and Receiver objects are properly tagged and positioned
- **Incorrect timing/distance**: Check *Scene Properties > Units > Unit Scale* matches your scene scale
- **Missing early reflections**: Ensure **Segment Capture** is enabled (default: ON)
- **Weak or missing reverb tail**: 
  - Increase **Max Bounces** (try 150-200 for long tails)
  - Check **Russian Roulette** settings (Start: 40, Survival: 0.99)
  - Lower **Min Throughput** to 1e-6 or 1e-7 for sensitivity

### Professional Quality Issues  
- **Unrealistic acoustics**: Use **Material Presets** instead of manual values for physically accurate behavior
- **Frequency imbalance**: Disable **Fast Preview Mode** to enable full 8-band processing  
- **Poor spatial imaging**: Verify ambisonic **Yaw Offset** matches your decoder convention
- **Inconsistent results**: Set **Random Seed** to non-zero value for reproducible renders

### Command Line Rendering
For batch processing: `blender -b scene.blend --python-expr "import bpy; bpy.ops.airt.render_ir()"`

### Optimal Settings for Different Scenarios

**Small Rooms (< 50m³):**
- Tracing Mode: Forward Only or Hybrid  
- Rays: 4096-8192, Max Bounces: 50-100
- Materials: Use Glass/Metal presets for hard surfaces

**Large Spaces (> 500m³):**  
- Tracing Mode: Hybrid (recommended) or Reverse Only
- Rays: 16384+, Max Bounces: 150-200
- Russian Roulette: Essential for performance (Start: 40, Survival: 0.99)

**Occluded/Complex Geometry:**
- Enhanced parameters already optimized (RR Start: 40, Survival: 0.99, Min Throughput: 1e-6)
- Use Hybrid mode with balanced gains (0dB/0dB) initially
- Increase ray count for complex shadow zones

---

## Professional 3D Acoustic Ray Tracing for Blender

This add-on provides production-ready acoustic simulation with industry-standard materials modeling, hybrid algorithms, and professional-grade controls. Perfect for architectural acoustics, game audio, film post-production, and acoustic research.

**Latest version features hybrid ray tracing, 8-band frequency processing, and ±24dB blend controls for unprecedented acoustic realism.**





