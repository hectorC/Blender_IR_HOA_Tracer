Code generated with AI asistance (ChatGPT).

# Ambisonic IR Tracer for Blender

Ambisonic IR Tracer is a Blender add-on that renders third-order ambisonic (ACN/SN3D) impulse responses directly from the geometry in your scene. It uses forward and reverse ray-tracing strategies with support for specular and diffuse reflections, per-object absorption/scatter controls, stochastic tails, and optional air absorption. The resulting 16-channel ambisonic WAV files can be used for spatial audio rendering, acoustic analysis, or convolution-based reverberation.

## Features
- Third-order ambisonic encoding with configurable orientation offsets
- Forward (stochastic) and reverse (specular) tracing modes with multi-bounce paths
- Per-object wideband absorption, scattering, and source/receiver tagging
- Receiver radius capture, Russian roulette termination, and micro-roughness controls
- Frequency-dependent air absorption based on ISO 9613-1 parameters
- Optional calibration of the direct path to a 1/r amplitude reference
- Batch averaging across multiple randomized passes with reproducible seeding

## Requirements
- Blender 4.5 or newer (matches the add-on `bl_info`)
- Python packages:
  - `soundfile` (used for writing multi-channel WAV files)
  - `scipy` (provides `scipy.special.lpmv` for spherical harmonics)

`numpy` ships with Blender and is used extensively by the add-on.

## Installation
1. **Download the add-on**
   - Clone or download this repository, or create a ZIP archive containing `ir_raytracer.py`.
2. **Install in Blender**
   - Launch Blender and open *Edit ? Preferences ? Add-ons*.
   - Click *Install�*, browse to the ZIP (or `ir_raytracer.py`), and confirm.
   - Enable the checkbox next to **Ambisonic IR Tracer**.
   - Blender will store the add-on under `scripts/addons` in your user configuration.
3. **Install Python dependencies** (only needs to be done once per Blender installation)
   - Open Blender�s *Scripting* workspace.
   - Copy-paste the snippet below into the Python console or a text block and run it:

```python
import bpy, subprocess, sys
pybin = bpy.app.binary_path_python
subprocess.check_call([pybin, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.check_call([pybin, "-m", "pip", "install", "soundfile", "scipy"])
```

   - Alternatively, from a shell you can run Blender�s bundled Python directly (adjust the path to match your platform):

```bash
"/path/to/Blender/4.5/python/bin/python3.10" -m pip install --upgrade pip
"/path/to/Blender/4.5/python/bin/python3.10" -m pip install soundfile scipy
```

   - Restart Blender after installing the packages so the add-on can import them.

## Usage
1. **Tag acoustic objects**
   - Select a mesh in the scene and open the *IR Tracer* panel (3D Viewport ? Sidebar ? *IR Tracer* tab).
   - Set per-object *Absorption* and *Scatter* coefficients.
   - Mark one object as *Acoustic Source* and another as *Acoustic Receiver*.
2. **Configure render settings**
   - Choose *Forward* or *Reverse* tracing mode.
   - Set the number of rays, averaging passes, maximum bounce order, sample rate, IR length, and receiver radius.
   - Optionally enable segment capture, Russian roulette, micro-roughness, direct-path calibration, and air absorption parameters (temperature, humidity, pressure).
   - Adjust orientation settings (yaw offset, invert Z) to match your downstream ambisonic decoder.
3. **Render the impulse response**
   - Click **Render Ambisonic IR** in the panel.
   - The add-on caches the BVH for the scene and traces all passes, averaging the result.
   - Status messages appear in the Info bar (and Blender�s console, if open).
4. **Retrieve the output**
   - By default the IR is saved as `ir_output.wav` next to the `.blend` file.
   - If Blender cannot write to that folder, it falls back to Blender�s temporary directory or the system temp directory. The console message includes the exact path.

## Parameter Reference

### Object Properties
- **Absorption**: Wideband energy absorption coefficient (0 reflective, 1 fully absorbing).
- **Scatter**: Fraction of reflected energy sent into diffuse (cosine) lobes instead of specular reflections (0 = specular, 1 = fully diffuse).
- **Acoustic Source**: Marks the object whose origin emits sound for reverse tracing and defines the starting point for forward tracing rays.
- **Acoustic Receiver**: Marks the listener position; impulse responses are captured at this location using the configured receiver radius.

### Render Settings (Scene)
- **Tracing Mode** (`Forward` / `Reverse`): Selects stochastic forward tracing with receiver capture or deterministic reverse tracing with line-of-sight connection checks.
- **Receiver Radius (m)**: Physical radius of the receiver capture sphere in meters; converted to Blender units based on the scene scale.
- **Rays**: Number of primary rays shot per pass; higher counts reduce variance but increase render time.
- **Averaging Passes**: Number of Monte Carlo passes to average; each pass optionally uses a different random seed.
- **Max Bounces**: Upper bound on the number of surface interactions per ray; set higher for long reverberation tails.
- **Sample Rate**: Audio sample rate of the rendered impulse response (Hz).
- **IR Length (s)**: Duration of the output impulse response in seconds.
- **Specular Tol (deg)**: Angular tolerance for matching specular reflections to the receiver direction (Gaussian falloff in degrees).
- **WAV Subtype**: Output PCM or float encoding used by `soundfile` when writing the WAV.
- **Specular Roughness (deg)**: Adds micro-jitter to specular reflections, broadening highlights by the given cone angle.
- **Capture Along Segments**: When enabled, accumulates energy for partial segments that graze the receiver sphere, increasing early reflection density.
- **Random Seed**: Base RNG seed; each pass adds its index to derive a reproducible sequence. Zero leaves the RNG unseeded.
- **Russian Roulette**: Toggles probabilistic termination of long ray paths to reduce runtime while maintaining expected energy.
- **RR Start Bounce**: Bounce index at which Russian roulette begins evaluating termination probability.
- **RR Survive Prob**: Probability that a ray survives each Russian roulette check; higher values retain more energy at the cost of performance.
- **Yaw Offset (deg)**: Rotates the ambisonic orientation about the vertical axis to align with downstream decoder conventions.
- **Flip Z (up/down)**: Inverts the ambisonic Z axis (useful for matching systems that assume left-handed coordinates).
- **Calibrate Direct (1/r)**: Scales the entire impulse response so the direct-path amplitude matches 1/distance, aiding distance cues.
- **Air Absorption (freq)**: Enables ISO 9613-1 based air absorption filtering per path length.
- **Air Temp (deg C)**: Air temperature used in the absorption model.
- **Rel Humidity (%)**: Relative humidity percentage for the absorption model.
- **Air Pressure (kPa)**: Barometric pressure in kilopascals for the absorption model.
## Tips
- For dense scenes or many passes, enable Russian roulette to keep runtimes manageable.
- Start with a modest ray count (e.g., 4096) and increase as needed for smoother late reverberation tails.
- Use the *Random Seed* field to generate multiple statistically independent IRs or to reproduce a previous run exactly.
- The 16-channel WAV output is ACN/SN3D encoded; ensure your renderer or DAW plug-in expects this convention.

## Troubleshooting
- If the add-on reports missing dependencies, rerun the installation commands above using Blender's bundled Python interpreter.
- When scripting renders from the command line (`blender -b`), pass `--python-expr "import bpy; bpy.ops.airt.render_ir()"` after setting up the scene in advance.
- For scenes with scaled units, the add-on compensates via the scene's unit scale; verify the *Unit Scale* value in *Scene Properties > Units* if delays or attenuation feel incorrect.

Enjoy tracing ambisonic impulse responses directly inside Blender!



