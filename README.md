# Ambisonic IR Tracer for Blender

Ambisonic IR Tracer turns Blender geometry into 16-channel, third-order
ambisonic impulse responses for convolution reverb, sound design, and
electroacoustic art.

This is a creative, physics-inspired renderer. It aims for plausible spatial
character and useful artistic control; it is not validated acoustic-engineering
software and should not be used for building or safety decisions.

## Current architecture

Version 2.3.1 combines deterministic direct sound and specular reflections
with bidirectional acoustic-energy transport.

- Direct sound is evaluated deterministically, including per-band transmission.
- Source radiation is evaluated in the source object's local frame for every
  direct, reflected, and diffracted path, with frequency-dependent focus.
- Planar specular reflections are found deterministically with finite image
  sources through a selectable first, second, or third order. With
  **Pressure-Coherent Early Paths** enabled, their signed pressure transfer is
  evaluated at the real angle of incidence using a simple passive wall
  impedance inferred from each material's reflected-energy bands.
- Remaining reflected energy is sampled from both source and receiver with
  diffuse, glossy, absorptive, and transmissive surface interactions. Retained
  subpaths are joined to form additional source-to-receiver routes.
- The receiver-only, source-only, and joined estimators are combined per path
  order with uniform multiple-importance weights. All-specular paths already
  covered by the image-source stage are omitted from the stochastic estimator.
- Arrivals between audio samples use 65-tap Kaiser-windowed sinc interpolation,
  with unity DC gain and timing compensated for the filter's support.
- Monte Carlo energy events are converted to pressure with repeatable randomized
  polarity per band and a power-complementary seven-band filter bank. Adjacent
  bands use sine/cosine crossovers on a log-frequency scale, so independent band
  powers combine at approximately constant level. Default finite filters keep
  the equal-band power response within 0.15 dB of unity.
- All 16 channels share each event's timing, band polarities, and filters; its
  directional encoding is preserved throughout reconstruction.
- Every arrival is encoded directly to third-order ACN/SN3D (AmbiX) using its
  listener-relative arrival direction.
- Optional bounded single-edge diffraction supplies a creative shadow-zone
  approximation.

Direct, early, and stochastic reflected components use the same scene geometry
and level convention.

## Requirements and installation

- Blender 5.2.1 LTS
- `numpy` (included with Blender)
- `soundfile` in Blender's Python environment for multichannel WAV export

This repository contains a **legacy Python add-on**, not a packaged Blender
extension: registration uses `bl_info`, and there is no extension manifest.

1. Install `soundfile` into the Python environment used by Blender, not merely
   into your system Python. For the default Windows installation:

   ```powershell
   & 'C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe' -m pip install soundfile
   ```

   Adjust the path for your installation; writing under Program Files may
   require an elevated terminal.
2. Create a ZIP containing the `ir_raytracer/` folder at its root, with
   `ir_raytracer/__init__.py` and its `core/`, `ui/`, and `utils/` subfolders.
   Do not zip only `__init__.py` or add an extra repository-folder wrapper.
3. In **Edit > Preferences > Add-ons**, use **Install from Disk** to select that
   ZIP, then enable **Ambisonic IR Tracer**. See Blender's
   [legacy add-on installation guide](https://docs.blender.org/manual/en/5.2/editors/preferences/addons.html#installing-legacy-add-ons).
4. Open **3D Viewport > Sidebar > IR Tracer**. The **Diagnostics** panel's
   **Check Audio Dependency** action confirms whether `soundfile` is available.

Updating the Git checkout does not update a separately installed add-on copy.
Reinstall the updated package or synchronize the installed `ir_raytracer/`
folder, then restart Blender before testing the new code.

## Basic workflow

1. Model a closed or partly open space at a meaningful Blender unit scale.
2. Select an object at the emitter position and choose **Use Active as Source**.
3. In **Source Radiation**, keep the neutral even pattern or choose and aim a
   directional shape. Source-local `+Y` is forward.
4. Select an object at the listening position and choose **Use Active as
   Receiver**. Empty objects work well for both endpoints.
5. Select each acoustic mesh and choose an Acoustic Material preset. Expand
   **Manual Band Details** when frequency-specific shaping is wanted.
6. Choose IR content, duration, sample rate, quality, and an output path.
7. Run **Validate Acoustic Scene**, then **Render Ambisonic IR**.

### Geometry and endpoint objects

Each render uses one source and one receiver at their **evaluated object
origins**, not at their mesh centers or surfaces. They are ideal points: their
shape, dimensions, and object scale do not define a radiating area or listening
radius. Rotation aims source radiation and, when enabled, orients the receiver's
sound field. Parent transforms can change the endpoints' world positions and
orientations; scaling a parent can therefore move a child endpoint indirectly.

The acoustic scene uses **visible mesh objects in the current view layer**,
with evaluated modifiers and world transforms. Viewport-hidden meshes are
excluded; the render-visibility flag alone is not the acoustic inclusion switch.
Convert non-mesh objects such as curves or text to meshes to include them.
Scene **Unit Scale** determines physical distances: at 1.0, one Blender unit
represents one metre. Room dimensions and scene unit scale affect travel time,
distance level, and air loss; endpoint display size does not.

Single-surface walls work from either side; thickness is not required for
reflection or occlusion. Keep endpoint origins off the surfaces themselves.
A panel with zero Transmission blocks a direct path that intersects it, while
indirect routes may still reach the receiver. Optional edge diffraction is
considered only when the straight path is blocked in every frequency band.

Use the assignment buttons for mesh endpoints so they are tagged and excluded
from both reflection and diffraction geometry. Selecting an untagged mesh only
through a Source/Receiver dropdown excludes it from the reflection mesh, but
the current diffraction extractor can still see its edges. Empties avoid this
distinction entirely.

## IR content modes

- **Full IR** includes direct sound, deterministic specular reflections,
  diffraction when needed and enabled, and the diffuse reflected field.
- **Wet Reflections** omits direct sound but retains deterministic early
  reflections, optional diffraction, and the diffuse field.
- **Diffuse Field Only** omits direct sound, deterministic early reflections,
  and diffraction. It contains only stochastic reflected energy.

Wet Reflections is useful when a dry signal will be mixed separately. Diffuse
Field Only is useful for reverb beds or for combining with a separate
early-reflection design.

"Diffuse" here means the stochastic reflected component, which can include
glossy/specular energy as well as material-scattered energy. It is not a
guarantee of a perfectly diffuse field. With **Deterministic Early Reflections**
disabled, the stochastic stage handles specular coverage. The Content selector
above Render and **IR Content** under **IR and Output** edit the same setting.

## Duration and rendering

**IR Duration** sets a fixed output window of **0.1 to 20 seconds**. Length is
set manually; the renderer does not automatically trim silence or apply an end
fade. Duration is an output limit, not a target reverberation time.
Initial propagation silence is retained, so the window must include travel time
to the receiver as well as the wanted decay. Energy still sounding at the end
is cut off. Choose a longer window if needed and apply an artistic fade in your
audio editor if desired.

Filter delays are compensated, so echo times follow acoustic travel times.
Band-limited reconstruction can ring around an arrival, including before its
nominal time. At 48 kHz, fractional-delay support extends up to 32 samples
(0.67 ms) either side of the nearest sample; the diffuse band filters extend
about 43 ms either side to resolve the lowest crossover. This filter support
does not represent extra acoustic paths. Samples outside the output window are
cropped without wraparound or boundary gain compensation.

The decay within that window follows material losses, air absorption, and
escape through openings, subject to the numerical limits set by **Maximum
Bounces**, **Minimum Path Energy**, and **Russian Roulette**. Increasing duration
alone cannot restore paths already stopped by those limits.

Interactive rendering snapshots the scene and runs acoustic processing in a
worker thread, with a progress indicator and a last-render event summary.
Changes made after the snapshot affect the next render, not the one underway.
Scene preparation and WAV/JSON export still run on Blender's main thread and
can briefly pause interaction. There is no dedicated render-cancel control;
keep Blender open until completion. Background/scripted execution is synchronous.

Event counts describe rendered arrivals, not polygon counts or reflection order.
The reported **early** count includes deterministic specular and optional
diffracted arrivals; **diffuse** counts stochastic arrivals. Zero early events
does not mean zero reflected sound: check content mode, material scattering,
path visibility, duration, and the early-path budget. The JSON sidecar records
the highest deterministic order evaluated and any orders skipped by that budget.

## Recommended starting settings

The defaults are intended as a useful first listening render:

| Setting | Default | Guidance |
| --- | ---: | --- |
| Quality | Balanced | 1,024 paths per side, 32 bounces, join depth 4 |
| Sample rate | 48 kHz | Good general production rate |
| Duration | 2.0 s | Increase for large or highly reflective spaces |
| Content | Full IR | Change to Wet or Diffuse for send effects |
| Source radiation | Even in Every Direction | Neutral and independent of source rotation |
| Early reflections | On, order 2 | Resolves first- and second-order specular paths cleanly |
| Pressure-Coherent Early Paths | On | Preserves angle-dependent level and polarity on distinct echoes |
| Bidirectional path search | On | Joins source and listener routes for better indirect coverage |
| Early Path Budget | 1,000,000 | Per-order search limit for second- and third-order echoes |
| Early Gain / Diffuse Gain | 0 dB / 0 dB | Neutral balance between distinct echoes and the stochastic wash |
| Air absorption | On | 20 C, 50% RH, 101.325 kPa |
| Random seed | 1 | Repeatable comparison between scene edits |
| Edge diffraction | Off | Enable only when shadowing needs it |
| Receiver orientation | On | Local +Y is front, -X is left, +Z is up; yaw 0 degrees, Z flip off |
| WAV format | 32-bit float | Allows peaks above 0 dBFS without integer clipping |
| Output level | Preserve Relative Level | Keeps the renderer's 1/r distance reference |

Use **Preview** (512 paths per side, 24 bounces, join depth 2) while moving
geometry or auditioning materials. Use **High** (4,096 paths per side, 64 bounces,
join depth 6) for a smoother final tail. **Ultra High** uses 16,384 paths from
each endpoint, 128 bounces, join depth 8, later path roulette, and a lower energy
cutoff for long or complex spaces. It only changes transport quality; sample
rate, duration, and IR content remain under explicit user control. If the tail
sounds grainy, raise Paths per Side before raising Maximum Bounces. If energy
stops too soon, increase IR Duration and Maximum Bounces together.

Quality presets change path count, bounce limit, join depth, roulette start and
survival probability, and minimum path energy. They do not turn bidirectional
search or roulette on/off, change deterministic order/budget, or alter materials.
With bidirectional search disabled, Paths per Side launches receiver paths only.

**Specular Roughness** (default 8 degrees) controls the spread of the glossy
part of stochastic reflections. It does not replace a material's per-band
**Unmodeled Scattering**, which sets the diffuse/specular energy split, and it
does not broaden deterministic image-source echoes.

**Russian Roulette** is enabled by default. In Balanced mode it starts at
bounce 20, keeps each subsequent path with probability 0.97, and compensates
surviving energy. Disabling it can reduce sampling variation at greater cost;
it is not an intended decay-length or loudness control. Maximum Bounces and
Minimum Path Energy (Balanced: `1e-6`) still apply.

**Bidirectional Path Search** traces the same number of paths from the source
and listener, then joins compatible surface points from the two searches.
**Subpath Join Depth** limits how many early vertices are retained at each end.
Increasing it can uncover longer routes through doorways, corridors, and
coupled spaces, but the number of attempted joins grows approximately with the
square of the depth. Multiple-importance weighting is applied separately at
every total reflection order. Joined routes obey Maximum Bounces as a total
path-order limit. The combined energy at each order and frequency band is
normalized to the average source and listener endpoint estimate; joined paths
contribute timing and directional information.

**Specular Order** controls the deterministic image-source depth independently
of stochastic render quality. Order 2 is the default. Order 3 includes paths
with three consecutive specular reflections. Candidate count grows rapidly
with the number of distinct reflector planes. Coplanar triangles on one object
are grouped before enumeration. If second or third order exceeds **Early Path
Budget**, that order and higher ones are skipped. The render reports the highest
completed order and records it in the JSON sidecar. First order is not
budget-limited. The UI allows budgets from 1,000 to 20,000,000 candidate sequences
per order. Raising the budget is useful when the geometry is acoustically
simple and additional waiting time is acceptable. Specular orders not searched
deterministically remain in the stochastic stage. For background on the
perceptual effects of image-source order, see [Johnson and Lee
(2016)](https://eprints.hud.ac.uk/id/eprint/28645/).

**Pressure-Coherent Early Paths** gives deterministic specular echoes an
angle-dependent signed pressure response. Direct sound is always coherent,
regardless of this option. Because the material library contains absorption
rather than measured complex impedance, the renderer infers a passive, purely
resistive locally reacting boundary in each band. Reflection strength and
polarity vary with incidence angle, especially near grazing. The response is
inferred, not based on measured material phase or impedance. With the option
disabled, deterministic reflection magnitudes come from reflected energy;
source radiation still supplies pressure polarity, and direct sound remains
coherent.

**Early Gain** adjusts deterministic specular and diffracted arrivals;
**Diffuse Gain** adjusts all stochastic arrivals. Neither changes the direct
arrival. Both default to 0 dB.

**Preserve Relative Level** is the default so Full IR exports retain distance
and material level differences. Float WAV is recommended because close sources
can exceed 0 dBFS in this mode. **Normalize for Audition** preserves
interchannel and time relationships within one IR but raises its peak to the
selected level, thereby removing its absolute distance reference. It is most
appropriate for quickly auditioning Wet or Diffuse IRs.

Levels are relative to a unit point source, not calibrated sound-pressure levels
in dB SPL. There is no source loudness/nonlinearity model: a louder convolution
input scales the result without changing the room's decay behavior.

## Acoustic materials

Each mesh object and opt-in Blender material has seven energy-domain
coefficient bands at 125, 250, 500, 1,000, 2,000, 4,000, and 8,000 Hz:

- **Absorption** removes incident energy.
- **Unmodeled Scattering** divides reflected energy between diffuse and
  specular behavior to represent surface detail absent from the acoustic mesh.
- **Transmission** allows energy to continue through a surface.

For every band, the stochastic reflected-energy fraction is
`clamp(1 - absorption - transmission, 0, 1)`; scattering only changes the
distribution of that reflection. The coherent early-path model uses this as
its normal-incidence reference and varies pressure with incidence angle.
Presets describe explicit constructions such as smooth painted concrete,
plaster on lath, carpet with underlay, or rough cave rock. They are
practical starting colors based on published room-acoustic material data, not
laboratory specifications for a particular commercial product. The source data
and construction-specific variation can be explored in the [ODEON material
library](https://odeon.dk/download/materials/Material.Li8).

The current library contains 22 named presets, including separate smooth,
rough-cave, and porous/weathered rock; smooth and rough concrete; solid and
cavity-backed wood; plaster, carpet, brick, glass, metal, tile/stone, gravel,
sand/soil, folded curtains, mineral wool, calm water, and upholstered audience
seating. Unassigned meshes use the **Custom** object fallback: absorption 0.2
in every band, gentle frequency-dependent scattering, and zero transmission.

Named presets are refreshed from the material library whenever a file loads.
Materials set to Custom retain their saved coefficients unchanged.

Evaluated mesh relief already changes reflection directions and must not be
counted again as material scattering. Use higher scattering when a simplified
surface stands in for missing joints, fractures, folds, seating, or other
detail; reduce it when those features are present in the mesh. Modifiers that
produce real evaluated geometry are included. Shader bump and normal maps are
visual only and can instead be represented by unmodeled scattering. Published
scattering guidance also varies strongly with the physical depth of unresolved
surface structure; see the [ODEON scattering
guidance](https://odeon.dk/pdf/ODEONManual12.pdf).

The broadband controls set all seven bands to one value. **Manual Band
Details** remain authoritative: editing an individual band preserves the
frequency-dependent curve and changes the preset to Custom. Transmission is
zero in the presets because transmission belongs to an entire wall assembly,
including thickness and backing, rather than only its visible finish.

The Acoustic Material panel follows the object's active Blender material slot.
Enable **Use Material Acoustics** to store coefficients on that material and use
them for every evaluated polygon assigned to it. Faces with no material, an
invalid slot, or a material whose acoustic option is disabled use the mesh
object's acoustic settings as a fallback.

Material assignment is read from the evaluated mesh, after modifiers. A
modifier can therefore preserve or generate distinct acoustic material regions
through its resulting material indices. A shared Blender material also shares
one acoustic setup across every object that uses it. **Copy Settings** copies
from the active material to the active materials of selected mesh objects, or
between object fallbacks when material acoustics is not enabled.

Thin transmissive surfaces are an artistic abstraction. A wall modeled with two
faces can apply transmission twice, and the diffraction model handles one edge
rather than a sequence of edges.

## Source radiation

Each source object can project sound with **Even in Every Direction**,
**Forward Focus**, **Front and Back**, **Focused Beam**,
**Loudspeaker-like**, or an advanced **Custom 3D Pattern**. The selected shape
affects the direct arrival, deterministic early reflections, stochastic
reflections, and diffraction from the direction in which the sound first
leaves the source. Turning a directional source changes its contribution to
both distinct echoes and the reverberant field.

Source-local `+Y` is the forward axis, `-X` is left, and `+Z` is up. The
evaluated world rotation is used, including parent rotation. Scale does not
alter the pattern. **Tone by Frequency** blends each band between an even spread
at 0 and the selected shape at 1. The Loudspeaker-like starting curve therefore
spreads bass broadly while aiming mids and treble more strongly.

Radiation shapes are normalized so their strongest direction retains unity
pressure. Choosing a directional shape does not automatically make its forward
sound louder than the even default; it reduces off-axis energy. Front and Back
and Custom 3D patterns can carry signed pressure, so opposite lobes may reverse
polarity. Focused Beam's width is specified at 6 dB below its on-axis pressure.
Custom 3D Pattern uses 16 signed third-order ACN/SN3D coefficients and is
peak-normalized after the shape is assembled.

## Ambisonic format and orientation

Output WAV files always contain 16 interleaved channels in ACN order with SN3D
normalization (AmbiX): `W, Y, Z, X, V, T, R, S, U, Q, O, M, K, L, N, P`.
Available sample rates are 44.1, 48, 96, and 192 kHz; WAV formats are 32-bit
float and 24-bit PCM. Higher sample rates do not add acoustic material bands
or increase the ambisonic order.

**Use Receiver Orientation** is enabled by default. Every arrival is transformed
into the receiver object's evaluated local rotation, including parent rotation,
before ambisonic encoding. Receiver-local Front (`+Y`) maps to AmbiX front
(`+X`), local `-X` maps to AmbiX left (`+Y`), and local `+Z` maps to up. Scale
and translation do not affect this orientation.

Disable the option to retain a Blender-world-aligned sound field using those
same axis mappings. **Ambisonic Yaw** applies an additional rotation around the
ambisonic up axis after the receiver transform; **Flip Ambisonic Z** supports
workflows with the opposite vertical convention.

A JSON sidecar is written next to each WAV with the format, channels, source,
receiver, render settings, normalization gain, and event counts. It also records
endpoint axes and rotations, source radiation, acoustic assignments, transport
statistics, and deterministic search coverage. For `ambisonic_ir.wav` the
sidecar is named `ambisonic_ir.wav.json`.

## Limitations

- Geometric acoustics does not reproduce low-frequency wave modes or modal
  pressure variation.
- Explicit specular image sources are limited to third order. Their cost grows
  combinatorially with distinct reflector planes, so highly tessellated curved
  geometry may reach the user-configurable early-path budget.
- Near-identical early paths from the same object sequence are consolidated
  within 0.25 ms and 12 degrees, retaining the strongest per-band response.
  This consolidation does not model physical caustics or finite-surface wave
  behavior.
- Early-reflection phase is inferred from absorption with a resistive boundary
  model; material-specific complex impedance and resonant phase are not
  represented.
- Bidirectional subpaths are randomly paired one-to-one and use uniform
  order-wise MIS, normalized against the average endpoint energy at each order
  and band.
- Diffraction is a bounded, single-edge approximation and is disabled by
  default.
- Sources and receivers are ideal points. Source radiation does not model
  the near field, cabinet edge diffraction, or the physical size of a radiator;
  receiver directivity is not modeled.
- Acoustic assignment follows evaluated polygon material indices; modifier
  configurations that discard or remap those indices necessarily change the
  resulting acoustic assignment.
- Stochastic tails converge progressively; identical nonzero seeds are
  repeatable, while seed zero intentionally creates a new realization.

Use listening tests, decay-envelope inspection, and comparison with simple
reference rooms to evaluate rendered results.

## Development and tests

The authoritative add-on is the `ir_raytracer/` package. Run all tests inside
the target Blender runtime:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --background --factory-startup --python-exit-code 1 `
  --python tests\run_blender_tests.py
```

Run this from the repository root. The runner imports the authoritative source,
exercises Blender's actual add-on loader (including restricted registration
access), and then discovers the regression tests. It does not depend on the
separately installed add-on being up to date.

The test suite checks ACN/SN3D encoding, air attenuation, power-complementary
band synthesis, fractional-delay magnitude and phase, boundary cropping,
cross-channel coherence, coherent pressure transfer, reciprocal and joined path
weighting, multi-order finite image-source paths, direct-path calibration, source
directivity and rotation, content separation, diffraction, evaluated Blender
geometry and material assignments, safe worker-thread snapshots, tooltips,
repeatability, and WAV/JSON export. These are regression checks, not a listening
study or acoustic-engineering validation.

## License

This project is licensed under the [MIT License](LICENSE.txt).
