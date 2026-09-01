# Ambisonic IR Tracer for Blender

Ambisonic IR Tracer turns Blender geometry into 16-channel, third-order
ambisonic impulse responses for convolution reverb, sound design, and
electroacoustic art.

This is a creative, physics-inspired renderer. It aims for plausible spatial
character and useful artistic control; it is not validated acoustic-engineering
software and should not be used for building or safety decisions.

> This project is also an experiment in AI-assisted software development. The
> code and documentation should be treated critically and verified against your
> own listening tests. Claims of accuracy require benchmarking that this project
> has not performed.

## Current architecture

Version 2 uses one receiver-centric acoustic energy renderer. It no longer
crossfades unrelated forward and reverse simulations.

- Direct sound is evaluated deterministically, including per-band transmission.
- Planar specular reflections are found deterministically with finite image
  sources through a selectable first, second, or third order.
- Remaining reflected energy is sampled from the receiver with diffuse,
  glossy, absorptive, and transmissive surface interactions. All-specular paths
  already covered by the image-source stage are omitted from this estimator.
- Monte Carlo energy events are converted to pressure with repeatable randomized
  phase and a complementary seven-band filter bank.
- Every arrival is encoded directly to third-order ACN/SN3D (AmbiX) using its
  listener-relative arrival direction.
- Optional bounded single-edge diffraction supplies a creative shadow-zone
  approximation.

Separating acoustic energy transport from pressure synthesis avoids treating
unrelated late paths as phase-coherent and gives direct, early, and diffuse
components a single geometry and level convention.

## Requirements and installation

- Blender 5.2.1 LTS
- `numpy` (included with Blender)
- `soundfile` in Blender's Python environment for multichannel WAV export

Install the `ir_raytracer` directory as a Blender extension/add-on, enable
**Ambisonic IR Tracer**, and open **3D Viewport > Sidebar > IR Tracer**. The
Diagnostics panel can confirm that `soundfile` is available.

## Basic workflow

1. Model a closed or partly open space at a meaningful Blender unit scale.
2. Select an object at the emitter position and choose **Use Active as Source**.
3. Select an object at the listening position and choose **Use Active as
   Receiver**. Empty objects work well for both endpoints.
4. Select each acoustic mesh and choose an Acoustic Material preset. Expand
   **Manual Band Details** when frequency-specific shaping is wanted.
5. Choose IR content, duration, sample rate, quality, and an output path.
6. Run **Validate Acoustic Scene**, then **Render Ambisonic IR**.

The renderer uses evaluated world-space geometry and endpoint transforms, so
parenting, modifiers, and object transforms are respected. Source and receiver
objects themselves are excluded from acoustic geometry.

## IR content modes

- **Full IR** includes direct sound, deterministic specular reflections,
  diffraction when needed and enabled, and the diffuse reflected field.
- **Wet Reflections** omits direct sound but retains deterministic early
  reflections, optional diffraction, and the diffuse field.
- **Diffuse Field Only** omits direct sound, deterministic early reflections,
  and diffraction. It contains only stochastic reflected energy.

Wet Reflections is useful when a dry signal will be mixed separately. Diffuse
Field Only is deliberately less literal and is useful for spacious reverb beds
or for combining with another early-reflection design.

## Recommended starting settings

The defaults are intended as a useful first listening render:

| Setting | Default | Guidance |
| --- | ---: | --- |
| Quality | Balanced | 1,024 receiver rays and 32 bounces |
| Sample rate | 48 kHz | Good general production rate |
| Duration | 2.0 s | Increase for large or highly reflective spaces |
| Content | Full IR | Change to Wet or Diffuse for send effects |
| Early reflections | On, order 2 | Resolves first- and second-order specular paths cleanly |
| Air absorption | On | 20 C, 50% RH, 101.325 kPa |
| Random seed | 1 | Repeatable comparison between scene edits |
| Edge diffraction | Off | Enable only when shadowing needs it |
| Output | 32-bit float | Preserves the renderer's 1/r distance level |

Use **Preview** while moving geometry or auditioning materials. Use **High** for
a smoother final tail. **Ultra High** uses 16,384 listener rays, 128 bounces,
later path roulette, and a lower energy cutoff for long or complex spaces. It
only changes transport quality; sample rate, duration, and IR content remain
under explicit user control. If the tail sounds grainy, raise Listener Rays
before raising Maximum Bounces. If energy stops too soon, increase IR Duration
and Maximum Bounces together.

**Specular Order** controls the deterministic image-source depth independently
of stochastic render quality. Order 2 is the default balance. Order 3 improves
discrete echoes and localization in corridors, coupled rooms, and other strongly
specular spaces, but its candidate count grows rapidly with the number of
distinct reflector planes. Coplanar triangles on one object are grouped before
enumeration. If an order exceeds **Early Path Budget**, that order and higher
ones are omitted rather than partially sampling a directionally biased subset;
the render reports the highest completed order and records it in the JSON
sidecar. Raising the budget is useful when the geometry is already acoustically
simple and additional waiting time is acceptable. This hybrid division follows
the perceptually motivated structure explored by [Johnson and Lee
(2016)](https://eprints.hud.ac.uk/id/eprint/28645/).

**Preserve Relative Level** is the default so Full IR exports retain distance
and material level differences. Float WAV is recommended because close sources
can exceed 0 dBFS in this mode. **Normalize for Audition** preserves
interchannel and time relationships within one IR but raises its peak to the
selected level, thereby removing its absolute distance reference. It is most
appropriate for quickly auditioning Wet or Diffuse IRs.

## Acoustic materials

Each mesh object and opt-in Blender material has seven energy-domain
coefficient bands at 125, 250, 500, 1,000, 2,000, 4,000, and 8,000 Hz:

- **Absorption** removes incident energy.
- **Unmodeled Scattering** divides reflected energy between diffuse and
  specular behavior to represent surface detail absent from the acoustic mesh.
- **Transmission** allows energy to continue through a surface.

For every band, reflected energy is clamped to
`1 - absorption - transmission`; scattering only changes the distribution of
that reflection. Presets describe explicit constructions such as smooth painted
concrete, plaster on lath, carpet with underlay, or rough cave rock. They are
practical starting colors based on published room-acoustic material data, not
laboratory specifications for a particular commercial product. The source data
and construction-specific variation can be explored in the [ODEON material
library](https://odeon.dk/download/materials/Material.Li8).

Named presets are refreshed from the current library whenever a file loads, so
calibration improvements also reach existing named assignments. Materials set
to Custom retain their saved coefficients unchanged.

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
object's acoustic settings as a fallback. This opt-in behavior preserves the
sound of existing scenes that already use Blender materials for appearance.

Material assignment is read from the evaluated mesh, after modifiers. A
modifier can therefore preserve or generate distinct acoustic material regions
through its resulting material indices. A shared Blender material also shares
one acoustic setup across every object that uses it. **Copy Settings** copies
from the active material to the active materials of selected mesh objects, or
between object fallbacks when material acoustics is not enabled.

Thin transmissive surfaces are an artistic abstraction. A wall modeled with two
faces can apply transmission twice, and the diffraction model handles one edge
rather than a sequence of edges.

## Ambisonic format and orientation

Output WAV files always contain 16 planar channels in ACN order with SN3D
normalization (AmbiX): `W, Y, Z, X, V, T, R, S, U, Q, O, M, K, L, N, P`.

**Use Receiver Orientation** is enabled by default. Every arrival is transformed
into the receiver object's evaluated local rotation, including parent rotation,
before ambisonic encoding. Receiver-local Front (`-Y`) maps to AmbiX front
(`+X`), local `+X` maps to AmbiX left (`+Y`), and local `+Z` maps to up. Scale
and translation do not affect this orientation.

Disable the option to retain a Blender-world-aligned sound field using those
same axis mappings. **Ambisonic Yaw** applies an additional rotation around the
ambisonic up axis after the receiver transform; **Flip Ambisonic Z** supports
workflows with the opposite vertical convention.

A JSON sidecar is written next to each WAV with the format, channels, source,
receiver, render settings, normalization gain, and event counts.

## Limitations

- Geometric acoustics does not reproduce low-frequency wave modes or modal
  pressure variation.
- Explicit specular image sources are limited to third order. Their cost grows
  combinatorially with distinct reflector planes, so highly tessellated curved
  geometry may reach the user-configurable early-path budget.
- Diffraction is a bounded, single-edge approximation and is disabled by
  default.
- Point source and point receiver directivity are currently omnidirectional.
- Acoustic assignment follows evaluated polygon material indices; modifier
  configurations that discard or remap those indices necessarily change the
  resulting acoustic assignment.
- Stochastic tails converge progressively; identical nonzero seeds are
  repeatable, while seed zero intentionally creates a new realization.

These constraints are intentional for the current artistic scope. Listening,
decay-envelope inspection, and comparison with simple reference rooms remain
important parts of using the tool.

## Development and tests

The authoritative add-on is the `ir_raytracer/` package. Run all tests inside
the target Blender runtime:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --background --factory-startup --python tests\run_blender_tests.py
```

The test suite checks ACN/SN3D encoding, air attenuation, complementary-band
synthesis, energy sampling, multi-order finite image-source paths, direct-path
calibration, content separation, diffraction, evaluated Blender geometry,
repeatability, and output levels.
