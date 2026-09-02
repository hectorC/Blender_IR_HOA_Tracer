"""Artist-facing Blender properties for the unified acoustic renderer."""
from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from ..core.acoustics import (
    DEFAULT_ABSORPTION_SPECTRUM,
    DEFAULT_SCATTER_SPECTRUM,
    MATERIAL_PRESET_DATA,
    MATERIAL_PRESETS,
    NUM_BANDS,
)
from ..core.directivity import (
    DEFAULT_CUSTOM_SH,
    DIRECTIVITY_STRENGTH_PRESETS,
)


_MATERIAL_GUARD = set()
_QUALITY_GUARD = set()


QUALITY_PROFILES = {
    'PREVIEW': {
        'airt_num_rays': 512,
        'airt_max_order': 24,
        'airt_rr_start': 12,
        'airt_rr_p': 0.95,
        'airt_min_throughput': 1e-5,
    },
    'BALANCED': {
        'airt_num_rays': 1024,
        'airt_max_order': 32,
        'airt_rr_start': 20,
        'airt_rr_p': 0.97,
        'airt_min_throughput': 1e-6,
    },
    'HIGH': {
        'airt_num_rays': 4096,
        'airt_max_order': 64,
        'airt_rr_start': 32,
        'airt_rr_p': 0.98,
        'airt_min_throughput': 1e-6,
    },
    'ULTRA': {
        'airt_num_rays': 16384,
        'airt_max_order': 128,
        'airt_rr_start': 48,
        'airt_rr_p': 0.99,
        'airt_min_throughput': 1e-8,
    },
}


def _material_items():
    items = [(
        'CUSTOM',
        'Custom',
        "Shape the surface by ear: absorption controls how quickly its "
        "reflections fade, while scattering controls how focused or washed "
        "out they feel",
    )]
    for (
        identifier, label, description, _absorption, _scatter
    ) in MATERIAL_PRESET_DATA:
        preset = MATERIAL_PRESETS[identifier]
        average_absorption = float(preset['absorption'])
        average_scattering = float(preset['scatter'])
        if average_absorption < 0.08:
            decay_character = "very lively, lingering reflections"
        elif average_absorption < 0.20:
            decay_character = "fairly lively reflections"
        elif average_absorption < 0.45:
            decay_character = "a moderately damped response"
        else:
            decay_character = "a dry, quickly damped response"
        if average_scattering < 0.10:
            reflection_character = "clear and focused echoes"
        elif average_scattering < 0.30:
            reflection_character = "gently softened echoes"
        elif average_scattering < 0.55:
            reflection_character = "broad, blended reflections"
        else:
            reflection_character = "a dense, diffuse wash"
        items.append((
            identifier,
            label,
            (
                f"{description}. A starting point for {decay_character} "
                f"with {reflection_character}"
            ),
        ))
    return items


def _apply_material_preset(owner):
    """Apply the canonical coefficients for one named preset owner."""
    self = owner
    preset = MATERIAL_PRESETS.get(getattr(self, 'airt_material_preset', 'CUSTOM'))
    if preset is None:
        return False
    key = id(self)
    _MATERIAL_GUARD.add(key)
    try:
        if hasattr(self, 'airt_acoustic_enabled'):
            self.airt_acoustic_enabled = True
        self.absorption = float(preset['absorption'])
        self.scatter = float(preset['scatter'])
        self.transmission = 0.0
        self.absorption_bands = preset['absorption_spectrum']
        self.scatter_bands = preset['scatter_spectrum']
        self.transmission_bands = tuple(0.0 for _ in range(NUM_BANDS))
    finally:
        _MATERIAL_GUARD.discard(key)
    return True


def _update_material_preset(self, _context):
    _apply_material_preset(self)


@persistent
def _refresh_named_material_presets(_unused):
    """Refresh named presets after a file load; leave Custom owners intact."""
    objects = getattr(bpy.data, 'objects', None)
    materials = getattr(bpy.data, 'materials', None)
    if objects is None or materials is None:
        return False
    for owner in tuple(objects) + tuple(materials):
        if hasattr(owner, 'airt_material_preset'):
            _apply_material_preset(owner)
    return True


def _deferred_named_material_refresh():
    """Wait until Blender releases restricted registration data access."""
    return None if _refresh_named_material_presets(None) else 0.1


def _enable_material_acoustics(self):
    if hasattr(self, 'airt_acoustic_enabled'):
        self.airt_acoustic_enabled = True


def _update_broadband_value(self, scalar_attr, vector_attr):
    """Apply a broadband edit uniformly to the authoritative band values."""
    _enable_material_acoustics(self)
    key = id(self)
    if key in _MATERIAL_GUARD:
        return
    _MATERIAL_GUARD.add(key)
    try:
        value = float(getattr(self, scalar_attr))
        setattr(self, vector_attr, tuple(value for _ in range(NUM_BANDS)))
        if self.airt_material_preset != 'CUSTOM':
            self.airt_material_preset = 'CUSTOM'
    finally:
        _MATERIAL_GUARD.discard(key)


def _update_band_values(self, scalar_attr, vector_attr):
    """Keep the broadband display in sync without flattening manual bands."""
    _enable_material_acoustics(self)
    key = id(self)
    if key in _MATERIAL_GUARD:
        return
    _MATERIAL_GUARD.add(key)
    try:
        values = tuple(float(value) for value in getattr(self, vector_attr))
        setattr(self, scalar_attr, sum(values) / max(len(values), 1))
        if self.airt_material_preset != 'CUSTOM':
            self.airt_material_preset = 'CUSTOM'
    finally:
        _MATERIAL_GUARD.discard(key)


def _update_absorption(self, _context):
    _update_broadband_value(self, 'absorption', 'absorption_bands')


def _update_absorption_bands(self, _context):
    _update_band_values(self, 'absorption', 'absorption_bands')


def _update_scatter(self, _context):
    _update_broadband_value(self, 'scatter', 'scatter_bands')


def _update_scatter_bands(self, _context):
    _update_band_values(self, 'scatter', 'scatter_bands')


def _update_transmission(self, _context):
    _update_broadband_value(self, 'transmission', 'transmission_bands')


def _update_transmission_bands(self, _context):
    _update_band_values(self, 'transmission', 'transmission_bands')


def _register_acoustic_owner_props(owner):
    """Register the coefficient schema on an Object or Material datablock."""
    owner.airt_material_preset = bpy.props.EnumProperty(
        name="Acoustic Material",
        description=(
            "Choose a familiar surface as a starting point for the tone, "
            "liveliness, and spread of its reflections"
        ),
        items=_material_items(),
        default='CUSTOM',
        update=_update_material_preset,
    )
    owner.absorption = bpy.props.FloatProperty(
        name="Absorption",
        description=(
            "How strongly this surface quiets each reflection across the "
            "whole spectrum; higher values make the space drier and shorten "
            "its decay"
        ),
        default=sum(DEFAULT_ABSORPTION_SPECTRUM) / NUM_BANDS,
        min=0.0,
        max=1.0,
        update=_update_absorption,
    )
    owner.absorption_bands = bpy.props.FloatVectorProperty(
        name="Absorption Bands",
        description=(
            "Shape which parts of the sound fade fastest, from bass at 125 "
            "Hz to brightness at 8 kHz; higher values damp that tonal region"
        ),
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=DEFAULT_ABSORPTION_SPECTRUM,
        update=_update_absorption_bands,
    )
    owner.scatter = bpy.props.FloatProperty(
        name="Unmodeled Scattering",
        description=(
            "How strongly texture missing from the mesh spreads reflections "
            "in many directions; higher values soften distinct echoes and "
            "create a denser reverberant wash"
        ),
        default=sum(DEFAULT_SCATTER_SPECTRUM) / NUM_BANDS,
        min=0.0,
        max=1.0,
        update=_update_scatter,
    )
    owner.scatter_bands = bpy.props.FloatVectorProperty(
        name="Unmodeled Scattering Bands",
        description=(
            "Choose whether unmodeled texture spreads the bass, midrange, or "
            "treble most; use less when that surface relief already exists "
            "in the evaluated mesh"
        ),
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=DEFAULT_SCATTER_SPECTRUM,
        update=_update_scatter_bands,
    )
    owner.transmission = bpy.props.FloatProperty(
        name="Transmission",
        description=(
            "How much sound passes through this surface instead of returning "
            "to the space; higher values make the boundary feel less solid"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        update=_update_transmission,
    )
    owner.transmission_bands = bpy.props.FloatVectorProperty(
        name="Transmission Bands",
        description=(
            "Shape which frequencies leak through the surface, from bass at "
            "125 Hz to treble at 8 kHz; higher values weaken reflections in "
            "that tonal region"
        ),
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=tuple(0.0 for _ in range(NUM_BANDS)),
        update=_update_transmission_bands,
    )
    owner.show_frequency_details = bpy.props.BoolProperty(
        name="Manual Band Details",
        description=(
            "Open the tonal controls for shaping how bass, midrange, and "
            "treble are damped, spread, or passed through the surface"
        ),
        default=False,
    )


def _update_quality_profile(self, _context):
    profile = QUALITY_PROFILES.get(getattr(self, 'airt_quality_preset', 'CUSTOM'))
    if profile is None:
        return
    key = id(self)
    _QUALITY_GUARD.add(key)
    try:
        for property_name, value in profile.items():
            setattr(self, property_name, value)
    finally:
        _QUALITY_GUARD.discard(key)


def _mark_quality_custom(self, _context):
    if id(self) not in _QUALITY_GUARD and self.airt_quality_preset != 'CUSTOM':
        self.airt_quality_preset = 'CUSTOM'


def _update_source_directivity(self, _context):
    """Give each radiation shape a useful tonal starting curve."""
    pattern = getattr(self, 'airt_source_directivity', 'OMNI')
    strengths = DIRECTIVITY_STRENGTH_PRESETS.get(pattern)
    if strengths is not None:
        self.airt_source_directivity_bands = strengths


def register_acoustic_props():
    obj = bpy.types.Object
    material = bpy.types.Material
    scene = bpy.types.Scene

    _register_acoustic_owner_props(obj)
    _register_acoustic_owner_props(material)
    material.airt_acoustic_enabled = bpy.props.BoolProperty(
        name="Use Material Acoustics",
        description=(
            "Make faces using this Blender material share its acoustic "
            "character; when disabled they use the object's fallback sound"
        ),
        default=False,
    )
    obj.is_acoustic_source = bpy.props.BoolProperty(
        name="Acoustic Source",
        description=(
            "Place the point where the imagined sound begins at this "
            "object's evaluated position"
        ),
        default=False,
    )
    obj.is_acoustic_receiver = bpy.props.BoolProperty(
        name="Acoustic Receiver",
        description=(
            "Place the virtual listener here; its rotation can also define "
            "the front, sides, and height of the ambisonic sound field"
        ),
        default=False,
    )
    obj.airt_source_directivity = bpy.props.EnumProperty(
        name="Radiation Shape",
        description=(
            "Choose how the source projects sound around itself; rotate the "
            "source object to aim any directional shape"
        ),
        items=[
            (
                'OMNI', 'Even in Every Direction',
                'Radiate equally around the source, ideal for a neutral starting point or an uncharacterized sound',
            ),
            (
                'CARDIOID', 'Forward Focus',
                'Keep the front strong while gently quieting the sides and rejecting the rear, like many microphones and small radiators',
            ),
            (
                'DIPOLE', 'Front and Back',
                'Project equally forward and backward with quiet sides; the rear has opposite polarity',
            ),
            (
                'FORWARD_CONE', 'Focused Beam',
                'Aim sound into an adjustable forward beam for horns, megaphones, or strongly projected artistic sources',
            ),
            (
                'LOUDSPEAKER', 'Loudspeaker-like',
                'Let bass spread broadly while mids and highs become progressively more forward-facing',
            ),
            (
                'CUSTOM_SH', 'Custom 3D Pattern',
                'Build an advanced third-order 3D radiation shape from signed AmbiX coefficients',
            ),
        ],
        default='OMNI',
        update=_update_source_directivity,
    )
    obj.airt_source_directivity_bands = bpy.props.FloatVectorProperty(
        name="Directional Focus by Frequency",
        description=(
            "Blend each tonal region between an even spread at 0 and the "
            "chosen radiation shape at 1, from bass through treble"
        ),
        size=NUM_BANDS,
        min=0.0,
        max=1.0,
        default=DIRECTIVITY_STRENGTH_PRESETS['OMNI'],
    )
    obj.airt_source_cone_width_deg = bpy.props.FloatProperty(
        name="Beam Width",
        description=(
            "Width in degrees of the focused beam at the point where it has "
            "fallen by 6 dB; smaller values create a tighter projection"
        ),
        default=90.0,
        min=10.0,
        max=170.0,
    )
    obj.airt_source_directivity_sh = bpy.props.FloatVectorProperty(
        name="Custom Pattern Coefficients",
        description=(
            "Signed third-order ACN/SN3D AmbiX weights used to sculpt the "
            "custom 3D source pattern; the strongest direction is kept at "
            "the same peak level"
        ),
        size=16,
        min=-4.0,
        max=4.0,
        default=DEFAULT_CUSTOM_SH,
    )
    obj.show_source_directivity_details = bpy.props.BoolProperty(
        name="Tone by Frequency",
        description=(
            "Open seven controls for making the source more or less "
            "directional in its bass, midrange, and treble"
        ),
        default=False,
    )
    scene.airt_source_object = bpy.props.PointerProperty(
        name="Source",
        description=(
            "Choose the object marking where the sound originates; distance "
            "from the listener affects arrival time and level"
        ),
        type=bpy.types.Object,
    )
    scene.airt_receiver_object = bpy.props.PointerProperty(
        name="Receiver",
        description=(
            "Choose the object marking the listening point and, when enabled, "
            "the facing direction of the ambisonic recording"
        ),
        type=bpy.types.Object,
    )
    scene.airt_quality_preset = bpy.props.EnumProperty(
        name="Render Quality",
        description=(
            "Balance waiting time against the smoothness, density, and "
            "stability of the reverberant tail"
        ),
        items=[
            (
                'PREVIEW', 'Preview',
                'A quick draft for judging scale and decay; the quiet tail may sound grainy',
            ),
            (
                'BALANCED', 'Balanced',
                'Recommended starting point with a convincing tail and short wait',
            ),
            (
                'HIGH', 'High',
                'A smoother, more stable reverberant texture for final listening',
            ),
            (
                'ULTRA', 'Ultra High',
                'The densest, most stable tail for very large or long-decaying spaces; significantly slower',
            ),
            (
                'CUSTOM', 'Custom',
                'Use the advanced controls to choose your own speed and texture balance',
            ),
        ],
        default='BALANCED',
        update=_update_quality_profile,
    )
    scene.airt_num_rays = bpy.props.IntProperty(
        name="Listener Rays",
        description=(
            "How many directions are listened into from the receiver; more "
            "rays make quiet reflections and the tail smoother, but take longer"
        ),
        default=1024,
        min=128,
        max=131072,
        update=_mark_quality_custom,
    )
    scene.airt_max_order = bpy.props.IntProperty(
        name="Maximum Bounces",
        description=(
            "How many surfaces a traced sound path may visit; higher values "
            "can preserve deep, lingering tails in reflective spaces, but "
            "increase render time"
        ),
        default=32,
        min=1,
        max=512,
        update=_mark_quality_custom,
    )
    scene.airt_sr = bpy.props.EnumProperty(
        name="Sample Rate",
        description=(
            "Choose the time resolution and playback rate of the exported IR; "
            "use the same rate as the session where it will be heard"
        ),
        items=[
            ('44100', '44.1 kHz', 'Common music-production rate with compact files'),
            ('48000', '48 kHz', 'Recommended general-purpose rate for audio and video work'),
            ('96000', '96 kHz', 'Finer timing and high-frequency headroom at roughly twice the data size'),
            ('192000', '192 kHz', 'Specialist maximum-resolution output with much larger files and processing cost'),
        ],
        default='48000',
    )
    scene.airt_ir_seconds = bpy.props.FloatProperty(
        name="IR Duration",
        description=(
            "Length of the exported IR in seconds; choose enough time for the "
            "space to fall quiet, because any tail still sounding at the end "
            "is cut off"
        ),
        default=2.0,
        min=0.1,
        max=20.0,
        precision=2,
    )
    scene.airt_output_content = bpy.props.EnumProperty(
        name="IR Content",
        description=(
            "Choose whether the file includes the original source arrival, "
            "recognizable echoes, or only the blended reverberant wash"
        ),
        items=[
            (
                'FULL', 'Full IR',
                'The complete virtual recording: direct arrival, distinct early echoes, and reverberant tail',
            ),
            (
                'REFLECTIONS', 'Wet Reflections',
                'Echoes and reverberant tail without the direct arrival; ideal when keeping the dry sound separately',
            ),
            (
                'DIFFUSE', 'Diffuse Field Only',
                'Only the blended stochastic reverberant field, without the direct arrival or explicit planar echoes',
            ),
        ],
        default='FULL',
    )
    scene.airt_early_reflections = bpy.props.BoolProperty(
        name="Deterministic Early Reflections",
        description=(
            "Preserve clear, precisely placed echoes from broad flat surfaces; "
            "especially useful for walls, corridors, chambers, and coupled rooms"
        ),
        default=True,
    )
    scene.airt_early_order = bpy.props.IntProperty(
        name="Specular Order",
        description=(
            "How many consecutive mirror-like reflections may form a distinct "
            "echo; higher values reveal slapback and exchanges between walls, "
            "but grow expensive on complex geometry"
        ),
        default=2,
        min=1,
        max=3,
    )
    scene.airt_early_path_budget = bpy.props.IntProperty(
        name="Early Path Budget",
        description=(
            "Safety limit for searching combinations of distinct reflective "
            "surfaces; raise only when important second- or third-bounce echoes "
            "are skipped, as cost can grow extremely quickly"
        ),
        default=1_000_000,
        min=1_000,
        max=20_000_000,
    )
    scene.airt_early_gain_db = bpy.props.FloatProperty(
        name="Early Gain",
        description=(
            "Turn distinct planar echoes up or down in dB; higher values make "
            "the room shape and source placement more obvious"
        ),
        default=0.0,
        min=-24.0,
        max=24.0,
    )
    scene.airt_diffuse_gain_db = bpy.props.FloatProperty(
        name="Diffuse Gain",
        description=(
            "Turn the blended reverberant wash up or down in dB; higher values "
            "make the space feel wetter and more enveloping"
        ),
        default=0.0,
        min=-24.0,
        max=24.0,
    )
    scene.airt_seed = bpy.props.IntProperty(
        name="Random Seed",
        description=(
            "Choose the fine-grained texture of the stochastic reflections; "
            "the same nonzero number repeats a render, while zero creates a "
            "fresh variation each time"
        ),
        default=1,
        min=0,
    )
    scene.airt_spec_rough_deg = bpy.props.FloatProperty(
        name="Specular Roughness",
        description=(
            "Width in degrees of glossy reflected directions; low values keep "
            "reflections focused and mirror-like, while high values soften and "
            "spread them"
        ),
        default=8.0,
        min=0.0,
        max=45.0,
    )
    scene.airt_rr_enable = bpy.props.BoolProperty(
        name="Russian Roulette",
        description=(
            "Keep very long reflected paths practical by sampling only some of "
            "the quietest late paths while compensating their level"
        ),
        default=True,
    )
    scene.airt_rr_start = bpy.props.IntProperty(
        name="Start Bounce",
        description=(
            "Number of reflections traced normally before late-path sampling "
            "begins; higher values resolve more of the deep tail directly but "
            "take longer"
        ),
        default=20,
        min=1,
        max=512,
        update=_mark_quality_custom,
    )
    scene.airt_rr_p = bpy.props.FloatProperty(
        name="Survival Probability",
        description=(
            "Chance that a very late path continues after each bounce; values "
            "closer to 1 make the quiet tail smoother and more stable, but slower"
        ),
        default=0.97,
        min=0.5,
        max=1.0,
        update=_mark_quality_custom,
    )
    scene.airt_min_throughput = bpy.props.FloatProperty(
        name="Minimum Path Energy",
        description=(
            "Stop following paths once they are too quiet to matter; lower "
            "values retain fainter tail detail, while higher values render faster"
        ),
        default=1e-6,
        min=1e-10,
        max=1e-2,
        update=_mark_quality_custom,
    )
    scene.airt_air_enable = bpy.props.BoolProperty(
        name="Air Absorption",
        description=(
            "Let long journeys through air gradually lose brightness, helping "
            "large spaces sound softer and more distant"
        ),
        default=True,
    )
    scene.airt_air_temp_c = bpy.props.FloatProperty(
        name="Temperature",
        description=(
            "Air temperature in degrees Celsius; warmer air makes arrivals "
            "slightly earlier and subtly changes tonal loss over distance"
        ),
        default=20.0,
        min=-30.0,
        max=50.0,
    )
    scene.airt_air_humidity = bpy.props.FloatProperty(
        name="Relative Humidity",
        description=(
            "Moisture in the air, used to shape distance-related high-frequency "
            "loss; most noticeable in very large spaces and long tails"
        ),
        default=50.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE',
    )
    scene.airt_air_pressure_kpa = bpy.props.FloatProperty(
        name="Pressure",
        description=(
            "Atmospheric pressure in kPa, which subtly changes how brightness "
            "fades across long distances; normal room work can keep the default"
        ),
        default=101.325,
        min=80.0,
        max=110.0,
    )
    scene.airt_enable_diffraction = bpy.props.BoolProperty(
        name="Edge Diffraction",
        description=(
            "Let muted sound bend around one sharp edge, adding believable "
            "arrivals when the source is hidden behind a wall or corner"
        ),
        default=False,
    )
    scene.airt_diffraction_samples = bpy.props.IntProperty(
        name="Maximum Edge Paths",
        description=(
            "Maximum alternative edge routes heard when direct sight is "
            "blocked; more routes can enrich an acoustic shadow but take longer"
        ),
        default=4,
        min=1,
        max=32,
    )
    scene.airt_diffraction_max_deg = bpy.props.FloatProperty(
        name="Maximum Bend Angle",
        description=(
            "Largest corner angle a diffracted path may turn through; higher "
            "values fill deeper shadows but make the effect less selective"
        ),
        default=45.0,
        min=1.0,
        max=90.0,
    )
    scene.airt_yaw_offset_deg = bpy.props.FloatProperty(
        name="Ambisonic Yaw",
        description=(
            "Rotate the finished horizontal sound field in degrees, allowing "
            "you to choose which Blender direction plays as front without "
            "moving the scene"
        ),
        default=0.0,
        min=-180.0,
        max=180.0,
    )
    scene.airt_use_receiver_orientation = bpy.props.BoolProperty(
        name="Use Receiver Orientation",
        description=(
            "Make the receiver's rotation define front, left, right, above, "
            "and below in the rendered sound field; disable to keep the field "
            "aligned with Blender's world axes"
        ),
        default=True,
    )
    scene.airt_invert_z = bpy.props.BoolProperty(
        name="Flip Ambisonic Z",
        description=(
            "Swap above and below in the exported ambisonic field for systems "
            "that expect the opposite vertical convention"
        ),
        default=False,
    )
    scene.airt_output_path = bpy.props.StringProperty(
        name="Output WAV",
        description=(
            "Where to save the 16-channel third-order AmbiX WAV; a matching "
            "JSON file records the scene and render settings"
        ),
        default="//ambisonic_ir.wav",
        subtype='FILE_PATH',
    )
    scene.airt_wav_subtype = bpy.props.EnumProperty(
        name="WAV Format",
        description=(
            "Choose how the IR amplitudes are stored; floating point keeps the "
            "most headroom for later sound design"
        ),
        items=[
            (
                'FLOAT', '32-bit Float',
                'Recommended: preserves quiet detail and peaks safely during editing and convolution',
            ),
            (
                'PCM_24', '24-bit PCM',
                'Widely compatible integer audio with slightly less headroom for processing',
            ),
        ],
        default='FLOAT',
    )
    scene.airt_normalization = bpy.props.EnumProperty(
        name="Output Level",
        description=(
            "Choose whether the WAV preserves perceived distance and material "
            "level relationships or is made immediately loud for auditioning"
        ),
        items=[
            (
                'PRESERVE', 'Preserve Relative Level',
                'Keep distance attenuation and level relationships so different placements remain meaningfully comparable',
            ),
            (
                'PEAK', 'Normalize for Audition',
                'Lift the loudest peak to a chosen level for convenient listening, removing absolute distance loudness',
            ),
        ],
        default='PRESERVE',
    )
    scene.airt_peak_db = bpy.props.FloatProperty(
        name="Normalized Peak",
        description=(
            "Loudest peak of a normalized file in dBFS; values below 0 leave "
            "headroom for convolution and later processing"
        ),
        default=-1.0,
        min=-24.0,
        max=0.0,
    )
    scene.airt_last_render_summary = bpy.props.StringProperty(
        name="Last Render",
        description=(
            "Summary of the most recent render's direct arrival, distinct "
            "planar echoes, and stochastic reflected events"
        ),
        default="",
    )
    if _refresh_named_material_presets not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_refresh_named_material_presets)
    if not bpy.app.timers.is_registered(_deferred_named_material_refresh):
        bpy.app.timers.register(
            _deferred_named_material_refresh,
            first_interval=0.0,
        )


def unregister_acoustic_props():
    if bpy.app.timers.is_registered(_deferred_named_material_refresh):
        bpy.app.timers.unregister(_deferred_named_material_refresh)
    if _refresh_named_material_presets in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_refresh_named_material_presets)
    acoustic_owner_names = (
        'airt_material_preset', 'absorption', 'absorption_bands', 'scatter',
        'scatter_bands', 'transmission', 'transmission_bands',
        'show_frequency_details',
    )
    object_names = acoustic_owner_names + (
        'is_acoustic_source', 'is_acoustic_receiver',
        'airt_source_directivity', 'airt_source_directivity_bands',
        'airt_source_cone_width_deg', 'airt_source_directivity_sh',
        'show_source_directivity_details',
    )
    material_names = acoustic_owner_names + ('airt_acoustic_enabled',)
    scene_names = (
        'airt_source_object', 'airt_receiver_object', 'airt_quality_preset',
        'airt_num_rays', 'airt_max_order', 'airt_sr', 'airt_ir_seconds',
        'airt_output_content', 'airt_early_reflections', 'airt_early_order',
        'airt_early_path_budget', 'airt_early_gain_db', 'airt_diffuse_gain_db',
        'airt_seed', 'airt_spec_rough_deg',
        'airt_rr_enable', 'airt_rr_start', 'airt_rr_p', 'airt_min_throughput',
        'airt_air_enable', 'airt_air_temp_c', 'airt_air_humidity',
        'airt_air_pressure_kpa', 'airt_enable_diffraction',
        'airt_diffraction_samples', 'airt_diffraction_max_deg',
        'airt_use_receiver_orientation', 'airt_yaw_offset_deg',
        'airt_invert_z', 'airt_output_path',
        'airt_wav_subtype', 'airt_normalization', 'airt_peak_db',
        'airt_last_render_summary',
    )
    for name in object_names:
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)
    for name in material_names:
        if hasattr(bpy.types.Material, name):
            delattr(bpy.types.Material, name)
    for name in scene_names:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
