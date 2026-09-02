"""End-to-end Blender scene tests for the unified acoustic renderer."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from collections import Counter
from math import pi, sqrt

import bpy
import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ir_raytracer  # noqa: E402
from ir_raytracer.core.acoustics import (  # noqa: E402
    DEFAULT_SCATTER_SPECTRUM,
    MATERIAL_PRESETS,
    MaterialProperties,
)
from ir_raytracer.core.ray_tracer import AmbisonicIREngine  # noqa: E402
from ir_raytracer.utils.scene_utils import (  # noqa: E402
    build_acoustic_scene,
    object_world_position,
    object_world_rotation,
)


class BlenderSceneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ir_raytracer.register()

    @classmethod
    def tearDownClass(cls):
        ir_raytracer.unregister()

    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material)
        scene = bpy.context.scene
        scene.airt_source_object = None
        scene.airt_receiver_object = None
        scene.airt_quality_preset = 'CUSTOM'
        scene.airt_num_rays = 128
        scene.airt_max_order = 4
        scene.airt_sr = '48000'
        scene.airt_ir_seconds = 0.2
        scene.airt_seed = 23
        scene.airt_air_enable = False
        scene.airt_enable_diffraction = False
        scene.airt_early_reflections = True
        scene.airt_early_order = 2
        scene.airt_use_receiver_orientation = True

    def _endpoint(self, name, location, source=False):
        obj = bpy.data.objects.new(name, None)
        obj.location = location
        obj.is_acoustic_source = source
        obj.is_acoustic_receiver = not source
        bpy.context.scene.collection.objects.link(obj)
        if source:
            bpy.context.scene.airt_source_object = obj
        else:
            bpy.context.scene.airt_receiver_object = obj
        return obj

    def _room(self):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
        room = bpy.context.object
        room.name = "Test Room"
        room.scale = (5.0, 4.0, 3.0)
        room.airt_material_preset = 'PLASTER'
        return room

    def _specular_wall(self, name, x, absorption=0.0):
        bpy.ops.mesh.primitive_plane_add(
            size=100.0,
            location=(x, 0.0, 0.0),
            rotation=(0.0, pi / 2.0, 0.0),
        )
        wall = bpy.context.object
        wall.name = name
        wall.airt_material_preset = 'CUSTOM'
        wall.absorption = absorption
        wall.absorption_bands = (absorption,) * 7
        wall.scatter = 0.0
        wall.scatter_bands = (0.0,) * 7
        wall.transmission = 0.0
        wall.transmission_bands = (0.0,) * 7
        return wall

    def _acoustic_material(self, name, absorption, enabled=True):
        material = bpy.data.materials.new(name)
        material.airt_material_preset = 'CUSTOM'
        material.absorption = absorption
        material.absorption_bands = (absorption,) * 7
        material.scatter = 0.0
        material.scatter_bands = (0.0,) * 7
        material.transmission = 0.0
        material.transmission_bands = (0.0,) * 7
        material.airt_acoustic_enabled = enabled
        return material

    def _render(self, content):
        bpy.context.scene.airt_output_content = content
        source = bpy.context.scene.airt_source_object
        receiver = bpy.context.scene.airt_receiver_object
        return AmbisonicIREngine(bpy.context).render(
            object_world_position(bpy.context, source),
            object_world_position(bpy.context, receiver),
        )

    def test_content_modes_separate_direct_early_and_diffuse_events(self):
        self._room()
        self._endpoint("Source", (-1.0, 0.0, 0.0), source=True)
        self._endpoint("Receiver", (1.0, 0.5, 0.0))

        full = self._render('FULL')
        wet = self._render('REFLECTIONS')
        diffuse = self._render('DIFFUSE')

        self.assertEqual(full.ir.shape, (16, 9600))
        self.assertEqual(full.transport.direct_events, 1)
        self.assertGreater(full.transport.early_events, 0)
        self.assertGreater(full.synthesis.diffuse_events, 0)
        self.assertEqual(wet.transport.direct_events, 0)
        self.assertGreater(wet.transport.early_events, 0)
        self.assertGreater(wet.synthesis.diffuse_events, 0)
        self.assertEqual(diffuse.transport.direct_events, 0)
        self.assertEqual(diffuse.transport.early_events, 0)
        self.assertEqual(diffuse.synthesis.early_events, 0)
        self.assertGreater(diffuse.synthesis.diffuse_events, 0)

    def test_nonzero_seed_is_repeatable(self):
        self._room()
        self._endpoint("Source", (-1.0, 0.0, 0.0), source=True)
        self._endpoint("Receiver", (1.0, 0.5, 0.0))

        first = self._render('FULL')
        second = self._render('FULL')
        np.testing.assert_array_equal(first.ir, second.ir)

    def test_blocked_source_receiver_scene_renders_diffraction(self):
        bpy.ops.mesh.primitive_plane_add(
            size=2.0,
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, pi / 2.0, 0.0),
        )
        self._endpoint("Source", (-3.0, 0.0, 0.0), source=True)
        self._endpoint("Receiver", (3.0, 0.0, 0.0))

        scene = bpy.context.scene
        scene.airt_max_order = 1
        scene.airt_ir_seconds = 0.1
        scene.airt_early_reflections = False
        scene.airt_enable_diffraction = True
        scene.airt_diffraction_samples = 4
        scene.airt_diffraction_max_deg = 60.0

        result = self._render('REFLECTIONS')
        self.assertGreater(result.transport.diffraction_events, 0)
        self.assertGreater(float(np.max(np.abs(result.ir))), 0.0)

    def test_evaluated_parent_transform_drives_endpoint_position(self):
        parent = bpy.data.objects.new("Parent", None)
        parent.location = (3.0, -2.0, 1.0)
        bpy.context.scene.collection.objects.link(parent)
        source = self._endpoint("Source", (1.0, 2.0, 3.0), source=True)
        source.parent = parent
        bpy.context.view_layer.update()

        np.testing.assert_allclose(
            object_world_position(bpy.context, source),
            Vector((4.0, 0.0, 4.0)),
            atol=1e-7,
        )

    def test_receiver_parent_rotation_orients_ambisonic_field(self):
        parent = bpy.data.objects.new("Receiver Parent", None)
        parent.rotation_euler = (0.0, 0.0, pi / 2.0)
        parent.scale = (2.0, 3.0, 4.0)
        bpy.context.scene.collection.objects.link(parent)
        self._endpoint("Source", (1.0, 0.0, 0.0), source=True)
        receiver = self._endpoint("Receiver", (0.0, 0.0, 0.0))
        receiver.parent = parent
        bpy.context.view_layer.update()

        rotation = object_world_rotation(bpy.context, receiver)
        np.testing.assert_allclose(
            rotation @ Vector((0.0, -1.0, 0.0)),
            Vector((1.0, 0.0, 0.0)),
            atol=1e-6,
        )

        scene = bpy.context.scene
        scene.airt_ir_seconds = 0.1
        scene.airt_use_receiver_orientation = True
        receiver_aligned = self._render('FULL')
        scene.airt_use_receiver_orientation = False
        world_aligned = self._render('FULL')

        # World +X is receiver-local front after the parent's +90 degree yaw.
        self.assertAlmostEqual(float(np.sum(receiver_aligned.ir[3])), 1.0, places=5)
        self.assertAlmostEqual(float(np.sum(receiver_aligned.ir[1])), 0.0, places=5)
        # With the option disabled, Blender world +X remains AmbiX left.
        self.assertAlmostEqual(float(np.sum(world_aligned.ir[1])), 1.0, places=5)
        self.assertAlmostEqual(float(np.sum(world_aligned.ir[3])), 0.0, places=5)

    def test_scene_builder_uses_evaluated_world_geometry(self):
        room = self._room()
        room.location = (7.0, 0.0, 0.0)
        bpy.context.view_layer.update()

        acoustic_scene = build_acoustic_scene(bpy.context)
        vertices = np.array([
            tuple(vertex)
            for face in acoustic_scene.faces
            for vertex in face.vertices
        ])
        self.assertGreater(float(np.min(vertices[:, 0])), 1.9)
        self.assertGreater(float(np.max(vertices[:, 0])), 11.9)

    def test_evaluated_faces_resolve_material_or_object_fallback(self):
        enabled = self._acoustic_material("Enabled Acoustic", 0.8)
        disabled = self._acoustic_material("Visual Only", 0.9, enabled=False)
        mesh = bpy.data.meshes.new("Mixed Material Mesh")
        mesh.from_pydata(
            [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                (3.0, 1.0, 0.0), (2.0, 1.0, 0.0),
            ],
            [],
            [(0, 1, 2, 3), (4, 5, 6, 7)],
        )
        mesh.materials.append(enabled)
        mesh.materials.append(disabled)
        mesh.polygons[0].material_index = 0
        mesh.polygons[1].material_index = 1
        obj = bpy.data.objects.new("Mixed Acoustic Surface", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.airt_material_preset = 'CUSTOM'
        obj.absorption = 0.1
        obj.absorption_bands = (0.1,) * 7

        faces = sorted(
            (
                face for face in build_acoustic_scene(bpy.context).faces
                if face.object_ref == obj
            ),
            key=lambda face: sum(vertex.x for vertex in face.vertices),
        )

        self.assertEqual(len(faces), 2)
        self.assertIs(faces[0].material_ref, enabled)
        self.assertIs(faces[0].acoustic_ref, enabled)
        np.testing.assert_allclose(
            MaterialProperties(faces[0].acoustic_ref).absorption_spectrum,
            0.8,
        )
        self.assertIs(faces[1].material_ref, disabled)
        self.assertIs(faces[1].acoustic_ref, obj)
        np.testing.assert_allclose(
            MaterialProperties(faces[1].acoustic_ref).absorption_spectrum,
            0.1,
        )

    def test_new_blender_materials_are_acoustically_disabled_by_default(self):
        material = bpy.data.materials.new("Visual Material")

        self.assertFalse(material.airt_acoustic_enabled)
        self.assertEqual(material.airt_material_preset, 'CUSTOM')
        self.assertAlmostEqual(material.absorption, 0.2)
        self.assertAlmostEqual(
            material.scatter,
            sum(DEFAULT_SCATTER_SPECTRUM) / len(DEFAULT_SCATTER_SPECTRUM),
        )
        np.testing.assert_allclose(
            tuple(material.scatter_bands),
            DEFAULT_SCATTER_SPECTRUM,
        )
        self.assertAlmostEqual(material.transmission, 0.0)

    def test_broadband_and_manual_band_edits_stay_synchronized(self):
        material = bpy.data.materials.new("Editable Acoustic Material")
        material.airt_material_preset = 'ROCK_CAVE'
        cave = MATERIAL_PRESETS['ROCK_CAVE']

        self.assertTrue(material.airt_acoustic_enabled)
        np.testing.assert_allclose(
            tuple(material.scatter_bands),
            cave['scatter_spectrum'],
        )
        self.assertAlmostEqual(material.scatter, cave['scatter'])

        material.scatter = 0.25
        self.assertEqual(material.airt_material_preset, 'CUSTOM')
        np.testing.assert_allclose(tuple(material.scatter_bands), 0.25)

        material.airt_material_preset = 'ROCK_CAVE'
        manual_bands = list(material.scatter_bands)
        manual_bands[0] = 0.33
        material.scatter_bands = manual_bands
        self.assertEqual(material.airt_material_preset, 'CUSTOM')
        np.testing.assert_allclose(tuple(material.scatter_bands), manual_bands)
        self.assertAlmostEqual(material.scatter, sum(manual_bands) / 7.0)

    def test_modifier_generated_faces_preserve_material_assignments(self):
        primary = self._acoustic_material("Primary Acoustic", 0.2)
        generated = self._acoustic_material("Generated Acoustic", 0.7)
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        obj = bpy.context.object
        obj.data.materials.append(primary)
        obj.data.materials.append(generated)
        modifier = obj.modifiers.new("Acoustic Thickness", 'SOLIDIFY')
        modifier.thickness = 0.2
        modifier.material_offset = 1
        modifier.material_offset_rim = 1
        bpy.context.view_layer.update()

        faces = [
            face for face in build_acoustic_scene(bpy.context).faces
            if face.object_ref == obj
        ]
        assignments = {face.acoustic_ref for face in faces}

        self.assertGreater(len(faces), 1)
        self.assertIn(primary, assignments)
        self.assertIn(generated, assignments)

    def test_deterministic_reflection_uses_material_at_finite_hit(self):
        absorbing = self._acoustic_material("Absorbing Half", 0.8)
        reflective = self._acoustic_material("Reflective Half", 0.0)
        mesh = bpy.data.meshes.new("Split Reflector")
        mesh.from_pydata(
            [
                (-2.0, -2.0, 0.0), (0.0, -2.0, 0.0),
                (0.0, 2.0, 0.0), (-2.0, 2.0, 0.0),
                (0.0, -2.0, 0.0), (2.0, -2.0, 0.0),
                (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
            ],
            [],
            [(0, 1, 2, 3), (4, 5, 6, 7)],
        )
        mesh.materials.append(absorbing)
        mesh.materials.append(reflective)
        mesh.polygons[0].material_index = 0
        mesh.polygons[1].material_index = 1
        obj = bpy.data.objects.new("Split Reflector", mesh)
        bpy.context.scene.collection.objects.link(obj)
        scene = bpy.context.scene
        scene.airt_output_content = 'REFLECTIONS'
        scene.airt_early_order = 1

        absorbing_events = AmbisonicIREngine(bpy.context)._first_order_specular_events(
            Vector((-1.0, -0.5, 1.0)),
            Vector((-1.0, 0.5, 1.0)),
        )
        reflective_events = AmbisonicIREngine(bpy.context)._first_order_specular_events(
            Vector((1.0, -0.5, 1.0)),
            Vector((1.0, 0.5, 1.0)),
        )

        self.assertEqual(len(absorbing_events), 1)
        self.assertEqual(len(reflective_events), 1)
        self.assertAlmostEqual(
            float(
                absorbing_events[0].energy_bands[0]
                / reflective_events[0].energy_bands[0]
            ),
            0.2,
            places=5,
        )

    def test_copy_settings_targets_active_blender_materials(self):
        source_material = self._acoustic_material("Acoustic Source", 0.73)
        target_material = self._acoustic_material(
            "Acoustic Target", 0.1, enabled=False
        )
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-1.0, 0.0, 0.0))
        source = bpy.context.object
        source.data.materials.append(source_material)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(1.0, 0.0, 0.0))
        target = bpy.context.object
        target.data.materials.append(target_material)
        bpy.ops.object.select_all(action='DESELECT')
        source.select_set(True)
        target.select_set(True)
        bpy.context.view_layer.objects.active = source

        status = bpy.ops.airt.copy_material()

        self.assertEqual(status, {'FINISHED'})
        self.assertTrue(target_material.airt_acoustic_enabled)
        np.testing.assert_allclose(
            tuple(target_material.absorption_bands),
            tuple(source_material.absorption_bands),
        )

    def test_copy_settings_preserves_named_preset_state(self):
        source_material = bpy.data.materials.new("Named Acoustic Source")
        source_material.airt_material_preset = 'ROCK_CAVE'
        target_material = bpy.data.materials.new("Named Acoustic Target")
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-1.0, 0.0, 0.0))
        source = bpy.context.object
        source.data.materials.append(source_material)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(1.0, 0.0, 0.0))
        target = bpy.context.object
        target.data.materials.append(target_material)
        bpy.ops.object.select_all(action='DESELECT')
        source.select_set(True)
        target.select_set(True)
        bpy.context.view_layer.objects.active = source

        status = bpy.ops.airt.copy_material()

        self.assertEqual(status, {'FINISHED'})
        self.assertEqual(target_material.airt_material_preset, 'ROCK_CAVE')
        np.testing.assert_allclose(
            tuple(target_material.scatter_bands),
            MATERIAL_PRESETS['ROCK_CAVE']['scatter_spectrum'],
        )

    def test_registered_defaults_are_a_balanced_listening_start(self):
        scene = bpy.data.scenes.new("Defaults Test")
        try:
            self.assertEqual(scene.airt_quality_preset, 'BALANCED')
            self.assertEqual(scene.airt_num_rays, 1024)
            self.assertEqual(scene.airt_max_order, 32)
            self.assertEqual(scene.airt_sr, '48000')
            self.assertAlmostEqual(scene.airt_ir_seconds, 2.0)
            self.assertEqual(scene.airt_output_content, 'FULL')
            self.assertTrue(scene.airt_early_reflections)
            self.assertEqual(scene.airt_early_order, 2)
            self.assertEqual(scene.airt_early_path_budget, 1_000_000)
            self.assertTrue(scene.airt_use_receiver_orientation)
            self.assertTrue(scene.airt_air_enable)
            self.assertFalse(scene.airt_enable_diffraction)
            self.assertEqual(scene.airt_seed, 1)
            self.assertEqual(scene.airt_wav_subtype, 'FLOAT')
            self.assertEqual(scene.airt_normalization, 'PRESERVE')
            self.assertAlmostEqual(scene.airt_peak_db, -1.0)
        finally:
            bpy.data.scenes.remove(scene)

    def test_every_artist_facing_control_has_tooltip_text(self):
        owner_properties = {
            'absorption',
            'absorption_bands',
            'scatter',
            'scatter_bands',
            'transmission',
            'transmission_bands',
            'show_frequency_details',
            'is_acoustic_source',
            'is_acoustic_receiver',
        }
        discovered = []
        for owner_type in (
            bpy.types.Object,
            bpy.types.Material,
            bpy.types.Scene,
        ):
            for prop in owner_type.bl_rna.properties:
                if (
                    prop.identifier.startswith('airt_')
                    or prop.identifier in owner_properties
                ):
                    discovered.append(
                        (owner_type.__name__, prop.identifier)
                    )
                    self.assertTrue(
                        prop.description.strip(),
                        f"{owner_type.__name__}.{prop.identifier} has no tooltip",
                    )
                    if prop.type == 'ENUM':
                        for item in prop.enum_items:
                            self.assertTrue(
                                item.description.strip(),
                                f"{owner_type.__name__}.{prop.identifier} "
                                f"item {item.identifier} has no tooltip",
                            )
        self.assertTrue(discovered)

        for operator_type in ir_raytracer.classes:
            if not issubclass(operator_type, bpy.types.Operator):
                continue
            self.assertTrue(
                operator_type.bl_description.strip(),
                f"{operator_type.__name__} has no tooltip",
            )

    def test_ultra_high_profile_increases_transport_quality_only(self):
        scene = bpy.data.scenes.new("Ultra High Test")
        try:
            original_sample_rate = scene.airt_sr
            original_duration = scene.airt_ir_seconds
            original_content = scene.airt_output_content
            original_early_order = scene.airt_early_order
            scene.airt_quality_preset = 'ULTRA'

            self.assertEqual(scene.airt_num_rays, 16384)
            self.assertEqual(scene.airt_max_order, 128)
            self.assertEqual(scene.airt_rr_start, 48)
            self.assertAlmostEqual(scene.airt_rr_p, 0.99)
            self.assertAlmostEqual(scene.airt_min_throughput, 1e-8)
            self.assertEqual(scene.airt_sr, original_sample_rate)
            self.assertEqual(scene.airt_ir_seconds, original_duration)
            self.assertEqual(scene.airt_output_content, original_content)
            self.assertEqual(scene.airt_early_order, original_early_order)

            scene.airt_num_rays = 20000
            self.assertEqual(scene.airt_quality_preset, 'CUSTOM')
        finally:
            bpy.data.scenes.remove(scene)

    def test_faceted_cylinder_does_not_multiply_axial_early_reflection(self):
        scene = bpy.context.scene
        scene.airt_ir_seconds = 1.0
        scene.airt_output_content = 'FULL'
        scene.airt_early_reflections = True
        self._endpoint("Source", (0.0, 0.0, 0.0), source=True)
        self._endpoint("Receiver", (0.0, 0.0, 151.528))
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=94,
            radius=6.0,
            depth=179.0,
            end_fill_type='NGON',
            location=(0.0, 0.0, 87.48),
        )
        cylinder = bpy.context.object
        cylinder.airt_material_preset = 'BRICK'

        engine = AmbisonicIREngine(bpy.context)
        source = object_world_position(bpy.context, scene.airt_source_object)
        receiver = object_world_position(bpy.context, scene.airt_receiver_object)
        events = engine._first_order_specular_events(source, receiver)

        # Ninety-four side polygons describe one temporally and spatially
        # unresolved cylindrical reflection. The two end caps remain separate.
        self.assertEqual(len(events), 3)
        side_events = [
            event for event in events
            if 0.441 < event.delay_seconds < 0.443
        ]
        self.assertEqual(len(side_events), 1)
        self.assertLess(side_events[0].arrival_direction.z, -0.99)

    def test_parallel_walls_resolve_second_and_third_order_image_paths(self):
        scene = bpy.context.scene
        scene.airt_ir_seconds = 1.0
        scene.airt_output_content = 'REFLECTIONS'
        scene.airt_early_order = 3
        self._endpoint("Source", (2.0, -1.0, 0.0), source=True)
        self._endpoint("Receiver", (7.0, 1.0, 0.0))
        self._specular_wall("Wall A", 0.0)
        self._specular_wall("Wall B", 10.0)

        engine = AmbisonicIREngine(bpy.context)
        events = engine._deterministic_specular_events(
            object_world_position(bpy.context, scene.airt_source_object),
            object_world_position(bpy.context, scene.airt_receiver_object),
        )

        self.assertEqual(Counter(event.order for event in events), {1: 2, 2: 2, 3: 2})
        expected_distances = {
            1: sorted((sqrt(85.0), sqrt(125.0))),
            2: sorted((sqrt(229.0), sqrt(629.0))),
            3: sorted((sqrt(845.0), sqrt(965.0))),
        }
        for order, distances in expected_distances.items():
            order_events = sorted(
                (event for event in events if event.order == order),
                key=lambda event: event.delay_seconds,
            )
            np.testing.assert_allclose(
                [
                    event.delay_seconds * engine.config.speed_of_sound_bu
                    for event in order_events
                ],
                distances,
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                [sqrt(float(event.energy_bands[0])) for event in order_events],
                [1.0 / distance for distance in distances],
                rtol=1e-6,
                atol=1e-6,
            )
        self.assertEqual(engine.tracer.stats.early_highest_order, 3)
        self.assertEqual(engine.tracer.stats.early_sequences_tested, 6)
        self.assertEqual(engine.tracer.stats.early_orders_skipped, 0)

    def test_opaque_divider_blocks_multi_order_image_paths(self):
        scene = bpy.context.scene
        scene.airt_ir_seconds = 1.0
        scene.airt_output_content = 'REFLECTIONS'
        scene.airt_early_order = 3
        self._endpoint("Source", (2.0, -1.0, 0.0), source=True)
        self._endpoint("Receiver", (7.0, 1.0, 0.0))
        self._specular_wall("Wall A", 0.0)
        self._specular_wall("Wall B", 10.0)
        self._specular_wall("Absorbing Divider", 5.0, absorption=1.0)

        engine = AmbisonicIREngine(bpy.context)
        events = engine._deterministic_specular_events(
            object_world_position(bpy.context, scene.airt_source_object),
            object_world_position(bpy.context, scene.airt_receiver_object),
        )
        self.assertEqual(events, [])

    def test_render_operator_writes_16_channel_wav_and_metadata(self):
        import soundfile

        self._room()
        self._endpoint("Source", (-1.0, 0.0, 0.0), source=True)
        self._endpoint("Receiver", (1.0, 0.5, 0.0))
        scene = bpy.context.scene
        scene.airt_output_content = 'FULL'
        scene.airt_ir_seconds = 0.1

        with tempfile.TemporaryDirectory(prefix="airt_export_") as directory:
            output_path = os.path.join(directory, "test_ir.wav")
            scene.airt_output_path = output_path
            status = bpy.ops.airt.render_ir()

            self.assertEqual(status, {'FINISHED'})
            audio, sample_rate = soundfile.read(output_path, always_2d=True)
            self.assertEqual(audio.shape, (4800, 16))
            self.assertEqual(sample_rate, 48000)
            with open(output_path + ".json", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata["channel_convention"], "ACN/SN3D (AmbiX)")
            self.assertEqual(len(metadata["channels"]), 16)
            self.assertEqual(metadata["deterministic_reflection_order"], 2)
            self.assertGreater(
                metadata["deterministic_path_stats"]["surface_sequences_tested"],
                0,
            )
            self.assertGreater(
                metadata["deterministic_path_stats"]["events_by_order"]["2"],
                0,
            )
            self.assertEqual(metadata["orientation"]["reference"], "receiver_local")
            self.assertEqual(
                len(metadata["orientation"]["receiver_world_quaternion_wxyz"]),
                4,
            )
            self.assertEqual(len(metadata["acoustic_assignments"]), 1)
            assignment = metadata["acoustic_assignments"][0]
            self.assertEqual(assignment["type"], "OBJECT_FALLBACK")
            self.assertEqual(assignment["name"], "Test Room")
            self.assertGreater(assignment["evaluated_face_count"], 0)

    def test_acoustic_snapshot_renders_safely_in_worker_thread(self):
        self._room()
        source_object = self._endpoint(
            "Source", (-1.0, 0.0, 0.0), source=True
        )
        receiver_object = self._endpoint("Receiver", (1.0, 0.5, 0.0))
        acoustic_scene = build_acoustic_scene(bpy.context)
        self.assertTrue(acoustic_scene.faces)
        self.assertTrue(all(
            face.material_snapshot is not None
            for face in acoustic_scene.faces
        ))

        engine = AmbisonicIREngine(bpy.context, acoustic_scene=acoustic_scene)
        engine.prepare_for_background()
        source = object_world_position(bpy.context, source_object)
        receiver = object_world_position(bpy.context, receiver_object)
        outcome = {}

        def render():
            try:
                outcome["result"] = engine.render(source, receiver)
            except BaseException as error:
                outcome["error"] = error

        worker = threading.Thread(target=render)
        worker.start()
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["result"].ir.shape[0], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
