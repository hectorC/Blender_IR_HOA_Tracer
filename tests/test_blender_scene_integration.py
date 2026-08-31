"""Small end-to-end Blender scene smoke test for the modular add-on."""
from __future__ import annotations

import os
import sys
import unittest
from math import pi

import bpy
import numpy as np
from mathutils import Vector


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ir_raytracer  # noqa: E402
from ir_raytracer.core.ray_tracer import trace_impulse_response  # noqa: E402
from ir_raytracer.utils.scene_utils import build_bvh  # noqa: E402


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

    def test_blocked_source_receiver_scene_renders_diffraction(self):
        # A zero-thickness screen is a true single-edge diffraction case. A
        # solid box would require a separate multiple-edge model.
        bpy.ops.mesh.primitive_plane_add(
            size=2.0,
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, pi / 2.0, 0.0),
        )

        source_object = bpy.data.objects.new("AIRT Test Source", None)
        source_object.location = (-3.0, 0.0, 0.0)
        source_object.is_acoustic_source = True
        bpy.context.scene.collection.objects.link(source_object)

        receiver_object = bpy.data.objects.new("AIRT Test Receiver", None)
        receiver_object.location = (3.0, 0.0, 0.0)
        receiver_object.is_acoustic_receiver = True
        bpy.context.scene.collection.objects.link(receiver_object)

        scene = bpy.context.scene
        scene.airt_trace_mode = 'FORWARD'
        scene.airt_output_content = 'REVERB_ONLY'
        scene.airt_num_rays = 6
        scene.airt_max_order = 1
        scene.airt_ir_seconds = 0.1
        scene.airt_quick_broadband = True
        scene.airt_enable_seg_capture = False
        scene.airt_enable_diffraction = True
        scene.airt_diffraction_samples = 4
        scene.airt_diffraction_max_deg = 60.0

        bvh, object_map = build_bvh(bpy.context)
        self.assertIsNotNone(bvh)
        ir = trace_impulse_response(
            bpy.context,
            Vector(source_object.location),
            Vector(receiver_object.location),
            bvh,
            object_map,
            directions=[
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ],
        )

        self.assertEqual(ir.shape[0], 16)
        self.assertGreater(float(np.max(np.abs(ir))), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
