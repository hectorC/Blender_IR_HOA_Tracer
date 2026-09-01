"""Discover and run the add-on regression suite inside Blender's Python."""
from __future__ import annotations

import os
import sys
import unittest

import addon_utils


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Exercise Blender's real add-on registration context before importing tests.
# addon_utils restricts bpy.data while register() runs, unlike a direct module
# call, and therefore catches registration-only failures seen in the UI.
addon_module = addon_utils.enable(
    "ir_raytracer",
    default_set=False,
    persistent=False,
)
if addon_module is None:
    raise RuntimeError("Blender add-on loader could not enable ir_raytracer")
from ir_raytracer.ui.properties import (  # noqa: E402
    _deferred_named_material_refresh,
)
if _deferred_named_material_refresh() is not None:
    raise RuntimeError("Deferred material refresh still sees restricted data")
addon_utils.disable("ir_raytracer", default_set=False)


suite = unittest.defaultTestLoader.discover(TEST_DIR, pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
