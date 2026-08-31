"""Discover and run the add-on regression suite inside Blender's Python."""
from __future__ import annotations

import os
import sys
import unittest


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


suite = unittest.defaultTestLoader.discover(TEST_DIR, pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
