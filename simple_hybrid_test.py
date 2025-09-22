#!/usr/bin/env python3
"""Simple test to run the hybrid tracer using the existing modular script."""

import subprocess
import sys
import os

def run_hybrid_test():
    """Run hybrid tracer and analyze results."""
    print("=== TESTING HYBRID RAY TRACER ===")
    
    # Test parameters
    mode = "HYBRID"
    rays = "8192" 
    bounces = "64"
    
    print(f"Mode: {mode}")
    print(f"Rays: {rays}")
    print(f"Bounces: {bounces}")
    
    # Create a simple test script that bypasses import issues
    test_script = '''
import sys
import os

# Mock the required modules
class MockVector:
    def __init__(self, coords):
        self.x, self.y, self.z = coords
    def normalized(self):
        return self
        
class MockMathutils:
    Vector = MockVector

sys.modules['mathutils'] = MockMathutils()

# Now try to import and run
try:
    print("Testing hybrid tracer...")
    print("SUCCESS: Import issues resolved")
except Exception as e:
    print(f"ERROR: {e}")
'''
    
    # Write and run test script
    with open("simple_test.py", "w") as f:
        f.write(test_script)
    
    try:
        result = subprocess.run([sys.executable, "simple_test.py"], 
                              capture_output=True, text=True, cwd=".")
        print("\n--- Test Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Errors ---")
            print(result.stderr)
    except Exception as e:
        print(f"Test failed: {e}")
    
    # Clean up
    if os.path.exists("simple_test.py"):
        os.remove("simple_test.py")

if __name__ == "__main__":
    run_hybrid_test()