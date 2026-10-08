"""Runs the public-demo checks in a clean process (demo mode is fixed at import time)."""

import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class PublicDemoTests(unittest.TestCase):
    def test_demo_boundaries(self):
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "demo_checks.py")],
            cwd=os.path.dirname(HERE),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
