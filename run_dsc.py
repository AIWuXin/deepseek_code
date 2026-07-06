"""Entry wrapper for Nuitka: imports and runs dsc.cli.main().
Compile this instead of dsc/cli.py so relative imports in the package work correctly.
"""
import sys
import os

# Ensure the package root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsc.cli import main
raise SystemExit(main())
