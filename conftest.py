"""Root conftest.py: ensure the testing/ directory is on sys.path."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
