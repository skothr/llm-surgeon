"""Root conftest.py: ensure the repo root is on sys.path."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
