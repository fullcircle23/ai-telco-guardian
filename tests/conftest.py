# tests/conftest.py
import os
import sys

# Add the project source root to sys.path (one level up from tests/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
