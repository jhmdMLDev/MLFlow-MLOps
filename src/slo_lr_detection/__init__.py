"""Init file
"""

import os
import sys

import toml

current_file_dir = os.path.abspath(os.path.dirname(__file__))
toml_path = os.path.join(current_file_dir, "..", "..", "pyproject.toml")

with open(toml_path, encoding="utf-8") as f:
    project_data = toml.load(f)

__version__ = project_data["tool"]["poetry"]["version"]
