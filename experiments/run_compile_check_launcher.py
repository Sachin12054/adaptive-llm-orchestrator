import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.compile_check import main if hasattr(py_compile, 'main') else None

import py_compile
file_path = os.path.join(project_root, "backend", "app", "services", "orchestration_pipeline.py")

try:
    py_compile.compile(file_path, doraise=True)
    print(f"PASS: {file_path} compiled successfully without syntax errors.")
except Exception as e:
    print(f"FAIL: Syntax error compiling {file_path}:\n{e}")
    sys.exit(1)
