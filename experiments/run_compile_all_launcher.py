import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.compile_all_backend import main if hasattr(py_compile, 'main') else None
import py_compile

backend_dir = os.path.join(project_root, "backend")

files_to_check = [
    os.path.join(backend_dir, "app", "services", "orchestration_pipeline.py"),
    os.path.join(backend_dir, "app", "api", "routes", "orchestration.py"),
    os.path.join(backend_dir, "app", "main.py"),
]

all_passed = True
for f in files_to_check:
    try:
        py_compile.compile(f, doraise=True)
        print(f"PASS: {f} compiled successfully.")
    except Exception as e:
        print(f"FAIL: {f} syntax error:\n{e}")
        all_passed = False

if not all_passed:
    sys.exit(1)
