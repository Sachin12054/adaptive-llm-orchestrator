import sys
import os
import io
import pytest

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "backend"))

    test_path = os.path.join(project_root, "backend", "tests")
    print(f"Executing pytest on '{test_path}'...\n")

    # Capture output
    exit_code = pytest.main([test_path, "-v"])
    print(f"\nPytest Exit Code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    main()
