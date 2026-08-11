import sys
import os
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

def main():
    test_path = os.path.join(project_root, "backend", "tests")
    print("Running full backend regression test suite: pytest backend/tests -q\n")
    exit_code = pytest.main([test_path, "-q"])
    print(f"\nExit Code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
