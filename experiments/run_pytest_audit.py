import sys
import os
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

def main():
    test_path = os.path.join(project_root, "backend", "tests")
    
    print("=" * 80)
    print(" 1. COLLECT ONLY TEST COUNT: pytest backend/tests --collect-only -q")
    print("=" * 80)
    exit_collect = pytest.main([test_path, "--collect-only", "-q"])

    print("\n" + "=" * 80)
    print(" 2. FULL REGRESSION RUN: pytest backend/tests -q")
    print("=" * 80)
    exit_run = pytest.main([test_path, "-q"])

    return exit_run

if __name__ == "__main__":
    sys.exit(main())
