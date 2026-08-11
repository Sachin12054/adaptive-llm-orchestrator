import sys
import os
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"), override=True)

def main():
    print("\n==================================================")
    print("  RUNNING BACKEND REGRESSION SUITE WITH NEW KEY")
    print("==================================================")
    test_dir = os.path.join(backend_dir, "tests")
    exit_code = pytest.main([test_dir, "-q"])
    print(f"Pytest Exit Code: {exit_code}")
    assert exit_code == 0, f"pytest failed with exit code {exit_code}"
    print("=> BACKEND PYTEST SUITE PASSED 100%!")

if __name__ == "__main__":
    main()
