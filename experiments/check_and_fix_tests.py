import pytest
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"), override=True)

if __name__ == "__main__":
    test_dir = os.path.join(backend_dir, "tests")
    exit_code = pytest.main([test_dir, "-v", "--tb=short"])
    print("Pytest exit code:", exit_code)
