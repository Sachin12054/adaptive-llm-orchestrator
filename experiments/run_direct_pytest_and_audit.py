import sys
import os
import json
import asyncio
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from experiments.run_real_e2e_audit import run_e2e_audit

def main():
    print("=== 1. RUNNING REAL E2E AUDIT ===")
    asyncio.run(run_e2e_audit())

    print("\n=== 2. RUNNING BACKEND REGRESSION SUITE (pytest backend/tests -q) ===")
    test_dir = os.path.join(backend_dir, "tests")
    exit_code = pytest.main([test_dir, "-q"])
    print(f"\nPytest Exit Code: {exit_code}")

if __name__ == "__main__":
    main()
