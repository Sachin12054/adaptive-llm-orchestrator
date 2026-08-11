import sys
import os
import asyncio
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from experiments.test_no_fallback_failure import test_task_allocator_requires_policy_model
from experiments.verify_all_model_identities import run_identity_audit

def main():
    print("=== 1. TESTING TASK ALLOCATOR NO-FALLBACK FAILURE CASE ===")
    test_task_allocator_requires_policy_model()

    print("\n=== 2. RUNNING COMPLEX LOCAL & ONLINE VERIFICATION ===")
    asyncio.run(run_identity_audit())

    print("\n=== 3. RUNNING BACKEND REGRESSION SUITE (pytest backend/tests -q) ===")
    test_dir = os.path.join(backend_dir, "tests")
    exit_code = pytest.main([test_dir, "-q"])
    print(f"\nPytest Exit Code: {exit_code}")

if __name__ == "__main__":
    main()
