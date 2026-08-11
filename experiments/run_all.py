import sys
import os
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)
sys.path.insert(0, project_root)

from verify_step20_real import run_step20_real_verification
from verify_step21_real import run_step21_real_verification
from verify_ollama_integration import verify_ollama_integration
from verify_step20_ollama_real import verify_step20_ollama_real

def run():
    print("=" * 80)
    print(" RUNNING FULL REGRESSION TEST SUITE (pytest)")
    print("=" * 80)

    test_path = os.path.join(project_root, "backend", "tests")
    exit_code = pytest.main([test_path, "-v", "--tb=short"])

    print("\n" + "=" * 80)
    print(f" PYTEST COMPLETED WITH EXIT CODE: {exit_code}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print(" RUNNING OLLAMA INTEGRATION VERIFICATION")
    print("=" * 80)
    verify_ollama_integration()

    print("\n" + "=" * 80)
    print(" RUNNING REAL STEP 20 OLLAMA VERIFICATION")
    print("=" * 80)
    verify_step20_ollama_real()

if __name__ == "__main__":
    run()
