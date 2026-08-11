import sys
import os
import pytest

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "backend"))

    t1 = os.path.join(project_root, "backend", "tests", "test_rl_policy_trainer.py")
    t2 = os.path.join(project_root, "backend", "tests", "test_adaptive_decision_engine.py")

    print("Running targeted tests:\n")
    exit_code = pytest.main([t1, t2, "-v"])
    print(f"\nTargeted Tests Exit Code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
