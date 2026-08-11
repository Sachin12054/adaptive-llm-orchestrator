import subprocess
import os
import sys
import pytest

def main():
    project_root = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    frontend_dir = os.path.join(project_root, "frontend")
    backend_dir = os.path.join(project_root, "backend")

    # Run correlation proof
    import experiments.verify_prompt_response_correlation
    experiments.verify_prompt_response_correlation.main()

    print("\n==================================================")
    print("  1. VITE FRONTEND BUILD (npm run build)")
    print("==================================================")
    env = os.environ.copy()
    build_res = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, env=env, capture_output=True, text=True)
    print(build_res.stdout)
    if build_res.stderr:
        print("STDERR:\n", build_res.stderr)
    assert build_res.returncode == 0, f"npm run build failed with exit code {build_res.returncode}"
    print("=> VITE FRONTEND BUILD PASSED WITH 0 ERRORS!\n")

    print("==================================================")
    print("  2. BACKEND PYTEST SUITE (pytest backend/tests -q)")
    print("==================================================")
    sys.path.insert(0, backend_dir)
    test_dir = os.path.join(backend_dir, "tests")
    exit_code = pytest.main([test_dir, "-q"])
    print(f"Pytest Exit Code: {exit_code}")
    assert exit_code == 0, f"pytest failed with exit code {exit_code}"
    print("=> BACKEND PYTEST SUITE PASSED 100%!\n")

if __name__ == "__main__":
    main()
