import subprocess
import os
import sys

def main():
    cwd = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    backend_dir = os.path.join(cwd, "backend")
    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir

    print("=== 1. VERIFYING BACKEND REGRESSION SUITE (pytest backend/tests -q) ===")
    res_pytest = subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "-q"], cwd=cwd, env=env, capture_output=True, text=True)
    print("PYTEST EXIT CODE:", res_pytest.returncode)
    print("PYTEST STDOUT:\n", res_pytest.stdout)
    if res_pytest.stderr:
        print("PYTEST STDERR:\n", res_pytest.stderr)

    print("\n=== 2. VERIFYING FRONTEND BUILD (npm run build) ===")
    frontend_dir = os.path.join(cwd, "frontend")
    res_build = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, env=env, capture_output=True, text=True)
    print("NPM BUILD EXIT CODE:", res_build.returncode)
    print("NPM BUILD STDOUT:\n", res_build.stdout)
    if res_build.stderr:
        print("NPM BUILD STDERR:\n", res_build.stderr)

if __name__ == "__main__":
    main()
