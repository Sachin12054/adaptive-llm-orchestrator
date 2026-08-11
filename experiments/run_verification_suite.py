import subprocess
import os
import sys

def main():
    cwd = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    backend_dir = os.path.join(cwd, "backend")
    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir

    out_file = os.path.join(cwd, "experiments", "audit_summary.txt")

    with open(out_file, "w") as f:
        f.write("=== 1. RUNNING REAL E2E AUDIT ===\n")
        res1 = subprocess.run([sys.executable, "experiments/execute_audit_now.py"], cwd=cwd, env=env, capture_output=True, text=True)
        f.write("STDOUT:\n" + res1.stdout + "\n")
        f.write("STDERR:\n" + res1.stderr + "\n")
        f.write(f"EXIT CODE: {res1.returncode}\n\n")

        f.write("=== 2. RUNNING BACKEND REGRESSION SUITE (pytest backend/tests -q) ===\n")
        res2 = subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "-q"], cwd=cwd, env=env, capture_output=True, text=True)
        f.write("STDOUT:\n" + res2.stdout + "\n")
        f.write("STDERR:\n" + res2.stderr + "\n")
        f.write(f"EXIT CODE: {res2.returncode}\n\n")

        f.write("=== 3. RUNNING NPM BUILD (Vite Frontend Build) ===\n")
        frontend_dir = os.path.join(cwd, "frontend")
        res3 = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, env=env, capture_output=True, text=True)
        f.write("STDOUT:\n" + res3.stdout + "\n")
        f.write("STDERR:\n" + res3.stderr + "\n")
        f.write(f"EXIT CODE: {res3.returncode}\n")

if __name__ == "__main__":
    main()
