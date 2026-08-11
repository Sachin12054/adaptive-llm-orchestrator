import subprocess
import os
import sys

def main():
    cwd = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    frontend_dir = os.path.join(cwd, "frontend")
    env = os.environ.copy()

    print("Executing Vite frontend build: npm run build...")
    res = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, env=env, capture_output=True, text=True)
    print("EXIT CODE:", res.returncode)
    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

    assert res.returncode == 0, "npm run build failed!"
    print("=> VITE FRONTEND BUILD SUCCEEDED WITH 0 ERRORS!")

if __name__ == "__main__":
    main()
