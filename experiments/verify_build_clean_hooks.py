import subprocess
import os
import sys

def main():
    project_root = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    frontend_dir = os.path.join(project_root, "frontend")

    print("\n==================================================")
    print("  RUNNING VITE FRONTEND BUILD (npm run build)")
    print("==================================================")
    env = os.environ.copy()
    build_res = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, env=env, capture_output=True, text=True)
    print(build_res.stdout)
    if build_res.stderr:
        print("STDERR:\n", build_res.stderr)

    assert build_res.returncode == 0, f"npm run build failed with exit code {build_res.returncode}"
    print("=> VITE FRONTEND BUILD PASSED WITH EXACTLY 0 ERRORS!")

if __name__ == "__main__":
    main()
