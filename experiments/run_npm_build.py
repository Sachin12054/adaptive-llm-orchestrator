import subprocess
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
frontend_dir = os.path.join(project_root, "frontend")

def main():
    print("Running frontend production build: npm run build\n")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    res = subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, capture_output=True, text=True)

    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)
    print(f"Exit Code: {res.returncode}")
    return res.returncode

if __name__ == "__main__":
    sys.exit(main())
