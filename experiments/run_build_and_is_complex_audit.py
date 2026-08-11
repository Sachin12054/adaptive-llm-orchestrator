import subprocess
import os
import sys

def main():
    project_root = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    frontend_dir = os.path.join(project_root, "frontend")
    frontend_src = os.path.join(frontend_dir, "src")

    print("\n==================================================")
    print("  1. GREP / SEARCH FOR ALL `isComplex` OCCURRENCES")
    print("==================================================")
    matches = []
    for root, _, files in os.walk(frontend_src):
        for f in files:
            if f.endswith(".jsx") or f.endswith(".js"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fc:
                    for line_num, line in enumerate(fc.readlines(), 1):
                        if "isComplex" in line:
                            matches.append((os.path.relpath(path, frontend_src), line_num, line.strip()))

    print(f"Total occurrences of 'isComplex': {len(matches)}\n")
    for rel_path, line_num, snippet in matches:
        print(f"File: {rel_path}:L{line_num} | {snippet}")

    print("\n==================================================")
    print("  2. RUNNING VITE FRONTEND BUILD (npm run build)")
    print("==================================================")
    env = os.environ.copy()
    build_res = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, env=env, capture_output=True, text=True)
    print(build_res.stdout)
    if build_res.stderr:
        print("STDERR:\n", build_res.stderr)

    assert build_res.returncode == 0, f"npm run build failed with exit code {build_res.returncode}!"
    print("=> VITE FRONTEND BUILD PASSED WITH 0 ERRORS!\n")

if __name__ == "__main__":
    main()
