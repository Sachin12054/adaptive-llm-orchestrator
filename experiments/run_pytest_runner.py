import subprocess
import os

def run_tests():
    cwd = r"C:\Users\sachi\Desktop\Amrita\Sem-7\RL\Project\adaptive-llm-orchestrator"
    backend_dir = os.path.join(cwd, "backend")
    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir
    res = subprocess.run(["python", "-m", "pytest", "backend/tests", "-q"], cwd=cwd, env=env, capture_output=True, text=True)
    print("RETURN CODE:", res.returncode)
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)

if __name__ == "__main__":
    run_tests()
