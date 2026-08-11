import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_controlled_fallback import main as run_fallback_verify

if __name__ == "__main__":
    run_fallback_verify()
