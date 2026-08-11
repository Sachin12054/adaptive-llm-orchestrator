import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.run_npm_build import main as run_build

if __name__ == "__main__":
    run_build()
