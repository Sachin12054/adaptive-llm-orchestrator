import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_stage_5_naming import main as run_stage5_test

if __name__ == "__main__":
    run_stage5_test()
