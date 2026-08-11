import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_all_8_tests_e2e import main as run_e2e_all

if __name__ == "__main__":
    run_e2e_all()
