import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_real_e2e_api_tests import main as run_api_tests

if __name__ == "__main__":
    run_api_tests()
