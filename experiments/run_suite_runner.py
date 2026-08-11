import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

import experiments.run_e2e_validation_suite

if __name__ == "__main__":
    experiments.run_e2e_validation_suite.main_suite()
