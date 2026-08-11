import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_dynamic_candidate_payloads import main as run_candidate_verification

if __name__ == "__main__":
    run_candidate_verification()
