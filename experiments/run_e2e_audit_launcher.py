import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.run_real_e2e_audit import main as run_audit

if __name__ == "__main__":
    run_audit()
