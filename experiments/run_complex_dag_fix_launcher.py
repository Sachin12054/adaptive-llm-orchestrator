import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_complex_dag_fix import main as run_verify

if __name__ == "__main__":
    run_verify()
