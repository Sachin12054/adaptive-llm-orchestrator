import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_auto_routing_e2e import main as run_e2e

if __name__ == "__main__":
    run_e2e()
