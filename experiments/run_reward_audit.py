import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.test_reward_variance import main as run_reward_audit

if __name__ == "__main__":
    run_reward_audit()
