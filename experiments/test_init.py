import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.services.adaptive_decision_engine import AdaptiveDecisionEngine

def main():
    engine = AdaptiveDecisionEngine()
    print(f"Decision engine initialized successfully with policy: '{engine.policy.policy_name}'")
    print(f"Shadow policy loaded: '{engine.shadow_rl_policy.policy_name}'")

if __name__ == "__main__":
    main()
