import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

def main():
    pipeline = OrchestrationPipeline()

    test_prompts = [
        "What is Python?",
        "Explain Python in one sentence.",
        "Write a Python function for binary search.",
        "def foo(",  # Incomplete / invalid prompt
        "What is a database?",
        "Design a distributed Python ML pipeline using Kafka, Spark, and XGBoost."
    ]

    print("=== REWARD SIGNAL VARIANCE AUDIT ===")
    for p in test_prompts:
        req = OrchestrationRequest(prompt=p)
        res = pipeline.run_pipeline(req)

        reward_res = res.reward
        bk = reward_res.reward_breakdown

        print(f"\nPrompt: '{p}'")
        print(f"  Model Selected: {res.selected_model}")
        print(f"  Final Reward: {reward_res.reward:.4f}")
        print("  Component Breakdown:")
        print(f"    - Structural Quality:  score={bk.quality.score:.2f}, weight={bk.quality.weight}, contribution={bk.quality.contribution:.4f}")
        print(f"    - Completeness:          score={bk.completeness.score:.2f}, weight={bk.completeness.weight}, contribution={bk.completeness.contribution:.4f}")
        print(f"    - Relevance:             score={bk.relevance.score:.2f}, weight={bk.relevance.weight}, contribution={bk.relevance.contribution:.4f}")
        print(f"    - Verification Pass:     score={bk.verification.score:.2f}, weight={bk.verification.weight}, contribution={bk.verification.contribution:.4f}")
        print(f"    - Execution Success:     score={bk.execution.score:.2f}, weight={bk.execution.weight}, contribution={bk.execution.contribution:.4f}")

if __name__ == "__main__":
    main()
