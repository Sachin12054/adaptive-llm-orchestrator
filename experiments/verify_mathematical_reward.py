import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

def main():
    pipeline = OrchestrationPipeline()

    prompts = [
        "What is Python?",
        "What is a Database?",
        "What is AI?",
        "Explain Python.",
        "Define machine learning.",
        "What is recursion?",
        "Write Python code to read a CSV using pandas.",
        "Implement binary search in Python and explain its time complexity.",
        "Design a distributed Python ML pipeline for millions of records per day using Kafka, Spark, PostgreSQL, object storage, model training, versioning, monitoring, and fault tolerance.",
        "Make it better."
    ]

    print("=" * 90)
    print("MATHEMATICAL REWARD FORMULA VERIFICATION (STEP 18)")
    print("Formula: 0.30*Qual + 0.20*Cmpl + 0.25*Rel + 0.15*Ver + 0.10*Exec")
    print("=" * 90)

    all_passed = True
    for p in prompts:
        req = OrchestrationRequest(prompt=p)
        res = pipeline.run_pipeline(req)
        r = res.reward
        b = r.reward_breakdown

        calc_reward = (
            0.30 * b.quality.score +
            0.20 * b.completeness.score +
            0.25 * b.relevance.score +
            0.15 * b.verification.score +
            0.10 * b.execution.score
        )
        calc_reward = round(calc_reward, 4)
        actual_reward = round(r.reward, 4)

        match = abs(calc_reward - actual_reward) < 1e-4
        if not match:
            all_passed = False

        print(f"\nPrompt: '{p}'")
        print(f"  Qual: {b.quality.score:.2f} | Cmpl: {b.completeness.score:.2f} | Rel: {b.relevance.score:.2f} | Ver: {b.verification.score:.2f} | Exec: {b.execution.score:.2f}")
        print(f"  Calculated Reward: {calc_reward:.4f} | Backend Output Reward: {actual_reward:.4f} | Math Match: {'EXACT MATCH (PASS)' if match else 'FAIL'}")

    print("\n" + "=" * 90)
    print(f"REWARD FORMULA MATHEMATICAL VERIFICATION RESULT: {'100% MATHEMATICALLY VERIFIED (PASS)' if all_passed else 'FAIL'}")
    print("=" * 90)

if __name__ == "__main__":
    main()
