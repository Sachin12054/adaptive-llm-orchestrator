import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

def audit_reward_integrity():
    print("=" * 80)
    print(" FOCUSED AUDIT: STEP 18 REWARD INTEGRITY & STEP 20 RECORD PRESERVATION")
    print("=" * 80)

    prompts = [
        "What is Python?",
        "Write Python code to read a CSV using pandas.",
        "Write a Python script that reads a CSV, cleans missing values, and trains a linear regression model."
    ]

    pipeline = OrchestrationPipeline()

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n--- PROMPT #{idx}: \"{prompt}\" ---")
        res = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))

        rew = res.reward
        bd = rew.reward_breakdown
        buffer_rec = pipeline.experience_buffer._buffer[-1]

        q_score = bd.quality.score
        c_score = bd.completeness.score
        r_score = bd.relevance.score
        v_score = bd.verification.score
        e_score = bd.execution.score

        q_contrib = bd.quality.contribution
        c_contrib = bd.completeness.contribution
        r_contrib = bd.relevance.contribution
        v_contrib = bd.verification.contribution
        e_contrib = bd.execution.contribution

        final_reward = rew.reward
        stored_reward = buffer_rec.reward
        state_vector = buffer_rec.state

        print(f"  Selected Model        : {res.selected_model}")
        print(f"  Quality Score         : {q_score:.4f} (Weight 0.30 -> Contrib: {q_contrib:.4f})")
        print(f"  Completeness Score    : {c_score:.4f} (Weight 0.20 -> Contrib: {c_contrib:.4f})")
        print(f"  Relevance Score       : {r_score:.4f} (Weight 0.25 -> Contrib: {r_contrib:.4f})")
        print(f"  Verification Score    : {v_score:.4f} (Weight 0.15 -> Contrib: {v_contrib:.4f})")
        print(f"  Execution Score       : {e_score:.4f} (Weight 0.10 -> Contrib: {e_contrib:.4f})")
        print(f"  Sum of Contributions  : {q_contrib + c_contrib + r_contrib + v_contrib + e_contrib:.4f}")
        print(f"  Final Step 18 Reward  : {final_reward:.4f}")
        print(f"  Step 20 Stored Reward : {stored_reward:.4f}")
        print(f"  State Vector Dim (D)  : {len(state_vector)} dimensions")
        print(f"  Shadow RL Proposal    : {res.decision.shadow_rl_decision['proposed_model']} (Shadow only)")

        assert abs(final_reward - stored_reward) < 1e-6, "Step 18 reward must exactly match Step 20 stored reward!"
        assert len(state_vector) == 12, "State vector dimension must be exactly 12!"
        assert res.decision.policy == "baseline_adaptive_policy", "Production routing authority must remain BaselineAdaptivePolicy!"

    print("\n" + "=" * 80)
    print(" ALL 8 INTEGRITY CHECKS VERIFIED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    audit_reward_integrity()
