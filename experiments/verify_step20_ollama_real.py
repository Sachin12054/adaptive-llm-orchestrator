import os
import sys
import json
import inspect

# Add backend directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP
from app.schemas.orchestration import OrchestrationRequest
import app.services.providers.ollama_provider as ollama_mod

def verify_step20_ollama_real():
    print("=" * 80)
    print(" PHASE 15 — REAL STEP 20 OLLAMA EXPERIENCE VERIFICATION")
    print("=" * 80)

    buffer_service = ExperienceBufferService()
    pipeline = OrchestrationPipeline(experience_buffer=buffer_service)

    initial_size = buffer_service.get_status().current_size
    print(f"Initial Buffer Size  : {initial_size}")

    prompt = "Write a Python function to compute the factorial of a number."
    print(f"Executing Prompt     : \"{prompt}\"")

    res = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))

    print(f"\n--- 1. PIPELINE EXECUTION SUMMARY ---")
    print(f"Pipeline Success     : {res.success}")
    print(f"Selected Model       : {res.selected_model}")
    print(f"Provider             : {res.generation.provider}")
    print(f"Decision Score       : {res.decision_score:.4f}")
    print(f"Step 18 Scalar Reward: {res.reward.scalar_reward:.4f}")
    print(f"Generated Text Snippet: \"{(res.generation.generated_text or '').strip()[:80]}...\"")

    if not res.success or not res.selected_model or res.generation.provider != "ollama":
        print("\n[FAIL] E2E pipeline execution failed or did not use Ollama provider.")
        return False

    latest_rec = buffer_service._buffer[-1]

    print(f"\n--- 2. STEP 20 RECORDED EXPERIENCE VERIFICATION ---")
    print(f"Experience ID        : {latest_rec.experience_id}")
    print(f"Recorded Action Model: {latest_rec.action_model_id}")
    print(f"Recorded Action Index: {latest_rec.action}")
    print(f"Recorded Reward      : {latest_rec.reward:.4f}")
    print(f"State Dimension      : {len(latest_rec.state)}")
    print(f"State Vector         : {latest_rec.state}")
    print(f"Done Flag            : {latest_rec.done}")
    print(f"Next State           : {latest_rec.next_state}")

    # Invariant Verification Checks
    selected_model_pass = (latest_rec.action_model_id == res.selected_model)
    reward_preservation_pass = (abs(latest_rec.reward - res.reward.scalar_reward) < 1e-6)
    state_dim_pass = (len(latest_rec.state) == 12)
    done_pass = (latest_rec.done is True)
    next_state_pass = (latest_rec.next_state is None)
    action_idx_pass = (latest_rec.action in [0, 1, 2])

    source = inspect.getsource(ollama_mod)
    no_gemini_pass = ("google.genai" not in source and "google.generativeai" not in source)

    print(f"\n--- 3. INVARIANT CHECKS ---")
    print(f"Selected Model Match : {'PASS' if selected_model_pass else 'FAIL'}")
    print(f"Reward Preservation  : {'PASS' if reward_preservation_pass else 'FAIL'}")
    print(f"State Dim == 12      : {'PASS' if state_dim_pass else 'FAIL'}")
    print(f"Action Index Valid   : {'PASS' if action_idx_pass else 'FAIL'}")
    print(f"Done Flag == True    : {'PASS' if done_pass else 'FAIL'}")
    print(f"Next State == None   : {'PASS' if next_state_pass else 'FAIL'}")
    print(f"No Gemini SDK in RL  : {'PASS' if no_gemini_pass else 'FAIL'}")

    all_passed = (
        selected_model_pass and
        reward_preservation_pass and
        state_dim_pass and
        action_idx_pass and
        done_pass and
        next_state_pass and
        no_gemini_pass
    )

    print("\n" + "=" * 80)
    if all_passed:
        print("REAL STEP 20 OLLAMA VERIFICATION: PASS")
    else:
        print("REAL STEP 20 OLLAMA VERIFICATION: FAIL")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    verify_step20_ollama_real()
