import os
import sys
import json
import time
import inspect

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

import app.services.experience_buffer as buffer_module
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

def run_step20_real_verification():
    print("=" * 80)
    print(" STEP 20 STRUCTURED TRANSITION LOGGING & EXPERIENCE REPLAY BUFFER VERIFICATION")
    print("=" * 80)

    buffer_service = ExperienceBufferService(capacity=1000)
    pipeline = OrchestrationPipeline(experience_buffer=buffer_service)
    status_resp = buffer_service.get_status()

    # 1. SERVICE READINESS
    print("\n--- 1. SERVICE READINESS ---")
    print(f"Service Name         : {status_resp.service}")
    print(f"Status               : {status_resp.status.upper()}")
    print(f"Buffer Capacity      : {status_resp.capacity}")
    print(f"Current Size         : {status_resp.current_size}")
    print(f"State Dimension (D)  : {status_resp.state_dim}")
    print(f"Persistence Path     : {status_resp.persistence_path}")

    # 2. STATE ENCODING & ACTION MAPPING
    print("\n--- 2. ACTION MAPPING & STATE ENCODING ---")
    target_model = "gemma-3-4b"
    mapped_action_idx = ACTION_MAP.get(target_model, -1)
    print(f"Action Mapping       : '{target_model}' -> Action Index {mapped_action_idx}")

    # 3. REAL STEP 19 -> STEP 20 EXECUTION
    print("\n--- 3. REAL STEP 19 -> STEP 20 EXECUTION ---")
    prompt = "What is the capital of France?"
    print(f"Prompt: \"{prompt}\"")

    t0 = time.perf_counter()
    orchestration_res = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))
    t1 = time.perf_counter()

    latest_record = buffer_service._buffer[-1]

    print(f"Selected Model       : {orchestration_res.selected_model}")
    print(f"Step 18 Reward       : {orchestration_res.reward.reward:.4f}")
    print(f"Step 20 Recorded ID  : {latest_record.experience_id}")
    print(f"Step 20 Action Index : {latest_record.action}")
    print(f"Step 20 Action Model : {latest_record.action_model_id}")
    print(f"State Vector (D={len(latest_record.state)}) : {latest_record.state}")
    print(f"Next State           : {latest_record.next_state}")
    print(f"Done Flag            : {latest_record.done}")
    print(f"Buffer Size          : {len(buffer_service._buffer)}")

    # 4. REWARD PRESERVATION CHECK
    reward_preservation_pass = (abs(orchestration_res.reward.reward - latest_record.reward) < 1e-4)
    print(f"\nREWARD PRESERVATION: {'PASS' if reward_preservation_pass else 'FAIL'}")
    print(f"  Step 18 Reward: {orchestration_res.reward.reward:.4f} | Recorded Reward: {latest_record.reward:.4f}")

    # 5. ACTION / MODEL PRESERVATION CHECK
    action_preservation_pass = (orchestration_res.selected_model == latest_record.action_model_id)
    print(f"\nACTION PRESERVATION: {'PASS' if action_preservation_pass else 'FAIL'}")
    print(f"  Step 15 Selected Model: '{orchestration_res.selected_model}' | Recorded Model: '{latest_record.action_model_id}'")

    # 6. BUFFER CAPACITY & FIFO TEST
    scratch_buffer = ExperienceBufferService(capacity=5, persistence_path="data/rl/scratch_capacity_test.jsonl")
    scratch_buffer.clear()
    for _ in range(10):
        scratch_buffer.record_from_orchestration(orchestration_res)
    buffer_capacity_pass = (len(scratch_buffer._buffer) == 5)
    print(f"\nBUFFER CAPACITY: {'PASS' if buffer_capacity_pass else 'FAIL'} (Capacity: 5 | Final Size: {len(scratch_buffer._buffer)})")
    scratch_buffer.clear()

    # 7. MINIBATCH SAMPLING TEST
    samples = buffer_service.sample_batch(min(5, len(buffer_service._buffer)))
    batch_sampling_pass = (len(samples) > 0 and len(samples[0].state) == 12)
    print(f"\nBATCH SAMPLING: {'PASS' if batch_sampling_pass else 'FAIL'} (Sampled: {len(samples)} records)")

    # 8. ARCHITECTURAL BOUNDARY CHECKS
    source_code = inspect.getsource(buffer_module)
    has_direct_gemini_import = ("google.genai" in source_code or "google.generativeai" in source_code)
    print(f"\nDIRECT GEMINI SDK IN STEP 20: {has_direct_gemini_import}")

    prohibited_learning_methods = ["train", "fit", "learn", "update_policy"]
    found_learning = [m for m in prohibited_learning_methods if hasattr(buffer_service, m)]
    print(f"RL POLICY UPDATE IN STEP 20: {len(found_learning) > 0}")

    all_passed = (
        reward_preservation_pass and
        action_preservation_pass and
        buffer_capacity_pass and
        batch_sampling_pass and
        not has_direct_gemini_import and
        len(found_learning) == 0
    )

    print("\n" + "=" * 80)
    if all_passed:
        print("STEP 20 VERIFICATION: PASS")
    else:
        print("STEP 20 VERIFICATION: FAIL")
    print("=" * 80)

    # Save detailed JSON verification log
    report_payload = {
        "status_check": status_resp.dict(),
        "reward_preservation_pass": reward_preservation_pass,
        "action_preservation_pass": action_preservation_pass,
        "buffer_capacity_pass": buffer_capacity_pass,
        "batch_sampling_pass": batch_sampling_pass,
        "direct_gemini_sdk_in_step20": has_direct_gemini_import,
        "rl_policy_update_in_step20": (len(found_learning) > 0),
        "overall_verification_status": "PASS" if all_passed else "FAIL",
        "latest_experience_record": latest_record.dict()
    }

    report_output_path = os.path.join(project_root, "data", "logs", "step20_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"Verification output written to '{report_output_path}'.")

if __name__ == "__main__":
    run_step20_real_verification()
