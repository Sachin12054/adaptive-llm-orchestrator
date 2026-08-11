import os
import sys
import json
import time
import inspect

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

import app.services.reward_signal as reward_module
from app.services.reward_signal import RewardSignal
from app.schemas.reward import RewardComputeRequest

def run_step18_real_verification():
    print("=" * 80)
    print(" STEP 18 REAL REWARD SIGNAL / LEARNING FRAMEWORK VERIFICATION PASS")
    print("=" * 80)

    reward_service = RewardSignal()
    status_resp = reward_service.get_status()

    print("\n--- 1. REWARD SIGNAL SERVICE STATUS CHECK ---")
    print(f"Service Name                 : {status_resp.service}")
    print(f"Status                       : {status_resp.status.upper()}")
    print(f"Online Learning Active       : {status_resp.learning_active}")

    report_payload = {
        "status_check": status_resp.dict(),
        "architectural_checks": {},
        "cases": []
    }

    # Architectural Boundary Checks
    print("\n--- 2. ARCHITECTURAL BOUNDARY CHECKS ---")
    prohibited_methods = [
        "select_model", "choose_model", "rank_models", "best_model",
        "fallback_model", "route_request", "score_models",
        "generate_response", "regenerate", "retry_with_fallback",
        "train", "fit", "learn", "update_policy"
    ]
    found_prohibited = [m for m in prohibited_methods if hasattr(reward_service, m)]
    print(f"Prohibited RL/Selection Methods Found : {found_prohibited} (Expected: [])")

    source_code = inspect.getsource(reward_module)
    has_direct_gemini_import = ("google.genai" in source_code or "google.generativeai" in source_code)
    print(f"Direct Gemini SDK Imported            : {has_direct_gemini_import} (Expected: False)")

    report_payload["architectural_checks"] = {
        "prohibited_methods_found": found_prohibited,
        "has_direct_gemini_import": has_direct_gemini_import,
        "boundary_preserved": (len(found_prohibited) == 0 and not has_direct_gemini_import)
    }

    test_cases = [
        (
            "CASE 1 — Successful Verified Baseline Response",
            RewardComputeRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                winning_score=0.82,
                execution_success=True,
                execution_status="completed",
                generated_text="Paris is the capital of France.",
                verification_status="VERIFIED_BASELINE",
                verified=True,
                response_present=True,
                structural_quality_score=1.0,
                completeness_score=0.4,
                relevance_score=0.5,
                factual_verification_status="not_verified"
            ),
            0.7550
        ),
        (
            "CASE 2 — Empty Response (Execution Credited Only)",
            RewardComputeRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                winning_score=0.82,
                execution_success=True,
                execution_status="completed",
                generated_text=" ",
                verification_status="EMPTY_RESPONSE",
                verified=False,
                response_present=False,
                structural_quality_score=0.0,
                completeness_score=0.0,
                relevance_score=0.0,
                factual_verification_status="not_verified"
            ),
            0.1000
        ),
        (
            "CASE 3 — Unconfigured Generation Execution",
            RewardComputeRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                winning_score=0.82,
                execution_success=False,
                execution_status="not_configured",
                generated_text=None,
                verification_status="NOT_VERIFIABLE",
                verified=False,
                response_present=False,
                structural_quality_score=0.0,
                completeness_score=0.0,
                relevance_score=0.0,
                factual_verification_status="not_verified"
            ),
            0.0000
        ),
        (
            "CASE 4 — Failed Generation Execution",
            RewardComputeRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                winning_score=0.82,
                execution_success=False,
                execution_status="failed",
                generated_text=None,
                verification_status="NOT_VERIFIABLE",
                verified=False,
                response_present=False,
                structural_quality_score=0.0,
                completeness_score=0.0,
                relevance_score=0.0,
                factual_verification_status="not_verified"
            ),
            0.0000
        )
    ]

    print("\n--- 3. REWARD COMPUTATION ACROSS 4 TEST CASES ---")

    for idx, (label, req, expected_reward) in enumerate(test_cases, 1):
        print(f"\n[{idx}/4] {label}")
        print(f"  Prompt: \"{req.prompt}\" | Model: {req.selected_model} | Execution: {req.execution_status}")

        t0 = time.perf_counter()
        res = reward_service.compute_reward(req)
        t1 = time.perf_counter()

        print(f"  Calculated Reward           : {res.reward:.4f} (Expected: {expected_reward:.4f})")
        print(f"  Reward Status               : {res.reward_status.upper()}")
        print(f"  Factual Verification Status : {res.factual_verification_status}")
        print("  Component Breakdown:")
        print(f"    - Quality (weight=0.30)   : Score={res.reward_breakdown.quality.score:.2f} -> Contribution={res.reward_breakdown.quality.contribution:.4f}")
        print(f"    - Complete (weight=0.20)  : Score={res.reward_breakdown.completeness.score:.2f} -> Contribution={res.reward_breakdown.completeness.contribution:.4f}")
        print(f"    - Relevance (weight=0.25) : Score={res.reward_breakdown.relevance.score:.2f} -> Contribution={res.reward_breakdown.relevance.contribution:.4f}")
        print(f"    - Verify (weight=0.15)    : Score={res.reward_breakdown.verification.score:.2f} -> Contribution={res.reward_breakdown.verification.contribution:.4f}")
        print(f"    - Exec (weight=0.10)      : Score={res.reward_breakdown.execution.score:.2f} -> Contribution={res.reward_breakdown.execution.contribution:.4f}")
        print(f"  Latency                     : {res.latency_ms} ms (Wrapper: {round((t1-t0)*1000, 2)} ms)")
        print("  Reasoning Bullets           :")
        for b in res.reasoning:
            print(f"    • {b}")

        report_payload["cases"].append({
            "case_label": label,
            "expected_reward": expected_reward,
            "request": req.dict(),
            "response": res.dict()
        })

    # Save detailed verification report
    report_output_path = os.path.join(project_root, "data", "logs", "step18_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Verification output written to '{report_output_path}'.")
    print("EXPLICIT STATEMENT: This is a deterministic baseline reward signal for future RL learning. It is NOT a trained RL policy.")
    print("=" * 80)

if __name__ == "__main__":
    run_step18_real_verification()
