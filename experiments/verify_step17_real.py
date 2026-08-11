import os
import sys
import json
import time
import inspect

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

import app.services.response_verifier as verifier_module
from app.services.response_verifier import ResponseVerifier
from app.schemas.verification import VerificationRequest

def run_step17_real_verification():
    print("=" * 80)
    print(" STEP 17 REAL RESPONSE VERIFIER VERIFICATION PASS (BASELINE EVALUATION ONLY)")
    print("=" * 80)

    verifier = ResponseVerifier()
    status_resp = verifier.get_status()

    print("\n--- 1. RESPONSE VERIFIER STATUS CHECK ---")
    print(f"Service Name                    : {status_resp.service}")
    print(f"Status                          : {status_resp.status.upper()}")
    print(f"Factual Verification Active     : {status_resp.factual_verification_available}")

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
        "generate_response", "regenerate", "retry_with_fallback"
    ]
    found_prohibited = [m for m in prohibited_methods if hasattr(verifier, m)]
    print(f"Prohibited Methods Found        : {found_prohibited} (Expected: [])")

    source_code = inspect.getsource(verifier_module)
    has_direct_gemini_import = ("google.genai" in source_code or "google.generativeai" in source_code)
    print(f"Direct Gemini SDK Imported       : {has_direct_gemini_import} (Expected: False)")

    report_payload["architectural_checks"] = {
        "prohibited_methods_found": found_prohibited,
        "has_direct_gemini_import": has_direct_gemini_import,
        "boundary_preserved": (len(found_prohibited) == 0 and not has_direct_gemini_import)
    }

    test_cases = [
        (
            "CASE 1: Valid Baseline Generated Response",
            VerificationRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                generated_text="Paris is the capital of France.",
                generation_success=True,
                execution_status="completed"
            )
        ),
        (
            "CASE 2: Empty Whitespace Response",
            VerificationRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                generated_text="   ",
                generation_success=True,
                execution_status="completed"
            )
        ),
        (
            "CASE 3: Unconfigured Generation Execution",
            VerificationRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                generated_text=None,
                generation_success=False,
                execution_status="not_configured"
            )
        ),
        (
            "CASE 4: Failed Generation Execution",
            VerificationRequest(
                prompt="What is the capital of France?",
                selected_model="gemini-2.5-flash",
                generated_text=None,
                generation_success=False,
                execution_status="failed"
            )
        )
    ]

    print("\n--- 3. VERIFICATION EVALUATION ACROSS 4 TEST CASES ---")

    for idx, (label, req) in enumerate(test_cases, 1):
        print(f"\n[{idx}/4] {label}")
        print(f"  Prompt: \"{req.prompt}\" | Model: {req.selected_model} | Execution: {req.execution_status}")

        t0 = time.perf_counter()
        res = verifier.verify_response(req)
        t1 = time.perf_counter()

        print(f"  Verified                    : {res.verified}")
        print(f"  Verification Status         : {res.verification_status.upper()}")
        print(f"  Response Present            : {res.response_present}")
        print(f"  Relevance Score             : {res.relevance_score:.4f}")
        print(f"  Structural Quality Score    : {res.structural_quality_score:.4f}")
        print(f"  Factual Verification Status : {res.factual_verification_status}")
        print(f"  Issues                      : {res.issues}")
        print(f"  Latency                     : {res.verification_latency_ms} ms (Wrapper: {round((t1-t0)*1000, 2)} ms)")
        print("  Reasoning Bullets           :")
        for b in res.verification_reasoning:
            print(f"    • {b}")

        report_payload["cases"].append({
            "case_label": label,
            "request": req.dict(),
            "response": res.dict()
        })

    # Save detailed verification report
    report_output_path = os.path.join(project_root, "data", "logs", "step17_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Verification output written to '{report_output_path}'.")
    print("EXPLICIT STATEMENT: This baseline Response Verifier evaluates structural quality and baseline prompt relevance ONLY.")
    print("It does NOT establish factual correctness, and does NOT generate or regenerate responses.")
    print("=" * 80)

if __name__ == "__main__":
    run_step17_real_verification()
