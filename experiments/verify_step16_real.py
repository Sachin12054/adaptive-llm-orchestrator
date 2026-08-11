import os
import sys
import json
import time
import inspect

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

import app.services.response_generator as rg_module
from app.services.response_generator import ResponseGenerator
from app.schemas.response import ResponseGenerationRequest

def run_step16_real_verification():
    print("=" * 80)
    print(" STEP 16 REAL RESPONSE GENERATOR VERIFICATION PASS (NO FAKE TEXT GENERATION)")
    print("=" * 80)

    generator = ResponseGenerator()
    status_resp = generator.get_status()

    print("\n--- 1. RESPONSE GENERATOR STATUS CHECK ---")
    print(f"Service Name              : {status_resp.service}")
    print(f"Status                    : {status_resp.status.upper()}")
    print(f"ModelManager Available    : {status_resp.model_manager_available}")

    report_payload = {
        "status_check": status_resp.dict(),
        "architectural_checks": {},
        "unconfigured_execution_test": None
    }

    # 2. Architectural Boundary Checks
    print("\n--- 2. ARCHITECTURAL BOUNDARY CHECKS ---")
    forbidden_attrs = ["select_model", "choose_model", "rank_models", "best_model", "fallback_model", "route_request", "score_models"]
    found_forbidden = [attr for attr in forbidden_attrs if hasattr(generator, attr)]
    print(f"Forbidden Selection Methods Found : {found_forbidden} (Expected: [])")

    source_code = inspect.getsource(rg_module)
    has_direct_gemini_import = ("google.genai" in source_code or "google.generativeai" in source_code)
    print(f"Direct Gemini SDK Imported        : {has_direct_gemini_import} (Expected: False)")

    report_payload["architectural_checks"] = {
        "forbidden_methods_found": found_forbidden,
        "has_direct_gemini_import": has_direct_gemini_import,
        "boundary_preserved": (len(found_forbidden) == 0 and not has_direct_gemini_import)
    }

    # 3. Real Unconfigured Generation Test
    print("\n--- 3. REAL RESPONSE GENERATION TEST (UNCONFIGURED API KEY) ---")
    target_model = "gemini-2.5-flash"
    test_prompt = "What is the capital of France?"

    print(f"Requesting ResponseGenerator with selected_model='{target_model}'...")
    print(f"Prompt: \"{test_prompt}\"\n")

    req = ResponseGenerationRequest(
        prompt=test_prompt,
        selected_model=target_model
    )

    t0 = time.perf_counter()
    resp = generator.generate_response(req)
    t1 = time.perf_counter()

    latency_ms = round((t1 - t0) * 1000, 2)

    print(f"Execution Success       : {resp.success} (Expected: False when unconfigured)")
    print(f"Preserved Model ID      : {resp.model_id}")
    print(f"Provider                : {resp.provider}")
    print(f"Execution Status        : {resp.execution_status}")
    print(f"Generated Text          : {resp.generated_text} (Expected: None)")
    print(f"Measured Latency        : {resp.latency_ms} ms (Total Wrapper: {latency_ms} ms)")
    print(f"Error Message           : {resp.error_message}")

    report_payload["unconfigured_execution_test"] = resp.dict()

    print("\n" + "!" * 80)
    print("REAL VERIFICATION RESULT NOTICE:")
    print("\"Real Gemini generation could not be verified because GEMINI_API_KEY is not configured.\"")
    print("To enable live text generation, add your API key to .env: GEMINI_API_KEY=your_key_here")
    print("!" * 80)

    # Save detailed verification log
    report_output_path = os.path.join(project_root, "data", "logs", "step16_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nVerification report written to '{report_output_path}'.")
    print("=" * 80)

if __name__ == "__main__":
    run_step16_real_verification()
