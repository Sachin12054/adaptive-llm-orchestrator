import os
import sys
import json
import time

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.services.providers.gemini_provider import GeminiProvider
from app.schemas.provider import ProviderGenerationRequest

def run_step13_real_verification():
    print("=" * 80)
    print(" STEP 13 REAL GEMINI PROVIDER INTEGRATION VERIFICATION PASS")
    print("=" * 80)

    provider = GeminiProvider()
    status_resp = provider.get_status()

    print("\n--- 1. PROVIDER CONFIGURATION & STATUS CHECK ---")
    print(f"Provider Name       : {status_resp.provider}")
    print(f"API Key Configured  : {status_resp.configured}")
    print(f"Operational Status  : {status_resp.status_message}")
    print(f"Active GA Models    : {', '.join(status_resp.models)}")

    report_payload = {
        "status_check": status_resp.dict(),
        "api_generation_test": None
    }

    if not status_resp.configured:
        print("\n" + "!" * 80)
        print("REAL VERIFICATION RESULT NOTICE:")
        print("\"Real Gemini generation could not be verified because GEMINI_API_KEY is not configured.\"")
        print("To enable live generation, add your API key to .env: GEMINI_API_KEY=your_key_here")
        print("!" * 80)
    else:
        print("\n--- 2. REAL GEMINI API GENERATION TEST ---")
        preferred_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        target_model = next((m for m in preferred_models if m in status_resp.models), status_resp.models[0])
        test_prompt = "What is the capital of France?"

        print(f"Executing live Gemini request for active GA model '{target_model}'...")
        print(f"Test Prompt: \"{test_prompt}\"\n")

        req = ProviderGenerationRequest(
            model_id=target_model,
            prompt=test_prompt,
            temperature=0.2
        )

        t0 = time.perf_counter()
        resp = provider.generate(req)
        t1 = time.perf_counter()

        print(f"Execution Success   : {resp.success}")
        print(f"Target Model        : {resp.model_id}")
        print(f"Finish Reason       : {resp.finish_reason}")
        print(f"Measured Latency    : {resp.latency_ms} ms (Wrapper Total: {round((t1-t0)*1000, 2)} ms)")
        if resp.usage:
            print(f"Token Usage         : Input={resp.usage.input_tokens}, Output={resp.usage.output_tokens}, Total={resp.usage.total_tokens}")
        
        if resp.success:
            print("\nGenerated Text Response:\n")
            print(resp.generated_text)
        else:
            print(f"\nError Message: {resp.error_message}")

        report_payload["api_generation_test"] = resp.dict()

    # Save verification report
    report_output_path = os.path.join(project_root, "data", "logs", "step13_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nVerification report written to '{report_output_path}'.")
    print("=" * 80)

if __name__ == "__main__":
    run_step13_real_verification()
