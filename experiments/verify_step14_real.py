import os
import sys
import json
import time

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.services.model_manager import ModelManager
from app.schemas.model_manager import ModelExecutionRequest

def run_step14_real_verification():
    print("=" * 80)
    print(" STEP 14 REAL MODEL MANAGER VERIFICATION PASS (NO FALLBACK / NO ROUTING LOGIC)")
    print("=" * 80)

    manager = ModelManager()
    status_resp = manager.get_status()

    print("\n--- 1. MODEL MANAGER STATUS & COMPONENT OVERVIEW ---")
    print(f"Manager Status          : {status_resp.status.upper()}")
    print(f"Registered Models Count : {status_resp.registered_models_count}")
    print(f"Executable LLMs Count   : {status_resp.executable_llm_models_count}")
    print("Providers Status:")
    for prov in status_resp.providers_status:
        print(f"  - Provider: {prov['provider']:<15} | Configured: {prov['configured']} | Available: {prov['available']}")
        print(f"    Models: {', '.join(prov['registered_models'])}")

    report_payload = {
        "status_check": status_resp.dict(),
        "embedding_rejection_test": None,
        "unregistered_rejection_test": None,
        "live_execution_test": None
    }

    # 2. Test Embedding Model Rejection (BAAI/bge-m3)
    print("\n--- 2. EMBEDDING MODEL REJECTION TEST (BAAI/bge-m3) ---")
    req_bge = ModelExecutionRequest(model_id="BAAI/bge-m3", prompt="What is the capital of France?")
    resp_bge = manager.execute(req_bge)
    print(f"Requested Model     : {resp_bge.model_id}")
    print(f"Execution Success   : {resp_bge.success} (Expected: False)")
    print(f"Execution Status    : {resp_bge.execution_status}")
    print(f"Error Message       : {resp_bge.error_message}")
    report_payload["embedding_rejection_test"] = resp_bge.dict()

    # 3. Test Unregistered Model Rejection without Fallback
    print("\n--- 3. UNREGISTERED MODEL REJECTION TEST (no fallback) ---")
    req_unreg = ModelExecutionRequest(model_id="unregistered-llm-xyz", prompt="What is the capital of France?")
    resp_unreg = manager.execute(req_unreg)
    print(f"Requested Model     : {resp_unreg.model_id}")
    print(f"Execution Success   : {resp_unreg.success} (Expected: False)")
    print(f"Execution Status    : {resp_unreg.execution_status}")
    print(f"Error Message       : {resp_unreg.error_message}")
    report_payload["unregistered_rejection_test"] = resp_unreg.dict()

    # 4. Live Provider Execution (Conditional on API Key)
    print("\n--- 4. LIVE MODEL EXECUTION TEST ---")
    gemini_prov_status = next((p for p in status_resp.providers_status if p['provider'] == "Google Gemini"), None)
    
    if not gemini_prov_status or not gemini_prov_status['configured']:
        print("\n" + "!" * 80)
        print("REAL VERIFICATION RESULT NOTICE:")
        print("\"Real model execution could not be verified because GEMINI_API_KEY is not configured.\"")
        print("To enable live execution, add your API key to .env: GEMINI_API_KEY=your_key_here")
        print("!" * 80)
    else:
        target_model = "gemini-2.5-flash"
        print(f"Executing ModelManager request for registered model '{target_model}'...")
        req_live = ModelExecutionRequest(
            model_id=target_model,
            prompt="What is the capital of France?",
            temperature=0.2
        )

        t0 = time.perf_counter()
        resp_live = manager.execute(req_live)
        t1 = time.perf_counter()

        print(f"Execution Success   : {resp_live.success}")
        print(f"Target Model        : {resp_live.model_id}")
        print(f"Provider Adapter    : {resp_live.provider}")
        print(f"Execution Status    : {resp_live.execution_status}")
        print(f"Measured Latency    : {resp_live.latency_ms} ms (Total Wrapper: {round((t1-t0)*1000, 2)} ms)")
        if resp_live.usage:
            print(f"Token Usage         : Total={resp_live.usage.total_tokens}")

        if resp_live.success:
            print("\nGenerated Response Text:\n")
            print(resp_live.generated_text)
        else:
            print(f"\nError Message: {resp_live.error_message}")

        report_payload["live_execution_test"] = resp_live.dict()

    # Save detailed verification log
    report_output_path = os.path.join(project_root, "data", "logs", "step14_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nVerification report written to '{report_output_path}'.")
    print("=" * 80)

if __name__ == "__main__":
    run_step14_real_verification()
