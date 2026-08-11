import os
import sys
import json
import urllib.request

# Add backend directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.model_manager import ModelManager
from app.schemas.model_manager import ModelExecutionRequest
from app.services.providers.ollama_provider import OllamaProvider

def verify_ollama_integration():
    print("=" * 80)
    print(" PHASE 14 — OLLAMA INTEGRATION & REAL MODEL EXECUTION VERIFICATION")
    print("=" * 80)

    provider = OllamaProvider()
    status = provider.get_status()

    print(f"\nOllama Status Message : {status.status_message}")
    print(f"Ollama Server Configured: {status.configured}")
    print(f"Ollama Server Available : {status.available}")

    if not status.available:
        print("\n[FAIL] Ollama server is not running on http://localhost:11434.")
        print("Start Ollama locally before running verification.")
        return False

    # Check installed models
    installed_models = []
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed_models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
    except Exception as e:
        print(f"\n[ERROR] Failed to query Ollama tags: {str(e)}")

    print(f"Installed Ollama Models : {installed_models}")

    target_models = ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    test_prompts = {
        "gemma-3-4b": "What is 2 + 2?",
        "qwen-coder-3b": "Write a Python function that returns the square of a number.",
        "deepseek-r1-7b": "What is 12 * 12? Explain briefly."
    }

    manager = ModelManager()
    all_success = True

    for model_id in target_models:
        print(f"\n--- Testing Model Execution: '{model_id}' ---")
        if model_id not in installed_models:
            print(f"[WARNING] Model '{model_id}' is not currently installed in Ollama.")
            print(f"Installed models: {installed_models}")
            all_success = False
            continue

        prompt = test_prompts[model_id]
        print(f"Prompt               : \"{prompt}\"")

        res = manager.execute(ModelExecutionRequest(model_id=model_id, prompt=prompt))

        print(f"Execution Status     : {res.execution_status}")
        print(f"Provider Executed    : {res.provider}")
        print(f"Latency (ms)         : {res.latency_ms}")
        print(f"Generated Text       : \"{(res.generated_text or '').strip()[:100]}...\"")

        is_valid = (
            res.success and
            res.provider == "ollama" and
            res.generated_text is not None and
            len(res.generated_text.strip()) > 0 and
            res.execution_status == "completed"
        )

        print(f"Result Verification  : {'PASS' if is_valid else 'FAIL'}")
        if not is_valid:
            all_success = False
            if res.error_message:
                print(f"Error Message        : {res.error_message}")

    print("\n" + "=" * 80)
    if all_success:
        print("OLLAMA INTEGRATION VERIFICATION: PASS")
    else:
        print("OLLAMA INTEGRATION VERIFICATION: FAIL (Check model availability)")
    print("=" * 80)
    return all_success

if __name__ == "__main__":
    verify_ollama_integration()
