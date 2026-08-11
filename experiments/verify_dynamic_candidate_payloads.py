import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def inspect_candidate_payload(test_name: str, prompt: str, execution_mode: str):
    print(f"\n==================================================")
    print(f"  {test_name} (Mode: '{execution_mode.upper()}')")
    print(f"==================================================")
    print(f"Prompt: \"{prompt}\"")

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode=execution_mode)

    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None, "Final response payload must exist"

    dec = final_res["decision"]
    gen = final_res["generation"]
    candidates = dec.get("candidates", [])

    print(f"\nObserved Backend Candidate Payload for {test_name}:")
    print(f" - Execution Mode: {execution_mode.upper()}")
    print(f" - Winning Provider: {gen.get('provider')}")
    print(f" - Winning Model: {gen.get('model_id')}")
    print(f" - Total Candidates Evaluated by Policy: {len(candidates)}")
    print(f"\nCandidate Breakdown Details:")

    for c in candidates:
        model_id = c.get("model_id")
        provider = c.get("provider") or c.get("provider_name") or "N/A"
        display_name = c.get("display_name") or model_id
        score = c.get("candidate_score", 0.0) * 100
        eligible = c.get("eligible")
        is_selected = (model_id == dec.get("selected_model"))
        
        status_label = "[ SELECTED ]" if is_selected else ("[ ELIGIBLE ]" if eligible else "[ INELIGIBLE ]")
        print(f"   • Model: {display_name} ({model_id}) | Provider: {provider} | Status: {status_label} | Score: {score:.1f}%")

    if execution_mode == "local":
        assert all(c.get("model_id") in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for c in candidates)
    else:
        assert all("gemini" in c.get("model_id").lower() or "mistral" in c.get("model_id").lower() or "llama" in c.get("model_id").lower() for c in candidates)

    print(f"\n=> {test_name} PASSED SUCCESSFULLY!\n")

def main():
    asyncio.run(inspect_candidate_payload("TEST A (LOCAL Coding)", "Write a Python script to sort a list.", "local"))
    asyncio.run(inspect_candidate_payload("TEST B (ONLINE Factual)", "What is the capital of France?", "online"))
    asyncio.run(inspect_candidate_payload("TEST C (ONLINE Coding)", "Write a production-ready Python FastAPI service.", "online"))

if __name__ == "__main__":
    main()
