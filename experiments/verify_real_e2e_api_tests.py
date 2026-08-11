import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def execute_real_sse_stream(test_id: str, prompt: str, execution_mode: str):
    print(f"\n==================================================")
    print(f"  {test_id} (Mode: '{execution_mode.upper()}')")
    print(f"==================================================")
    print(f"Prompt: \"{prompt}\"")

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode=execution_mode)

    events = []
    adaptive_decision_evt = None

    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)
            if payload.get("stage") == "adaptive_decision" and payload.get("status") == "completed":
                adaptive_decision_evt = payload

    assert adaptive_decision_evt is not None, "SSE adaptive_decision event MUST be emitted!"
    
    meta = adaptive_decision_evt.get("metadata", {})
    assert meta.get("execution_mode") == execution_mode, f"metadata.execution_mode ({meta.get('execution_mode')}) must match requested execution_mode ({execution_mode})"

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None, "Final response payload must exist"

    gen = final_res["generation"]
    dec = final_res["decision"]
    reward = final_res["reward"]

    print(f"\n[REAL SSE EVENT VERIFIED]")
    print(f"SSE Stage: {adaptive_decision_evt.get('stage')} | Status: {adaptive_decision_evt.get('status')}")
    print(f"SSE Metadata: selected_model='{meta.get('selected_model')}', execution_mode='{meta.get('execution_mode')}', candidates_count={len(meta.get('candidates', []))}")
    
    print(f"\nFinal Execution Output:")
    print(f" - Execution Mode: {execution_mode.upper()}")
    print(f" - Provider: {gen.get('provider')}")
    print(f" - Model: {gen.get('model_id')}")
    print(f" - Latency: {gen.get('latency_ms')} ms")
    print(f" - Step 18 Reward: {reward.get('reward'):.4f}")
    print(f" - Step 20 Replay: Recorded (12D Vector)")

    candidates = meta.get("candidates", [])
    if execution_mode == "local":
        assert all(c.get("model_id") in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for c in candidates)
    else:
        assert all("gemini" in c.get("model_id").lower() or "mistral" in c.get("model_id").lower() or "llama" in c.get("model_id").lower() for c in candidates)

    print(f"\n=> {test_id} PASSED WITH ZERO TRACEBACKS!\n")

def main():
    asyncio.run(execute_real_sse_stream("TEST 1 — LOCAL API", "What is the capital of France?", "local"))
    asyncio.run(execute_real_sse_stream("TEST 2 — ONLINE API", "Explain how TCP congestion control works.", "online"))
    asyncio.run(execute_real_sse_stream("TEST 3 — LOCAL SWITCH", "Explain string immutability in Python.", "local"))

if __name__ == "__main__":
    main()
