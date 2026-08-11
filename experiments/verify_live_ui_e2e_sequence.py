import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def run_ui_sequence_test(test_id: str, mode: str, prompt: str):
    print(f"\n==================================================")
    print(f"  {test_id} (Mode: '{mode.upper()}')")
    print(f"==================================================")
    print(f"Prompt: \"{prompt}\"")

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode=mode)

    events = []
    sse_decision_evt = None

    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)
            if payload.get("stage") == "adaptive_decision" and payload.get("status") == "completed":
                sse_decision_evt = payload

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None, "Final response payload must exist"
    assert sse_decision_evt is not None, "SSE adaptive_decision event must exist"

    dec = final_res["decision"]
    gen = final_res["generation"]

    header_label = "Online API Pool" if mode == "online" else "100% Local Ollama"
    candidates = sse_decision_evt.get("metadata", {}).get("candidates", [])

    print(f"\nLive SSE & Backend Decision Payload for {test_id}:")
    print(f" - UI Mode Header Label: '{header_label}'")
    print(f" - Selected Model: {dec.get('selected_model')}")
    print(f" - Selected Provider: {gen.get('provider')}")
    print(f" - SSE Candidates Transmitted: {len(candidates)}")

    selected_count = 0
    for c in candidates:
        m_id = c.get("model_id")
        provider = c.get("provider") or c.get("provider_name") or "N/A"
        score_pct = (c.get("candidate_score", 0.0) * 100)
        is_sel = (m_id == dec.get("selected_model"))
        if is_sel:
            selected_count += 1
        status = "[ SELECTED ]" if is_sel else ("[ ELIGIBLE ]" if c.get("eligible") else "[ INELIGIBLE ]")
        print(f"    ├─ {m_id} ({provider}) -> {status} Score: {score_pct:.1f}%")

    assert selected_count == 1, "Exactly one candidate must be marked [ SELECTED ]"

    if mode == "local":
        assert all(c.get("model_id") in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for c in candidates)
        assert not any("gemini" in c.get("model_id").lower() or "mistral" in c.get("model_id").lower() for c in candidates)
    else:
        assert all("gemini" in c.get("model_id").lower() or "mistral" in c.get("model_id").lower() or "llama" in c.get("model_id").lower() for c in candidates)
        assert not any(c.get("model_id") in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for c in candidates)

    print(f"\n=> {test_id} VERIFIED SUCCESSFULLY!")

def main():
    print("Executing End-to-End Live UI Sequence Verification...")
    asyncio.run(run_ui_sequence_test("TEST 1 — LOCAL", "local", "What is the capital of France?"))
    asyncio.run(run_ui_sequence_test("TEST 2 — ONLINE", "online", "What is the capital of France?"))
    asyncio.run(run_ui_sequence_test("TEST 3 — ONLINE CODING", "online", "Write a production-ready FastAPI service with JWT authentication."))
    asyncio.run(run_ui_sequence_test("TEST 4 — ONLINE COMPLEX REASONING", "online", "Analyze the time and space complexity of a distributed consensus algorithm."))
    asyncio.run(run_ui_sequence_test("TEST 5 — SWITCH BACK TO LOCAL", "local", "Explain string immutability in Python."))

if __name__ == "__main__":
    main()
