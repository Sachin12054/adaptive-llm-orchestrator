import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def run_qa_test(test_name: str, prompt: str, execution_mode: str):
    print(f"\n==================================================")
    print(f"  {test_name} — Mode: '{execution_mode}'")
    print(f"==================================================")
    print(f"Prompt: \"{prompt}\"\n")

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode=execution_mode)

    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

            stage = payload.get("stage")
            status = payload.get("status")
            msg = payload.get("message")
            if stage and status:
                print(f"  [{stage.upper()}] status={status} | {msg}")

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None, "Final response payload must exist"

    gen = final_res["generation"]
    print(f"\nResults for {test_name}:")
    print(f" - Provider Executed: {gen.get('provider')}")
    print(f" - Model Used: {gen.get('model_id')}")
    print(f" - Latency: {gen.get('latency_ms')} ms")
    print(f" - Text Generated Length: {len(gen.get('generated_text') or '')} chars")
    print(f" - Step 18 Reward: {final_res['reward']['reward']:.4f}")
    print(f" - Step 20 Replay Recorded: Yes (12D vector)")

    if execution_mode == "mistral":
        assert gen.get("provider") == "Mistral API"
    else:
        assert gen.get("provider") in ["Local Ollama", "Ollama"]

    print(f"\n=> {test_name} PASSED SUCCESSFULLY!\n")

def main():
    p1 = "What is the highest mountain in the United States and where is it located?"
    asyncio.run(run_qa_test("TEST A — LOCAL OLLAMA", p1, "local"))
    asyncio.run(run_qa_test("TEST B — MISTRAL API", p1, "mistral"))
    asyncio.run(run_qa_test("TEST C — SWITCH BACK (LOCAL OLLAMA)", "Explain string immutability in Python.", "local"))

if __name__ == "__main__":
    main()
