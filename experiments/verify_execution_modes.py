import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def test_mode(mode: str):
    print(f"\n--- Testing Execution Mode: '{mode}' ---")
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="What is Python?", execution_mode=mode)
    
    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

    print(f"Total SSE events received: {len(events)}")
    stages_seen = [e.get("stage") for e in events if e.get("stage")]
    print(f"Stages executed: {set(stages_seen)}")

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None, "Final response payload must be returned"
    
    gen_provider = final_res.get("generation", {}).get("provider")
    print(f"Generation Provider: {gen_provider}")
    print(f"Reward Score: {final_res.get('reward', {}).get('reward')}")
    print(f"Pipeline Latency: {final_res.get('pipeline_latency_ms')} ms")

    if mode == "gemini":
        assert "Gemini" in gen_provider, f"Provider should be Gemini, got {gen_provider}"
    else:
        assert "Ollama" in gen_provider or "Local" in gen_provider, f"Provider should be Ollama, got {gen_provider}"

    print(f"SUCCESS: Execution mode '{mode}' passed E2E SSE pipeline!")

def main():
    asyncio.run(test_mode("local"))
    asyncio.run(test_mode("gemini"))

if __name__ == "__main__":
    main()
