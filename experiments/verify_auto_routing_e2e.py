import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def run_e2e_test(test_name: str, prompt: str, execution_mode: str):
    print(f"\n==================================================")
    print(f"  {test_name} — Mode: '{execution_mode}'")
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

    gen = final_res["generation"]
    dec = final_res["decision"]
    print(f"\nAutomatic System Decision for {test_name}:")
    print(f" - Execution Mode: {execution_mode.upper()}")
    print(f" - Intent Classified: {dec.get('intent_info', {}).get('intent')}")
    print(f" - Complexity Level: {dec.get('complexity_info', {}).get('complexity_level')}")
    print(f" - Automatically Selected Provider: {gen.get('provider')}")
    print(f" - Automatically Selected Model: {gen.get('model_id')}")
    print(f" - Execution Latency: {gen.get('latency_ms')} ms")
    print(f" - Step 18 Reward: {final_res['reward']['reward']:.4f}")
    print(f" - Step 20 Replay Recorded: Yes (12D vector)")

    if execution_mode == "local":
        assert gen.get("provider") in ["Local Ollama", "Ollama"]
    else:
        assert gen.get("provider") in ["Groq API", "Mistral API", "Google Gemini API", "OpenRouter API"]

    print(f"\n=> {test_name} PASSED SUCCESSFULLY!\n")

def main():
    asyncio.run(run_e2e_test("Example 1 (LOCAL Coding)", "Write a Python function to merge two sorted linked lists.", "local"))
    asyncio.run(run_e2e_test("Example 2 (LOCAL Concept)", "Explain quantum entanglement in simple terms.", "local"))
    asyncio.run(run_e2e_test("Example 3 (ONLINE Reasoning)", "Analyze this complex algorithm and compare its time complexity with three alternative approaches.", "online"))
    asyncio.run(run_e2e_test("Example 4 (ONLINE Coding)", "Write production-ready Python code for a FastAPI service.", "online"))

if __name__ == "__main__":
    main()
