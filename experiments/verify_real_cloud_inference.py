import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def verify_online_real_inference():
    print("\n==================================================")
    print("  VERIFYING REAL ONLINE CLOUD API INFERENCE")
    print("==================================================")
    prompt = "What is the highest peak in the United States?"
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode="online")

    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            events.append(json.loads(raw_evt[6:].strip()))

    final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
    assert final_evt is not None, "final_response SSE event MUST exist!"

    payload = final_evt["payload"]
    gen = payload["generation"]
    dec = payload["decision"]
    ver = payload["verification"]
    rwd = payload["reward"]

    print(f"\nREAL ONLINE CLOUD API EXECUTION PAYLOAD:")
    print(f" - Execution Mode: ONLINE")
    print(f" - Selected Model: {payload['selected_model']}")
    print(f" - Provider: {gen['provider']}")
    print(f" - Generation Latency: {gen['latency_ms']} ms")
    print(f" - Verification Status: {ver['verification_status']} (Verified: {ver['verified']})")
    print(f" - Step 18 Reward: {rwd['reward']:.4f}")
    print(f" - Generated Text Snippet: '{gen['generated_text'][:120]}...' (Length: {len(gen['generated_text'] if gen['generated_text'] else '')})")

    assert gen['provider'] != "Unknown", "Provider MUST NOT be 'Unknown'!"
    assert gen['generated_text'] is not None and len(gen['generated_text'].strip()) > 20, "Generated text MUST NOT be empty!"
    assert gen['latency_ms'] > 50.0, "Real API latency must be recorded (>50ms, not 0.5ms fake)!"
    assert ver['verified'] == True, "Verification MUST pass for valid generated response!"
    assert rwd['reward'] > 0.0, "Step 18 reward MUST be computed for successful inference!"

    print("\n=> REAL ONLINE CLOUD API INFERENCE VERIFIED SUCCESSFULLY!")

async def verify_local_real_inference():
    print("\n==================================================")
    print("  VERIFYING REAL LOCAL OLLAMA INFERENCE")
    print("==================================================")
    prompt = "What is the highest peak in the United States?"
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode="local")

    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            events.append(json.loads(raw_evt[6:].strip()))

    final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
    assert final_evt is not None, "final_response SSE event MUST exist!"

    payload = final_evt["payload"]
    gen = payload["generation"]
    rwd = payload["reward"]

    print(f"\nREAL LOCAL OLLAMA EXECUTION PAYLOAD:")
    print(f" - Execution Mode: LOCAL")
    print(f" - Selected Model: {payload['selected_model']}")
    print(f" - Provider: {gen['provider']}")
    print(f" - Generation Latency: {gen['latency_ms']} ms")
    print(f" - Step 18 Reward: {rwd['reward']:.4f}")
    print(f" - Generated Text Snippet: '{gen['generated_text'][:120]}...' (Length: {len(gen['generated_text'] if gen['generated_text'] else '')})")

    assert gen['provider'] == "Local Ollama", "Provider must be 'Local Ollama'"
    assert gen['generated_text'] is not None and len(gen['generated_text'].strip()) > 10, "Generated text MUST NOT be empty!"

    print("\n=> REAL LOCAL OLLAMA INFERENCE VERIFIED SUCCESSFULLY!")

def main():
    asyncio.run(verify_online_real_inference())
    asyncio.run(verify_local_real_inference())

if __name__ == "__main__":
    main()
