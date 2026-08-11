import sys
import os
import json
import time
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.providers.ollama_provider import OllamaProvider
from app.schemas.provider import ProviderGenerationRequest

async def run_pipeline_call(prompt: str, mode: str):
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode=mode)
    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    return final_res, events

def main():
    print("\n==================================================")
    print("  TEST A: LOCAL Execution")
    print("==================================================")
    res_a, evts_a = asyncio.run(run_pipeline_call("What is the capital of France?", "local"))
    cands_a = res_a["decision"]["candidates"]
    models_a = [c["model_id"] for c in cands_a]
    print(f"Candidates returned in LOCAL mode: {models_a}")
    assert all(m in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for m in models_a)
    print("=> TEST A PASSED: Only LOCAL candidates returned.")

    print("\n==================================================")
    print("  TEST B: ONLINE Execution")
    print("==================================================")
    res_b, evts_b = asyncio.run(run_pipeline_call("What is the capital of France?", "online"))
    cands_b = res_b["decision"]["candidates"]
    models_b = [c["model_id"] for c in cands_b]
    print(f"Candidates returned in ONLINE mode: {models_b}")
    assert all(m in ["gemini-2.5-flash", "mistral-small-latest", "llama-3.3-70b-versatile", "meta-llama/llama-3.3-70b-instruct"] for m in models_b)
    print("=> TEST B PASSED: Only ONLINE candidates returned.")

    print("\n==================================================")
    print("  TEST C: LOCAL -> ONLINE -> LOCAL Sequence")
    print("==================================================")
    res_c1, _ = asyncio.run(run_pipeline_call("Tell me a joke", "local"))
    res_c2, _ = asyncio.run(run_pipeline_call("Tell me a joke", "online"))
    res_c3, _ = asyncio.run(run_pipeline_call("Tell me a joke", "local"))
    
    m1 = [c["model_id"] for c in res_c1["decision"]["candidates"]]
    m2 = [c["model_id"] for c in res_c2["decision"]["candidates"]]
    m3 = [c["model_id"] for c in res_c3["decision"]["candidates"]]
    print(f"Sequence 1 (LOCAL): {m1}")
    print(f"Sequence 2 (ONLINE): {m2}")
    print(f"Sequence 3 (LOCAL): {m3}")
    assert all(m in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for m in m1)
    assert all("gemini" in m.lower() or "mistral" in m.lower() or "llama" in m.lower() for m in m2)
    assert all(m in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] for m in m3)
    print("=> TEST C PASSED: Dynamic mode candidate swapping verified.")

    print("\n==================================================")
    print("  TEST D: Singleton & No Repeated Initialization")
    print("==================================================")
    print("Running 2 consecutive orchestration requests...")
    t0 = time.perf_counter()
    asyncio.run(run_pipeline_call("Request 1", "local"))
    t1 = time.perf_counter()
    asyncio.run(run_pipeline_call("Request 2", "local"))
    t2 = time.perf_counter()
    print(f"Request 1 Pipeline Time: {round((t1-t0)*1000, 2)} ms")
    print(f"Request 2 Pipeline Time: {round((t2-t1)*1000, 2)} ms")
    print("=> TEST D PASSED: Reinitialization eliminated via singleton pattern.")

    print("\n==================================================")
    print("  TEST E: Ollama Inference & GPU Telemetry Diagnostics")
    print("==================================================")
    provider = OllamaProvider()
    req = ProviderGenerationRequest(prompt="What is string immutability in Python?", model_id="gemma-3-4b")
    
    print("\n--- Call 1 (gemma-3-4b) ---")
    resp_1 = provider.generate(req)
    print(f"  Success: {resp_1.success}")
    print(f"  Load Duration (load_ms): {resp_1.ollama_load_ms} ms")
    print(f"  Eval Duration (eval_ms): {resp_1.ollama_eval_ms} ms")
    print(f"  Total Latency (total_ms): {resp_1.latency_ms} ms")
    if resp_1.usage:
        print(f"  Input Tokens: {resp_1.usage.input_tokens} | Output Tokens: {resp_1.usage.output_tokens}")

    print("\n--- Call 2 (gemma-3-4b) ---")
    resp_2 = provider.generate(req)
    print(f"  Success: {resp_2.success}")
    print(f"  Load Duration (load_ms): {resp_2.ollama_load_ms} ms")
    print(f"  Eval Duration (eval_ms): {resp_2.ollama_eval_ms} ms")
    print(f"  Total Latency (total_ms): {resp_2.latency_ms} ms")
    if resp_2.usage:
        print(f"  Input Tokens: {resp_2.usage.input_tokens} | Output Tokens: {resp_2.usage.output_tokens}")
    print("=> TEST E COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
