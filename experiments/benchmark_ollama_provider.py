import time
import json
import urllib.request
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.providers.ollama_provider import OllamaProvider
from app.schemas.provider import ProviderGenerationRequest

def benchmark():
    prompt = "Write a Python program that reads a CSV dataset using pandas."
    
    print("=" * 80)
    print(" BENCHMARKING OLLAMA DIRECT VS OLLAMAPROVIDER")
    print("=" * 80)

    # 1. Direct Request with keep_alive & optimal params
    payload_direct = {
        "model": "qwen-coder-3b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": 256
        }
    }

    t0 = time.perf_counter()
    req_data = json.dumps(payload_direct).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        res1 = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    direct_latency_ms = round((t1 - t0) * 1000, 2)

    print(f"\n1. Direct Ollama Latency: {direct_latency_ms} ms")
    print(f"   Total Duration      : {res1.get('total_duration', 0) / 1e6:.2f} ms")
    print(f"   Load Duration       : {res1.get('load_duration', 0) / 1e6:.2f} ms")
    print(f"   Prompt Eval Duration: {res1.get('prompt_eval_duration', 0) / 1e6:.2f} ms")
    print(f"   Eval Duration       : {res1.get('eval_duration', 0) / 1e6:.2f} ms ({res1.get('eval_count', 0)} tokens)")
    print(f"   Response Output snippet:\n{res1.get('response', '')[:150]}...")

    # 2. Existing Provider Execution
    provider = OllamaProvider()
    t0_p = time.perf_counter()
    prov_res = provider.generate(ProviderGenerationRequest(
        model_id="qwen-coder-3b",
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=256
    ))
    t1_p = time.perf_counter()
    provider_latency_ms = round((t1_p - t0_p) * 1000, 2)

    print(f"\n2. OllamaProvider Latency: {provider_latency_ms} ms")
    print(f"   Success: {prov_res.success}")
    print(f"   Response Output snippet:\n{str(prov_res.generated_text)[:150]}...")

if __name__ == "__main__":
    benchmark()
