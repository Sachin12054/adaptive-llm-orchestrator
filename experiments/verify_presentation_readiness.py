import os
import sys
import json
import time
from fastapi.testclient import TestClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

client = TestClient(app)

def run_presentation_verification():
    print("=" * 80)
    print(" VERIFYING PRESENTATION READINESS & API RESPONSE FIELDS")
    print("=" * 80)

    # 1. API Endpoints
    r_health = client.get("/api/health")
    print(f"\n1. GET /api/health           : {r_health.status_code}")
    assert r_health.status_code == 200

    r_resource = client.get("/api/resource/snapshot")
    print(f"2. GET /api/resource/snapshot : {r_resource.status_code}")
    assert r_resource.status_code == 200
    res_json = r_resource.json()
    print("   GPU Telemetry Fields:")
    print(f"     - Hardware GPU Name  : {res_json['gpu']['name']}")
    print(f"     - Total VRAM         : {res_json['gpu']['total_vram_gb']} GB")
    print(f"     - Ollama GPU Status  : {res_json['gpu']['ollama_gpu_status']}")
    print(f"     - PyTorch CUDA Status: {res_json['gpu']['pytorch_cuda_status']}")

    r_exp = client.get("/api/experience/status")
    print(f"3. GET /api/experience/status : {r_exp.status_code}")
    assert r_exp.status_code == 200

    r_rl = client.get("/api/rl/status")
    print(f"4. GET /api/rl/status         : {r_rl.status_code}")
    assert r_rl.status_code == 200

    # 2. Simple Request Execution
    pipeline = OrchestrationPipeline()
    simple_prompt = "What is the capital of France?"
    print(f"\n5. Real Simple Request Execution: \"{simple_prompt}\"")
    t0 = time.perf_counter()
    simple_res = pipeline.run_pipeline(OrchestrationRequest(prompt=simple_prompt))
    t1 = time.perf_counter()

    print(f"   Selected Model   : {simple_res.selected_model}")
    print(f"   Complexity Level : {simple_res.decision.complexity_info.complexity_level.upper()}")
    print(f"   Complexity Score : {simple_res.decision.complexity_info.complexity_score:.4f}")
    print(f"   Intent Detected  : {simple_res.decision.intent_info.intent}")
    print(f"   Total Latency    : {simple_res.pipeline_latency_ms} ms")
    assert simple_res.selected_model == "gemma-3-4b"

    # 3. Medium Request Execution ("Explain how binary search works in Python with an example.")
    medium_prompt = "Explain how binary search works in Python with an example."
    print(f"\n6. Real Medium Request Execution: \"{medium_prompt}\"")
    t0_m = time.perf_counter()
    medium_res = pipeline.run_pipeline(OrchestrationRequest(prompt=medium_prompt))
    t1_m = time.perf_counter()

    print(f"   Selected Model   : {medium_res.selected_model}")
    print(f"   Complexity Level : {medium_res.decision.complexity_info.complexity_level.upper()}")
    print(f"   Complexity Score : {medium_res.decision.complexity_info.complexity_score:.4f}")
    print(f"   Intent Detected  : {medium_res.decision.intent_info.intent}")
    print(f"   Total Latency    : {medium_res.pipeline_latency_ms} ms")
    assert medium_res.selected_model == "qwen-coder-3b"
    assert medium_res.decision.policy == "baseline_adaptive_policy"

    print("\n" + "=" * 80)
    print(" ALL PRESENTATION READINESS VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_presentation_verification()
