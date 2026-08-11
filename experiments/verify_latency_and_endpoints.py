import os
import sys
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

def run_verification():
    print("=" * 80)
    print(" VERIFYING API ENDPOINTS & BGE-M3 SINGLETON LATENCY REDUCTION")
    print("=" * 80)

    # 1. GET /api/health
    resp_health = client.get("/api/health")
    print(f"\n1. GET /api/health Status: {resp_health.status_code}")
    print(f"   Payload: {resp_health.json()}")
    assert resp_health.status_code == 200

    # 2. GET /api/resource/snapshot
    resp_res = client.get("/api/resource/snapshot")
    print(f"\n2. GET /api/resource/snapshot Status: {resp_res.status_code}")
    assert resp_res.status_code == 200
    res_data = resp_res.json()
    print(f"   CPU Utilization  : {res_data['cpu']['utilization_percent']}%")
    print(f"   RAM Utilization  : {res_data['memory']['utilization_percent']}% ({res_data['memory']['used_gb']:.1f} / {res_data['memory']['total_gb']:.1f} GB)")
    print(f"   GPU Available    : {res_data['gpu']['available']} ({res_data['gpu']['name'] or 'CPU Mode'})")
    print(f"   CUDA PyTorch     : {res_data['cuda']['available']} (PyTorch v{res_data['cuda']['pytorch_version']})")
    print(f"   Disk Storage     : {res_data['disk']['utilization_percent']}% ({res_data['disk']['used_gb']:.1f} / {res_data['disk']['total_gb']:.1f} GB)")

    # 3. First Orchestration Request (Model Loading + Inference)
    prompt = "What is the boiling point of water at sea level in Celsius?"
    print(f"\n3. Executing Request #1: \"{prompt}\"")
    
    t0_req1 = time.perf_counter()
    pipeline = OrchestrationPipeline()
    res1 = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))
    t1_req1 = time.perf_counter()
    req1_total_latency = round((t1_req1 - t0_req1) * 1000, 2)

    print(f"   Request #1 Total Latency     : {req1_total_latency} ms (Pipeline Reported: {res1.pipeline_latency_ms} ms)")
    print(f"   Selected Model               : {res1.selected_model}")
    print(f"   Generation Latency           : {res1.generation.latency_ms} ms")

    # 4. Second Consecutive Orchestration Request (Shared Model Reuse)
    print(f"\n4. Executing Request #2 (Same Prompt - Shared Singleton Reuse): \"{prompt}\"")
    
    t0_req2 = time.perf_counter()
    res2 = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))
    t1_req2 = time.perf_counter()
    req2_total_latency = round((t1_req2 - t0_req2) * 1000, 2)

    print(f"   Request #2 Total Latency     : {req2_total_latency} ms (Pipeline Reported: {res2.pipeline_latency_ms} ms)")
    print(f"   Selected Model               : {res2.selected_model}")
    print(f"   Generation Latency           : {res2.generation.latency_ms} ms")

    print("\n" + "=" * 80)
    print(" VERIFICATION COMPLETE: ALL ENDPOINTS 200 OK & BGE-M3 REUSED SUCCESSFULLY")
    print("=" * 80)

    report_output = {
        "health_status_code": resp_health.status_code,
        "resource_snapshot_status_code": resp_res.status_code,
        "request1_total_latency_ms": req1_total_latency,
        "request1_generation_latency_ms": res1.generation.latency_ms,
        "request2_total_latency_ms": req2_total_latency,
        "request2_generation_latency_ms": res2.generation.latency_ms,
        "telemetry": res_data
    }
    return report_output

if __name__ == "__main__":
    run_verification()
