import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def verify_runtime_latencies():
    pipeline = OrchestrationPipeline()

    print("\n==================================================")
    print(" 1. TESTING LOCAL MODE RUNTIME LATENCY RENDERING")
    print("==================================================")
    req_local = OrchestrationRequest(prompt="What is the capital of France?", execution_mode="local")
    events_local = []
    async for raw_evt in pipeline.run_pipeline_stream(req_local):
        if raw_evt.startswith("data: "):
            events_local.append(json.loads(raw_evt[6:].strip()))

    final_local = next((e for e in events_local if e.get("stage") == "final_response"), None)
    assert final_local is not None, "Local final response event must exist"
    payload_local = final_local["payload"]
    gen_local = payload_local["generation"]
    pipe_lat_local = payload_local["pipeline_latency_ms"]

    print(f"LOCAL Execution Mode:")
    print(f" - Model: {payload_local['selected_model']}")
    print(f" - Provider: {gen_local['provider']}")
    print(f" - Backend pipeline_latency_ms field: {pipe_lat_local} (Units: ms, preserved!)")
    print(f" - Backend generation latency_ms field: {gen_local['latency_ms']} (Units: ms, preserved!)")

    # Verify SSE log messages do not append raw "ms"
    comp_evt_local = next((e for e in events_local if e.get("stage") in ["local_inference", "online_inference"] and e.get("status") == "completed"), None)
    if comp_evt_local:
        msg = comp_evt_local.get("message", "")
        print(f" - SSE Stage 5 Log Message: '{msg}'")
        assert "sec" in msg or "min" in msg, f"Message must contain human-readable sec/min! Got: '{msg}'"
        assert not msg.endswith("ms."), f"Message must NOT contain raw ms! Got: '{msg}'"
        assert "latency_ms" in comp_evt_local.get("metadata", {}), "SSE metadata MUST retain 'latency_ms' key!"

    print("\n==================================================")
    print(" 2. TESTING ONLINE MODE RUNTIME LATENCY RENDERING")
    print("==================================================")
    req_online = OrchestrationRequest(prompt="What is the highest peak in the United States?", execution_mode="online")
    events_online = []
    async for raw_evt in pipeline.run_pipeline_stream(req_online):
        if raw_evt.startswith("data: "):
            events_online.append(json.loads(raw_evt[6:].strip()))

    final_online = next((e for e in events_online if e.get("stage") == "final_response"), None)
    assert final_online is not None, "Online final response event must exist"
    payload_online = final_online["payload"]
    gen_online = payload_online["generation"]
    pipe_lat_online = payload_online["pipeline_latency_ms"]

    print(f"ONLINE Execution Mode:")
    print(f" - Model: {payload_online['selected_model']}")
    print(f" - Provider: {gen_online['provider']}")
    print(f" - Backend pipeline_latency_ms field: {pipe_lat_online} (Units: ms, preserved!)")
    print(f" - Backend generation latency_ms field: {gen_online['latency_ms']} (Units: ms, preserved!)")

    comp_evt_online = next((e for e in events_online if e.get("stage") in ["local_inference", "online_inference"] and e.get("status") == "completed"), None)
    if comp_evt_online:
        msg = comp_evt_online.get("message", "")
        print(f" - SSE Stage 5 Log Message: '{msg}'")
        assert "sec" in msg or "min" in msg, f"Message must contain human-readable sec/min! Got: '{msg}'"
        assert not msg.endswith("ms."), f"Message must NOT contain raw ms! Got: '{msg}'"
        assert "latency_ms" in comp_evt_online.get("metadata", {}), "SSE metadata MUST retain 'latency_ms' key!"
        assert "latency_seconds" not in comp_evt_online.get("metadata", {}), "SSE metadata MUST NOT rename key to latency_seconds!"

    print("\n=> REAL RUNTIME VERIFICATION PASSED SUCCESSFULLY!")

def main():
    asyncio.run(verify_runtime_latencies())

if __name__ == "__main__":
    main()
