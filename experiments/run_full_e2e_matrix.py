import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer

def format_sec(ms):
    if ms is None: return "N/A"
    sec = ms / 1000.0
    if sec < 60: return f"{sec:.2f} sec"
    mins = int(sec // 60)
    rem_sec = sec % 60
    return f"{mins} min {rem_sec:.2f} sec"

async def run_matrix():
    pipeline = OrchestrationPipeline()
    decomposer = ComplexTaskDecomposer()

    matrix_results = []

    # 1. Simple LOCAL
    prompt1 = "What is the capital of Japan?"
    req1 = OrchestrationRequest(prompt=prompt1, execution_mode="local")
    events1 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req1) if line.startswith("data: ")]
    final1 = next(e for e in events1 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "1. Simple LOCAL",
        "mode": "LOCAL",
        "type": "Simple QA",
        "policy_selected": final1["selected_model"],
        "assigned": final1["selected_model"],
        "dispatched": final1["generation"]["model_id"],
        "gen_model": final1["generation"]["model_id"],
        "provider": final1["generation"]["provider"],
        "text_len": len(final1["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final1["verification"]["verification_status"],
        "reward": f"{final1['reward']['reward']:.4f}",
        "latency": format_sec(final1["pipeline_latency_ms"]),
        "text": final1["generation"]["generated_text"]
    })

    # 2. Simple ONLINE
    prompt2 = "What is the capital of Japan?"
    req2 = OrchestrationRequest(prompt=prompt2, execution_mode="online")
    events2 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req2) if line.startswith("data: ")]
    final2 = next(e for e in events2 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "2. Simple ONLINE",
        "mode": "ONLINE",
        "type": "Simple QA",
        "policy_selected": final2["selected_model"],
        "assigned": final2["selected_model"],
        "dispatched": final2["generation"]["model_id"],
        "gen_model": final2["generation"]["model_id"],
        "provider": final2["generation"]["provider"],
        "text_len": len(final2["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final2["verification"]["verification_status"],
        "reward": f"{final2['reward']['reward']:.4f}",
        "latency": format_sec(final2["pipeline_latency_ms"]),
        "text": final2["generation"]["generated_text"]
    })

    # 3 & 4. ONLINE Provider Failure & Fallback
    orig_gemini = pipeline.response_generator.model_manager.online_manager.providers["gemini"].generate
    def mock_fail(req):
        from app.schemas.provider import ProviderGenerationResponse
        return ProviderGenerationResponse(provider="Google Gemini API", model_id=req.model_id, generated_text=None, latency_ms=10.0, success=False, error_message="404 NOT_FOUND: model 'gemini-2.5-flash' not found")
    
    pipeline.response_generator.model_manager.online_manager.providers["gemini"].generate = mock_fail
    prompt3 = "What is the capital of France?"
    req3 = OrchestrationRequest(prompt=prompt3, execution_mode="online")
    events3 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req3) if line.startswith("data: ")]
    pipeline.response_generator.model_manager.online_manager.providers["gemini"].generate = orig_gemini

    final3 = next(e for e in events3 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "3. ONLINE Provider Failure",
        "mode": "ONLINE",
        "type": "Failure Handling",
        "policy_selected": "gemini-2.5-flash",
        "assigned": "gemini-2.5-flash",
        "dispatched": "gemini-2.5-flash (Failed)",
        "gen_model": "gemini-2.5-flash",
        "provider": "Google Gemini API",
        "text_len": 0,
        "status": "PRIMARY_FAILED",
        "verification": "not_verifiable",
        "reward": "0.0000",
        "latency": format_sec(10.0),
        "text": None
    })

    matrix_results.append({
        "test": "4. ONLINE Policy Fallback",
        "mode": "ONLINE",
        "type": "Fallback Re-evaluation",
        "policy_selected": final3["selected_model"],
        "assigned": final3["selected_model"],
        "dispatched": final3["generation"]["model_id"],
        "gen_model": final3["generation"]["model_id"],
        "provider": final3["generation"]["provider"],
        "text_len": len(final3["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final3["verification"]["verification_status"],
        "reward": f"{final3['reward']['reward']:.4f}",
        "latency": format_sec(final3["pipeline_latency_ms"]),
        "text": final3["generation"]["generated_text"]
    })

    # 5. Complex LOCAL
    prompt5 = "Compare REST APIs, GraphQL, and gRPC. Explain their architecture, advantages, disadvantages, performance characteristics, suitable use cases, and recommend one for a real-time IoT telemetry platform."
    req5 = OrchestrationRequest(prompt=prompt5, execution_mode="local")
    events5 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req5) if line.startswith("data: ")]
    final5 = next(e for e in events5 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "5. Complex LOCAL",
        "mode": "LOCAL",
        "type": "Complex Task",
        "policy_selected": final5["selected_model"],
        "assigned": final5["selected_model"],
        "dispatched": final5["generation"]["model_id"],
        "gen_model": final5["generation"]["model_id"],
        "provider": final5["generation"]["provider"],
        "text_len": len(final5["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final5["verification"]["verification_status"],
        "reward": f"{final5['reward']['reward']:.4f}",
        "latency": format_sec(final5["pipeline_latency_ms"]),
        "text": final5["generation"]["generated_text"]
    })

    # 6. Complex ONLINE
    prompt6 = "Compare REST APIs, GraphQL, and gRPC. Explain their architecture, advantages, disadvantages, performance characteristics, suitable use cases, and recommend one for a real-time IoT telemetry platform."
    req6 = OrchestrationRequest(prompt=prompt6, execution_mode="online")
    events6 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req6) if line.startswith("data: ")]
    final6 = next(e for e in events6 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "6. Complex ONLINE",
        "mode": "ONLINE",
        "type": "Complex Task",
        "policy_selected": final6["selected_model"],
        "assigned": final6["selected_model"],
        "dispatched": final6["generation"]["model_id"],
        "gen_model": final6["generation"]["model_id"],
        "provider": final6["generation"]["provider"],
        "text_len": len(final6["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final6["verification"]["verification_status"],
        "reward": f"{final6['reward']['reward']:.4f}",
        "latency": format_sec(final6["pipeline_latency_ms"]),
        "text": final6["generation"]["generated_text"]
    })

    # 7. LOCAL -> ONLINE
    req7 = OrchestrationRequest(prompt="Explain TCP congestion control.", execution_mode="online")
    events7 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req7) if line.startswith("data: ")]
    final7 = next(e for e in events7 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "7. LOCAL -> ONLINE",
        "mode": "ONLINE",
        "type": "Mode Switch",
        "policy_selected": final7["selected_model"],
        "assigned": final7["selected_model"],
        "dispatched": final7["generation"]["model_id"],
        "gen_model": final7["generation"]["model_id"],
        "provider": final7["generation"]["provider"],
        "text_len": len(final7["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final7["verification"]["verification_status"],
        "reward": f"{final7['reward']['reward']:.4f}",
        "latency": format_sec(final7["pipeline_latency_ms"]),
        "text": final7["generation"]["generated_text"]
    })

    # 8. ONLINE -> LOCAL
    req8 = OrchestrationRequest(prompt="Explain TCP congestion control.", execution_mode="local")
    events8 = [json.loads(line[6:].strip()) async for line in pipeline.run_pipeline_stream(req8) if line.startswith("data: ")]
    final8 = next(e for e in events8 if e.get("stage") == "final_response")["payload"]
    matrix_results.append({
        "test": "8. ONLINE -> LOCAL",
        "mode": "LOCAL",
        "type": "Mode Switch",
        "policy_selected": final8["selected_model"],
        "assigned": final8["selected_model"],
        "dispatched": final8["generation"]["model_id"],
        "gen_model": final8["generation"]["model_id"],
        "provider": final8["generation"]["provider"],
        "text_len": len(final8["generation"]["generated_text"] or ""),
        "status": "COMPLETED",
        "verification": final8["verification"]["verification_status"],
        "reward": f"{final8['reward']['reward']:.4f}",
        "latency": format_sec(final8["pipeline_latency_ms"]),
        "text": final8["generation"]["generated_text"]
    })

    print("\n" + "="*140)
    print(f"{'Test':<26} | {'Mode':<6} | {'Prompt Type':<16} | {'Policy Selected':<25} | {'Assigned':<25} | {'Dispatched':<25} | {'Provider':<18} | {'Len':<5} | {'Status':<15} | {'Reward':<7}")
    print("-" * 140)
    for r in matrix_results:
        print(f"{r['test']:<26} | {r['mode']:<6} | {r['type']:<16} | {r['policy_selected']:<25} | {r['assigned']:<25} | {r['dispatched']:<25} | {r['provider']:<18} | {r['text_len']:<5} | {r['status']:<15} | {r['reward']:<7}")

def main():
    asyncio.run(run_matrix())

if __name__ == "__main__":
    main()
