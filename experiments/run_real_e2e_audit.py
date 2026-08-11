import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer

prompts = [
    "What is the capital of Japan?",
    "Explain how TCP congestion control works in simple terms.",
    "Write a Python function that detects whether a string is a palindrome and explain its time complexity.",
    "A company has 5 servers with different processing capacities and 1000 requests arriving per minute. Design a strategy to distribute the requests while minimizing latency and explain your reasoning.",
    "Compare REST APIs, GraphQL, and gRPC. Explain their architecture, advantages, disadvantages, performance characteristics, suitable use cases, and recommend one for a real-time IoT telemetry platform."
]

def format_sec(ms):
    if ms is None: return "N/A"
    sec = ms / 1000.0
    if sec < 60:
        return f"{sec:.2f} sec"
    mins = int(sec // 60)
    rem_sec = sec % 60
    return f"{mins} min {rem_sec:.2f} sec"

async def run_e2e_audit():
    pipeline = OrchestrationPipeline()
    decomposer = ComplexTaskDecomposer()

    audit_results = []

    print("\n" + "="*80)
    print("  COMPLETE E2E AUDIT & VERIFICATION MATRIX (LOCAL & ONLINE)")
    print("="*80 + "\n")

    for mode in ["local", "online"]:
        print(f"\n--- TESTING MODE: {mode.upper()} ---")
        for idx, p in enumerate(prompts, 1):
            print(f"\n[TEST {idx} - {mode.upper()}] Prompt: '{p[:60]}...'")
            
            # Check if complex plan is generated for complex visualizer
            plan = None
            if decomposer.is_complex_prompt(p):
                plan = decomposer.decompose(p, execution_mode=mode)
                print(f" -> Decomposed complex plan: {plan.total_subtasks} subtasks across {len(plan.execution_levels)} levels.")

            req = OrchestrationRequest(prompt=p, execution_mode=mode)
            events = []
            async for raw_evt in pipeline.run_pipeline_stream(req):
                if raw_evt.startswith("data: "):
                    events.append(json.loads(raw_evt[6:].strip()))

            final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
            decision_evt = next((e for e in events if e.get("stage") == "adaptive_decision"), None)

            assert final_evt is not None, f"TEST {idx} ({mode}) MUST emit final_response event!"
            assert decision_evt is not None, f"TEST {idx} ({mode}) MUST emit adaptive_decision event!"

            payload = final_evt["payload"]
            gen = payload["generation"]
            ver = payload["verification"]
            rwd = payload["reward"]
            decision_meta = decision_evt.get("metadata", {})

            sel_model = payload["selected_model"]
            disp_model = gen["model_id"]
            disp_prov = gen["provider"]
            gen_text = gen["generated_text"]
            gen_len = len(gen_text) if gen_text else 0
            lat_sec = format_sec(payload.get("pipeline_latency_ms", gen.get("latency_ms")))
            ver_status = ver.get("verification_status", "passed" if ver.get("verified") else "failed")
            reward_val = rwd.get("reward", 0.0)

            # Strict Model Invariant Check: selected_model == dispatched_model
            assert sel_model == disp_model, f"MODEL CONSISTENCY MISMATCH: Selected '{sel_model}' != Dispatched '{disp_model}'"
            assert gen_text is not None and len(gen_text.strip()) > 0, f"EMPTY OUTPUT DISALLOWED: {sel_model} returned empty response!"
            assert gen["success"] == True, f"EXECUTION FAILED for {sel_model}"

            # Strict Mode Isolation Check
            if mode == "local":
                assert disp_prov == "Local Ollama", f"LOCAL LEAK: Provider '{disp_prov}' in local mode!"
                assert sel_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"], f"NON-LOCAL MODEL in local mode: {sel_model}"
            else:
                assert disp_prov != "Local Ollama", f"ONLINE LEAK: Provider '{disp_prov}' in online mode!"
                assert sel_model in ["gemini-2.5-flash", "mistral-small-latest", "llama-3.3-70b-versatile", "meta-llama/llama-3.3-70b-instruct"], f"NON-ONLINE MODEL in online mode: {sel_model}"

            res_entry = {
                "prompt_num": idx,
                "mode": mode,
                "prompt_snippet": p[:45] + "...",
                "selected_model": sel_model,
                "dispatched_model": disp_model,
                "assigned_model_ui": sel_model,
                "provider": disp_prov,
                "text_len": gen_len,
                "latency": lat_sec,
                "verified": ver["verified"],
                "verification_status": ver_status,
                "reward": f"{reward_val:.4f}",
                "sample_output": gen_text[:80].replace("\n", " ") + "..."
            }
            audit_results.append(res_entry)

            print(f" -> Selected Model:   {sel_model}")
            print(f" -> Dispatched Model: {disp_model}")
            print(f" -> UI Assigned Model:{sel_model}")
            print(f" -> Provider:         {disp_prov}")
            print(f" -> Response Text Len:{gen_len} chars")
            print(f" -> Latency:          {lat_sec}")
            print(f" -> Verification:     {ver_status} (Verified={ver['verified']})")
            print(f" -> Reward:           {reward_val:.4f}")

    print("\n" + "="*80)
    print("  SUMMARY AUDIT TABLE")
    print("="*80)
    print(f"{'#':<3} | {'Mode':<6} | {'Selected / Dispatched Model':<30} | {'Provider':<18} | {'Len':<5} | {'Latency':<10} | {'Reward':<7}")
    print("-" * 90)
    for r in audit_results:
        print(f"{r['prompt_num']:<3} | {r['mode'].upper():<6} | {r['selected_model']:<30} | {r['provider']:<18} | {r['text_len']:<5} | {r['latency']:<10} | {r['reward']:<7}")

    with open("experiments/audit_results.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    print("\n=> ALL 10 E2E PIPELINE RUNS COMPLETED SUCCESSFULLY WITH 100% INVARIANT PASS!")

def main():
    asyncio.run(run_e2e_audit())

if __name__ == "__main__":
    main()
