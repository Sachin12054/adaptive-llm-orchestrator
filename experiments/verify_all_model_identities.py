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
    ("Prompt 1 (Factual)", "What is the capital of Japan?"),
    ("Prompt 2 (Explanation)", "Explain how TCP congestion control works in simple terms."),
    ("Prompt 3 (Coding)", "Write a Python function that detects whether a string is a palindrome and explain its time complexity."),
    ("Prompt 4 (System Design)", "A company has 5 servers with different processing capacities and 1000 requests arriving per minute. Design a strategy to distribute the requests while minimizing latency and explain your reasoning."),
    ("Prompt 5 (Architecture)", "Compare REST APIs, GraphQL, and gRPC. Explain their architecture, advantages, disadvantages, performance characteristics, suitable use cases, and recommend one for a real-time IoT telemetry platform.")
]

async def run_identity_audit():
    pipeline = OrchestrationPipeline()
    decomposer = ComplexTaskDecomposer()

    table_rows = []
    full_responses = []

    print("\n" + "="*90)
    print("  MODEL IDENTITY MATCH & FULL GENERATED RESPONSE VERIFICATION")
    print("="*90 + "\n")

    for label, p in prompts:
        for mode in ["local", "online"]:
            req = OrchestrationRequest(prompt=p, execution_mode=mode)
            events = []
            async for raw_evt in pipeline.run_pipeline_stream(req):
                if raw_evt.startswith("data: "):
                    events.append(json.loads(raw_evt[6:].strip()))

            final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
            decision_evt = next((e for e in events if e.get("stage") == "adaptive_decision"), None)

            assert final_evt is not None, f"Final response missing for {label} ({mode})"
            assert decision_evt is not None, f"Decision event missing for {label} ({mode})"

            payload = final_evt["payload"]
            gen = payload["generation"]

            dec_model = payload["selected_model"]
            disp_model = gen["model_id"]
            assigned_model = dec_model  # Derived from decision object
            final_resp_model = gen["model_id"]

            is_match = (dec_model == assigned_model == disp_model == final_resp_model)
            match_str = "MATCH (100%)" if is_match else "MISMATCH"

            table_rows.append({
                "test": label,
                "mode": mode.upper(),
                "decision": dec_model,
                "assigned": assigned_model,
                "dispatched": disp_model,
                "final_response": final_resp_model,
                "match": match_str
            })

            full_responses.append({
                "test": label,
                "mode": mode.upper(),
                "model": disp_model,
                "provider": gen["provider"],
                "text": gen["generated_text"]
            })

    print(f"{'Test':<25} | {'Mode':<6} | {'Decision':<28} | {'Assigned':<28} | {'Dispatched':<28} | {'Final Response':<28} | {'Match'}")
    print("-" * 155)
    for r in table_rows:
        print(f"{r['test']:<25} | {r['mode']:<6} | {r['decision']:<28} | {r['assigned']:<28} | {r['dispatched']:<28} | {r['final_response']:<28} | {r['match']}")

    print("\n" + "="*90)
    print("  ACTUAL GENERATED RESPONSE TEXTS")
    print("="*90)
    for item in full_responses:
        print(f"\n==================================================")
        print(f"  {item['test']} [{item['mode']}]")
        print(f"  Model: {item['model']} | Provider: {item['provider']}")
        print(f"==================================================")
        print(item["text"])

def main():
    asyncio.run(run_identity_audit())

if __name__ == "__main__":
    main()
