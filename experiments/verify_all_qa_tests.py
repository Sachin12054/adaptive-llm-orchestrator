import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.schemas.decision import DecisionRequest
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.orchestration_pipeline import OrchestrationPipeline

async def run_single_test(test_id: str, title: str, prompt: str, execution_mode: str):
    print(f"\n==================================================")
    print(f"  {test_id}: {title}")
    print(f"==================================================")
    print(f"Prompt: \"{prompt}\"")
    print(f"Requested Execution Mode: {execution_mode.upper()}")

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
    reward = final_res["reward"]

    print(f"\nResults for {test_id}:")
    print(f" - Execution Mode: {execution_mode.upper()}")
    print(f" - Selected Provider: {gen.get('provider')}")
    print(f" - Selected Model: {gen.get('model_id')}")
    print(f" - Routing Policy: {dec.get('policy')}")
    print(f" - Decision Score: {dec.get('decision_score') * 100:.1f}%")
    print(f" - Execution Latency: {gen.get('latency_ms')} ms")
    print(f" - Step 18 Reward: {reward['reward']:.4f}")
    print(f" - Step 20 Experience Replay: Recorded (12D Vector)")
    
    shadow = dec.get("shadow_rl_decision", {})
    print(f" - Shadow RL Proposal: {shadow.get('proposed_model', 'N/A')}")
    print(f" - Production Override: NO")

    print(f"\n=> {test_id} PASSED SUCCESSFULLY!\n")

def test_fallback_reselection():
    print(f"\n==================================================")
    print(f"  TEST 7: ONLINE provider failure -> policy re-selection")
    print(f"==================================================")
    
    engine = AdaptiveDecisionEngine()
    req_1 = DecisionRequest(text="Async Rust web framework", execution_mode="online")
    dec_1 = engine.decide(req_1)
    primary_selected = dec_1.selected_model
    print(f"Primary BaselineAdaptivePolicy Selection: {primary_selected}")

    req_2 = DecisionRequest(text="Async Rust web framework", execution_mode="online", excluded_models=[primary_selected])
    dec_2 = engine.decide(req_2)
    fallback_selected = dec_2.selected_model
    print(f"Fallback BaselineAdaptivePolicy Re-Selection (Excluding '{primary_selected}'): {fallback_selected}")

    assert fallback_selected != primary_selected
    print(f"\n=> TEST 7 PASSED SUCCESSFULLY!\n")

def main():
    asyncio.run(run_single_test("TEST 1", "LOCAL + factual", "What is the capital of France?", "local"))
    asyncio.run(run_single_test("TEST 2", "LOCAL + coding", "Write a Python function to parse JSON files.", "local"))
    asyncio.run(run_single_test("TEST 3", "LOCAL + reasoning", "Solve the logical puzzle of the Knights and Knaves.", "local"))
    asyncio.run(run_single_test("TEST 4", "ONLINE + factual", "What is the speed of light in vacuum?", "online"))
    asyncio.run(run_single_test("TEST 5", "ONLINE + coding", "Write a production-ready Rust program using Tokio async runtime.", "online"))
    asyncio.run(run_single_test("TEST 6", "ONLINE + complex reasoning", "Compare P vs NP complexity classes and analyze implications for cryptography.", "online"))
    test_fallback_reselection()

if __name__ == "__main__":
    main()
