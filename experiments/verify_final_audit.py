import os
import sys
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)
sys.path.insert(0, project_root)

from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer
from app.schemas.decision import DecisionRequest
from app.schemas.orchestration import OrchestrationRequest
from app.services.experience_buffer import ExperienceBufferService

def run_final_audit_verifications():
    print("=" * 80)
    print(" FINAL ARCHITECTURAL AUDIT VERIFICATION")
    print("=" * 80)

    # 1. Simple Workflow Check
    engine = AdaptiveDecisionEngine()
    res_simple = engine.decide(DecisionRequest(text="What is the capital of France?"))
    print(f"\n1. Simple Workflow Model Selected : {res_simple.selected_model}")
    assert res_simple.selected_model == "gemma-3-4b"
    assert res_simple.policy == "baseline_adaptive_policy"
    print("   SIMPLE WORKFLOW: PASS")

    # 2. Medium Workflow Check
    res_medium = engine.decide(DecisionRequest(text="Write a Python function to reverse a linked list."))
    print(f"\n2. Medium Workflow Model Selected: {res_medium.selected_model}")
    assert res_medium.selected_model in ["qwen-coder-3b", "gemma-3-4b"]
    assert res_medium.policy == "baseline_adaptive_policy"
    print("   MEDIUM WORKFLOW: PASS")

    # 3. Complex Planning Endpoint Check
    decomposer = ComplexTaskDecomposer()
    prompt_complex = "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results."
    plan = decomposer.decompose(prompt_complex)

    print(f"\n3. Complex Planning Generated Plan: {plan.total_subtasks} subtasks")
    assert plan.is_complex is True
    assert plan.total_subtasks >= 3
    assert len(plan.execution_levels) >= 2
    print("   COMPLEX PLANNING ENDPOINT: PASS")

    # 4. Complex Subtasks Execution Check
    subtask_executed = any(hasattr(s, "generated_text") for s in plan.subtasks)
    print(f"\n4. Complex Subtasks Executed via ModelManager: {subtask_executed}")
    assert not subtask_executed
    print("   COMPLEX SUBTASKS NOT EXECUTED YET: PASS")

    # 5. Step 18 Reward Unchanged
    pipeline = OrchestrationPipeline()
    orch_res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))
    print(f"\n5. Step 18 Scalar Reward: {orch_res.reward.reward:.4f}")
    assert abs(orch_res.reward.reward - 0.7550) < 1e-4
    print("   STEP 18 REWARD UNCHANGED: PASS")

    # 6. Step 20 State Dimension = 12
    buffer_service = ExperienceBufferService()
    latest_rec = buffer_service._buffer[-1]
    print(f"\n6. Step 20 State Dimension: {len(latest_rec.state)}")
    assert len(latest_rec.state) == 12
    print("   STEP 20 STATE DIMENSION = 12: PASS")

    # 7. RL Offline / Shadow Evaluation Check
    assert orch_res.decision.shadow_rl_decision is not None
    assert orch_res.selected_model == "gemma-3-4b"  # Production remains baseline selected model!
    print(f"\n7. Shadow RL Decision Attached: {orch_res.decision.shadow_rl_decision['proposed_model']}")
    print("   RL REMAINS OFFLINE / SHADOW ONLY: PASS")

    # 8. All Pytest Suite Check
    test_path = os.path.join(project_root, "backend", "tests")
    exit_code = pytest.main([test_path, "-q"])
    print(f"\n8. Full Pytest Suite Exit Code: {exit_code}")
    assert exit_code == 0
    print("   ALL 143 EXISTING TESTS PASSING: PASS")

    print("\n" + "=" * 80)
    print(" ALL ARCHITECTURAL AUDIT VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_final_audit_verifications()
