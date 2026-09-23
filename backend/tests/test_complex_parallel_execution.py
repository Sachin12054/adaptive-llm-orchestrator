import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch

from app.schemas.complex import SubTask, ComplexTaskPlan
from app.services.complex.dynamic_decomposer import DynamicTaskDecomposer, EMBEDDING_MODEL_ID, EMBEDDING_ACTION_INDEX
from app.services.complex.dependency_graph import DependencyGraphBuilder
from app.services.complex.parallel_scheduler import ParallelTaskScheduler
from app.services.complex.task_aggregator import TaskAggregator
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer
from app.schemas.response import ResponseGenerationResponse

def test_simple_prompt_does_not_decompose():
    decomposer = DynamicTaskDecomposer()
    assert not decomposer.is_decomposable_complex("What is the capital of France?", complexity_score=0.1, complexity_level="low")

def test_complex_multi_question_prompt_decomposes():
    decomposer = DynamicTaskDecomposer()
    prompt = "What is the capital of France? What are the best places to visit in France? What are the best hotels to stay at?"
    assert decomposer.is_decomposable_complex(prompt, complexity_score=0.85, complexity_level="very_high")

def test_masking_embedding_action_7():
    decomposer = DynamicTaskDecomposer()

    with patch.object(decomposer.decision_engine, "decide") as mock_decide:
        mock_res = MagicMock()
        mock_res.selected_model = EMBEDDING_MODEL_ID
        mock_res.decision_score = 0.95
        mock_decide.return_value = mock_res

        prod_model, rl_shadow_model, rl_agreed, q_val, provider = decomposer.route_subtask(
            subtask_description="Test subtask",
            category="general",
            execution_mode="local"
        )

        assert prod_model != EMBEDDING_MODEL_ID, "Embedding model (BAAI/bge-m3) must be masked from production response generation"
        assert prod_model == "gemma-3-4b"

def test_cycle_detection_rejects_cycles():
    builder = DependencyGraphBuilder()
    task1 = SubTask(task_id="TASK-1", description="Task 1", category="general", assigned_model="gemma-3-4b", dependencies=["TASK-2"])
    task2 = SubTask(task_id="TASK-2", description="Task 2", category="general", assigned_model="gemma-3-4b", dependencies=["TASK-1"])

    with pytest.raises(ValueError, match="Cycle detected"):
        builder.compute_execution_levels([task1, task2])

def test_dag_topological_levels():
    builder = DependencyGraphBuilder()
    t1 = SubTask(task_id="TASK-1", description="Task 1", category="general", assigned_model="gemma-3-4b", dependencies=[])
    t2 = SubTask(task_id="TASK-2", description="Task 2", category="general", assigned_model="gemma-3-4b", dependencies=[])
    t3 = SubTask(task_id="TASK-3", description="Task 3", category="general", assigned_model="gemma-3-4b", dependencies=["TASK-1", "TASK-2"])
    t4 = SubTask(task_id="TASK-4", description="Task 4", category="general", assigned_model="gemma-3-4b", dependencies=["TASK-3"])

    levels = builder.compute_execution_levels([t1, t2, t3, t4])
    assert len(levels) == 3
    assert set(levels[0]) == {"TASK-1", "TASK-2"}  # Independent tasks in Level 1
    assert levels[1] == ["TASK-3"]
    assert levels[2] == ["TASK-4"]

@pytest.mark.asyncio
async def test_parallel_execution_overlapping():
    scheduler = ParallelTaskScheduler(max_concurrency=4)

    # Mock response_generator with controlled 0.1s delay
    def mock_generate(req):
        time.sleep(0.1)
        return ResponseGenerationResponse(
            success=True,
            model_id=req.selected_model,
            provider="TestProvider",
            generated_text=f"Output for {req.prompt[:30]}",
            latency_ms=100.0,
            execution_status="completed"
        )

    scheduler.response_generator.generate_response = mock_generate

    t1 = SubTask(task_id="TASK-1", description="Task 1", category="general", assigned_model="gemini-3.5-flash", dependencies=[])
    t2 = SubTask(task_id="TASK-2", description="Task 2", category="general", assigned_model="mistral-small-latest", dependencies=[])
    t3 = SubTask(task_id="TASK-3", description="Task 3", category="general", assigned_model="llama-3.3-70b-versatile", dependencies=[])

    execution_levels = [["TASK-1", "TASK-2", "TASK-3"]]

    start_time = time.perf_counter()
    executed_tasks = await scheduler.execute_plan_async(
        subtasks=[t1, t2, t3],
        execution_levels=execution_levels,
        execution_mode="online"
    )
    total_elapsed = time.perf_counter() - start_time

    # 3 tasks running in parallel with 0.1s delay should complete in ~0.15s, NOT 0.30s sequential!
    assert total_elapsed < 0.25, f"Expected parallel execution < 0.25s, but took {total_elapsed:.2f}s"
    assert all(t.execution_success for t in executed_tasks)
    assert all(t.status == "COMPLETED" for t in executed_tasks)

@pytest.mark.asyncio
async def test_partial_completion_and_error_handling():
    scheduler = ParallelTaskScheduler(max_concurrency=4)

    def mock_generate(req):
        if "TASK-2" in req.prompt:
            return ResponseGenerationResponse(
                success=False,
                model_id=req.selected_model,
                provider="TestProvider",
                generated_text=None,
                latency_ms=10.0,
                execution_status="failed",
                error_message="429 Rate Limit"
            )
        return ResponseGenerationResponse(
            success=True,
            model_id=req.selected_model,
            provider="TestProvider",
            generated_text="Success Output",
            latency_ms=10.0,
            execution_status="completed"
        )

    scheduler.response_generator.generate_response = mock_generate

    t1 = SubTask(task_id="TASK-1", description="TASK-1 description", category="general", assigned_model="gemma-3-4b", dependencies=[])
    t2 = SubTask(task_id="TASK-2", description="TASK-2 description", category="general", assigned_model="qwen-coder-3b", dependencies=[])

    executed_tasks = await scheduler.execute_plan_async(
        subtasks=[t1, t2],
        execution_levels=[["TASK-1", "TASK-2"]],
        execution_mode="local"
    )

    t1_res = next(t for t in executed_tasks if t.task_id == "TASK-1")
    t2_res = next(t for t in executed_tasks if t.task_id == "TASK-2")

    assert t1_res.execution_success is True
    assert t2_res.execution_success is False
    assert t2_res.status == "FAILED"
    assert "429 Rate Limit" in t2_res.error_message

def test_task_aggregator_gracefully_handles_partial_failure():
    aggregator = TaskAggregator()
    t1 = SubTask(task_id="TASK-1", description="Understand requirements", category="general", assigned_model="gemma-3-4b", dependencies=[], execution_success=True, generated_text="Requirements analyzed.")
    t2 = SubTask(task_id="TASK-2", description="Build API", category="coding", assigned_model="qwen-coder-3b", dependencies=[], execution_success=False, error_message="Timeout")

    # Mock response_generator for synthesis pass
    aggregator.response_generator.generate_response = MagicMock(return_value=ResponseGenerationResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="Local Ollama",
        generated_text="Synthesized final summary containing valid subtask outputs.",
        latency_ms=20.0,
        execution_status="completed"
    ))

    aggregated_text = aggregator.aggregate_results(
        original_prompt="Build application",
        subtasks=[t1, t2],
        execution_mode="local"
    )

    assert "Synthesized final summary" in aggregated_text or "TASK-1" in aggregated_text

def test_production_safety_invariants():
    from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
    from app.schemas.decision import DecisionRequest

    engine = AdaptiveDecisionEngine()
    status = engine.get_status()
    assert status.policy == "rl_contextual_bandit_policy", "RLContextualBanditPolicy is production decision authority"

    dec_res = engine.decide(DecisionRequest(text="Test safety prompt"))
    assert dec_res.policy == "rl_contextual_bandit_policy"
