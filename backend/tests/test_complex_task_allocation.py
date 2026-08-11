import sys
import os
import pytest
import inspect
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer
from app.services.complex.task_allocator import TaskAllocator
from app.services.complex.dependency_graph import DependencyGraphBuilder
from app.schemas.complex import SubTask, ComplexTaskPlan, ComplexPlanRequest
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.decision import DecisionRequest

client = TestClient(app)

def test_complex_task_detection():
    decomposer = ComplexTaskDecomposer()

    simple_prompt = "What is the capital of France?"
    assert decomposer.is_complex_prompt(simple_prompt, complexity_score=0.20, complexity_level="low") is False

    complex_prompt = "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results."
    assert decomposer.is_complex_prompt(complex_prompt, complexity_score=0.85, complexity_level="very_high") is True

def test_subtask_generation():
    decomposer = ComplexTaskDecomposer()
    prompt = "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results."

    plan = decomposer.decompose(prompt, primary_selected_model="gemini-2.0-flash")
    assert plan.is_complex is True
    assert plan.total_subtasks >= 3
    assert len(plan.subtasks) == plan.total_subtasks

def test_unique_task_ids():
    decomposer = ComplexTaskDecomposer()
    prompt = "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results."

    plan = decomposer.decompose(prompt, primary_selected_model="gemini-2.0-flash")
    task_ids = [t.task_id for t in plan.subtasks]

    assert len(task_ids) == len(set(task_ids)), "Subtask IDs must be unique"
    for i, t_id in enumerate(task_ids, start=1):
        assert t_id == f"TASK-{i}"

def test_dependency_graph_construction():
    builder = DependencyGraphBuilder()
    subtasks = [
        SubTask(task_id="TASK-1", description="Init", category="general", assigned_model="gemini-2.0-flash", dependencies=[]),
        SubTask(task_id="TASK-2", description="Code", category="coding", assigned_model="gemini-2.0-flash", dependencies=["TASK-1"]),
        SubTask(task_id="TASK-3", description="Doc", category="explanation", assigned_model="gemini-2.0-flash", dependencies=["TASK-2"])
    ]

    levels = builder.compute_execution_levels(subtasks)
    assert levels == [["TASK-1"], ["TASK-2"], ["TASK-3"]]

def test_cycle_detection():
    builder = DependencyGraphBuilder()
    subtasks = [
        SubTask(task_id="TASK-1", description="Task 1", category="general", assigned_model="gemini-2.0-flash", dependencies=["TASK-2"]),
        SubTask(task_id="TASK-2", description="Task 2", category="coding", assigned_model="gemini-2.0-flash", dependencies=["TASK-1"])
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        builder.compute_execution_levels(subtasks)

def test_parallel_execution_level_generation():
    decomposer = ComplexTaskDecomposer()
    prompt = "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results."

    plan = decomposer.decompose(prompt, primary_selected_model="gemini-2.0-flash")
    levels = plan.execution_levels

    assert len(levels) >= 2
    assert levels[0] == ["TASK-1"]
    assert "TASK-2" in levels[1] and "TASK-3" in levels[1]

def test_model_allocation_requires_policy_selected_model():
    allocator = TaskAllocator()
    model = allocator.allocate_model("coding", policy_selected_model="gemini-2.0-flash")
    assert model == "gemini-2.0-flash"

def test_model_allocation_missing_policy_model_raises_value_error():
    allocator = TaskAllocator()
    with pytest.raises(ValueError, match="requires authoritative policy_selected_model"):
        allocator.allocate_model("coding", policy_selected_model=None)

def test_existing_simple_workflow_preservation():
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="What is the capital of France?", execution_mode="local")
    res = engine.decide(req)

    assert res.selected_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    assert res.policy == "baseline_adaptive_policy"

def test_existing_medium_workflow_preservation():
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="Write a Python function to reverse a singly linked list.", execution_mode="local")
    res = engine.decide(req)

    assert res.selected_model in ["qwen-coder-3b", "gemma-3-4b", "deepseek-r1-7b"]
    assert res.policy == "baseline_adaptive_policy"

def test_no_complex_subtask_execution():
    decomposer = ComplexTaskDecomposer()
    prompt = "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results."

    plan = decomposer.decompose(prompt, primary_selected_model="gemini-2.0-flash")
    assert isinstance(plan, ComplexTaskPlan)
    assert plan.plan_latency_ms > 0.0
    for subtask in plan.subtasks:
        assert not hasattr(subtask, "generated_text")
        assert not hasattr(subtask, "execution_status")

def test_api_complex_plan_endpoint():
    payload = {
        "prompt": "Build a Python web scraper that collects product prices, cleans the data, calculates statistics, and explains the results.",
        "primary_selected_model": "gemini-2.0-flash"
    }
    response = client.post("/api/complex/plan", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["is_complex"] is True
    assert "subtasks" in data
    assert "execution_levels" in data
    assert data["total_subtasks"] >= 3

    models_assigned = [s["assigned_model"] for s in data["subtasks"]]
    for m in models_assigned:
        assert m == "gemini-2.0-flash"
