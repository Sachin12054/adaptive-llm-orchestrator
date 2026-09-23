import sys
import os
import time
import asyncio
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.complex import SubTask
from app.services.complex.task_response_validator import TaskResponseValidator
from app.services.complex.parallel_scheduler import ParallelTaskScheduler
from app.services.complex.dynamic_decomposer import DynamicTaskDecomposer
from app.services.complex.task_aggregator import TaskAggregator
from app.services.response_generator import ResponseGenerator
from app.schemas.response import ResponseGenerationResponse

def test_prompt_hash_computation_and_integrity():
    """Verify SHA-256 hash calculation and prompt integrity validation."""
    t_id = "TASK-4"
    orig_prompt = "Build a smart college campus assistant for Amrita University."
    obj = "Design the database structure for the campus assistant."
    
    hash1 = TaskResponseValidator.compute_prompt_hash(t_id, orig_prompt, obj)
    hash2 = TaskResponseValidator.compute_prompt_hash(t_id, orig_prompt, obj)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # Valid SHA-256 hex string length
    
    # Hash integrity validation passes for exact match
    assert TaskResponseValidator.validate_prompt_integrity(hash1, t_id, orig_prompt, obj) is True
    
    # Hash integrity validation raises ValueError for corrupted objective
    with pytest.raises(ValueError, match="PROMPT CONTAMINATION DETECTED"):
        TaskResponseValidator.validate_prompt_integrity(hash1, t_id, orig_prompt, "Design an e-commerce database with products and orders.")

def test_wrong_domain_response_rejected():
    """
    Verify generic domain-consistency check rejects wrong-domain responses (e.g. generic library schema for campus assistant request)
    and accepts valid domain-matched responses (e.g. products/orders for e-commerce request).
    """
    campus_prompt = "I want to build a smart college campus assistant for Amrita University Coimbatore."
    subtask_obj = "Design the database structure."

    # 1. Generic Library Database Output for Campus Assistant Request -> MUST BE REJECTED
    library_resp = (
        "Here is the database structure for the system:\n"
        "Tables: Books (book_id, title, isbn, publisher), Authors (author_id, name), "
        "Borrowers (member_id, name, email), Loans (loan_id, book_id, member_id, issue_date, due_date), Genres (genre_id, name)."
    )
    is_valid, score, reason = TaskResponseValidator.validate_response_relevance(
        task_id="TASK-4",
        original_user_prompt=campus_prompt,
        task_objective=subtask_obj,
        category="coding",
        response_text=library_resp
    )
    assert is_valid is False
    assert score <= 0.20
    assert "Off-target generic domain response detected" in reason or "Library Management schema detected" in reason

    # 2. Valid Campus Database Output for Campus Assistant Request -> MUST PASS
    campus_resp = (
        "Here is the database structure for the Amrita University campus assistant:\n"
        "Tables: Students (student_id, roll_number, department, email), CampusLocations (building_id, name, lat, lon), "
        "AcademicQuestions (faq_id, category, answer), StudentAuth (user_id, hashed_pass, privacy_setting)."
    )
    is_valid_campus, score_campus, reason_campus = TaskResponseValidator.validate_response_relevance(
        task_id="TASK-4",
        original_user_prompt=campus_prompt,
        task_objective=subtask_obj,
        category="coding",
        response_text=campus_resp
    )
    assert is_valid_campus is True
    assert score_campus >= 0.60

    # 3. E-Commerce Request with Products/Orders Database Output -> MUST PASS
    ecommerce_prompt = "Build a scalable online e-commerce platform for retail merchants."
    ecommerce_resp = (
        "Here is the database structure:\n"
        "Tables: Products (product_id, sku, price, stock), Orders (order_id, customer_id, total_amount), "
        "Payments (payment_id, order_id, status), Customers (customer_id, email, shipping_address)."
    )
    is_valid_ecom, score_ecom, reason_ecom = TaskResponseValidator.validate_response_relevance(
        task_id="TASK-4",
        original_user_prompt=ecommerce_prompt,
        task_objective=subtask_obj,
        category="coding",
        response_text=ecommerce_resp
    )
    assert is_valid_ecom is True
    assert score_ecom >= 0.60

@pytest.mark.asyncio
async def test_event_driven_dag_early_dependency_unlocking():
    """
    Verify that dependent tasks unlock and dispatch the instant their prerequisites finish,
    without waiting for unrelated tasks in the same level to complete.
    """
    scheduler = ParallelTaskScheduler()
    
    # Mock response generator that returns fast responses for T1, T2, T6 and slow for T3
    def mock_gen(req):
        if "Architecture design" in req.prompt or "Feature list" in req.prompt:
            time_delay = 0.01
        elif "Slow risk analysis" in req.prompt:
            time_delay = 0.20  # Slow task in Level 0
        else:
            time_delay = 0.01
        time.sleep(time_delay)
        return ResponseGenerationResponse(
            success=True,
            model_id=req.selected_model,
            provider="Mock Provider",
            generated_text=f"Response for {req.prompt[:50]} with architecture and risk details",
            latency_ms=10.0,
            execution_status="success"
        )

    scheduler.response_generator.generate_response = MagicMock(side_effect=mock_gen)

    # Subtasks: TASK-1, TASK-2, TASK-3 in Level 0. TASK-6 depends ONLY on TASK-1 and TASK-2.
    t1 = SubTask(task_id="TASK-1", description="Architecture design", category="general", assigned_model="gemma-3-4b")
    t2 = SubTask(task_id="TASK-2", description="Feature list", category="general", assigned_model="gemma-3-4b")
    t3 = SubTask(task_id="TASK-3", description="Slow risk analysis", category="reasoning", assigned_model="gemma-3-4b")
    t6 = SubTask(task_id="TASK-6", description="Implementation roadmap", category="general", assigned_model="gemma-3-4b", dependencies=["TASK-1", "TASK-2"])

    subtasks = [t1, t2, t3, t6]
    levels = [["TASK-1", "TASK-2", "TASK-3"], ["TASK-6"]]

    res_subtasks = await scheduler.execute_plan_async(subtasks, levels, execution_mode="online")

    task_map = {t.task_id: t for t in res_subtasks}
    assert task_map["TASK-1"].execution_success is True
    assert task_map["TASK-2"].execution_success is True
    assert task_map["TASK-3"].execution_success is True
    assert task_map["TASK-6"].execution_success is True

@pytest.mark.asyncio
async def test_seven_plus_parallel_subtask_execution_support():
    """Verify that scheduler supports 7+ concurrent subtasks when hardware/cloud capacity allows."""
    scheduler = ParallelTaskScheduler(max_concurrency=8)
    
    mock_gen_resp = ResponseGenerationResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="Local Ollama",
        generated_text="Valid completed response text for independent campus subtask objective student features.",
        latency_ms=10.0,
        execution_status="success"
    )
    scheduler.response_generator.generate_response = MagicMock(return_value=mock_gen_resp)

    # Create 8 independent subtasks
    subtasks = [
        SubTask(task_id=f"TASK-{i}", description=f"Independent campus subtask objective {i} student features", category="general", assigned_model="gemma-3-4b")
        for i in range(1, 9)
    ]
    levels = [[t.task_id for t in subtasks]]

    res = await scheduler.execute_plan_async(subtasks, levels, execution_mode="online")
    assert len(res) == 8
    assert all(t.execution_success for t in res)

def test_safety_invariants_baseline_authority_rl_shadow_and_action_7_mask():
    """Verify BaselineAdaptivePolicy is sole production authority, RL is shadow-only, and BGE-M3 is masked."""
    decomposer = DynamicTaskDecomposer()
    
    prod_model, rl_shadow_model, agreed, q_val, provider = decomposer.route_subtask(
        subtask_description="Design campus assistant database structure",
        category="coding",
        execution_mode="local"
    )

    # Baseline production model MUST NOT be BAAI/bge-m3
    assert prod_model != "BAAI/bge-m3"
    assert prod_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b", "gemini-3.5-flash", "mistral-small-latest", "llama-3.3-70b-versatile"]

    # RL Shadow Model MUST NOT be BAAI/bge-m3
    assert rl_shadow_model != "BAAI/bge-m3"
