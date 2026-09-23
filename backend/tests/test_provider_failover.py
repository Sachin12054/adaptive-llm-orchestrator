import sys
import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.complex import SubTask
from app.schemas.response import ResponseGenerationResponse
from app.services.provider_failover import classify_failure, RequestProviderTracker
from app.services.complex.parallel_scheduler import ParallelTaskScheduler
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.decision import DecisionRequest
from app.schemas.provider import TokenUsage

def test_classify_failure():
    # Quota 429
    retry, f_type = classify_failure("Gemini API error 429 RESOURCE_EXHAUSTED")
    assert retry is True
    assert f_type == "quota_exhausted"

    # Server error 503
    retry, f_type = classify_failure("HTTP 503 Service Unavailable")
    assert retry is True
    assert f_type == "server_error"

    # Timeout
    retry, f_type = classify_failure("Connection timed out after 10000ms")
    assert retry is True
    assert f_type == "timeout"

    # Non-retryable
    retry, f_type = classify_failure("Safety refusal: content policy violation")
    assert retry is False
    assert f_type == "non_retryable"

def test_request_provider_tracker():
    tracker = RequestProviderTracker()
    assert tracker.is_provider_exhausted("Google Gemini API") is False

    tracker.mark_provider_exhausted("Google Gemini API", "quota_exhausted")
    assert tracker.is_provider_exhausted("Google Gemini API") is True
    assert tracker.get_exhausted_list() == ["Google Gemini API"]

def test_rl_shadow_mode_invariant():
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="Explain quantum computing", execution_mode="online")
    res = engine.decide(req)

    assert res.policy == "rl_contextual_bandit_policy"
    assert res.shadow_rl_decision is not None
    assert res.shadow_rl_decision.get("fallback_used") is False

@pytest.mark.asyncio
async def test_subtask_failover_gemini_429_to_mistral_success():
    """Test Gemini returns 429, failover dynamically attempts Mistral and succeeds."""
    mock_engine = MagicMock()
    mock_dec_res = MagicMock()
    mock_dec_res.selected_model = "mistral-small-latest"
    mock_engine.decide.return_value = mock_dec_res
    mock_meta_gemini = MagicMock()
    mock_meta_gemini.provider = "Google Gemini API"
    mock_meta_mistral = MagicMock()
    mock_meta_mistral.provider = "Mistral API"
    mock_engine.model_registry.get_model.side_effect = lambda m: mock_meta_gemini if "gemini" in m else mock_meta_mistral

    scheduler = ParallelTaskScheduler(decision_engine=mock_engine)
    tracker = RequestProviderTracker()

    subtask = SubTask(
        task_id="TASK-1",
        description="Explain Paris travel spots",
        category="explanation",
        assigned_model="gemini-3.5-flash",
        provider="Google Gemini API"
    )

    resp_gemini_429 = ResponseGenerationResponse(
        success=False,
        model_id="gemini-3.5-flash",
        provider="Google Gemini API",
        generated_text=None,
        latency_ms=50.0,
        execution_status="failed",
        error_message="Gemini API 429 RESOURCE_EXHAUSTED quota exceeded"
    )
    resp_mistral_ok = ResponseGenerationResponse(
        success=True,
        model_id="mistral-small-latest",
        provider="Mistral API",
        generated_text="Here are top spots in Paris: Eiffel Tower, Louvre.",
        finish_reason="STOP",
        latency_ms=120.0,
        execution_status="completed"
    )

    scheduler.response_generator.generate_response = MagicMock(side_effect=[resp_gemini_429, resp_mistral_ok])

    res = await scheduler._execute_single_subtask(
        subtask=subtask,
        completed_outputs={},
        execution_mode="online",
        provider_tracker=tracker
    )

    assert res.execution_success is True
    assert res.assigned_model == "mistral-small-latest"
    assert res.initial_model == "gemini-3.5-flash"
    assert res.failover_used is True
    assert res.attempts == 2
    assert res.generated_text == "Here are top spots in Paris: Eiffel Tower, Louvre."
    assert tracker.is_provider_exhausted("Google Gemini API") is True

@pytest.mark.asyncio
async def test_subtask_failover_tracks_real_executed_model_cost_and_usage():
    """The final successful provider/model must become the actual executed execution telemetry after failover."""
    mock_engine = MagicMock()
    mock_dec_res = MagicMock()
    mock_dec_res.selected_model = "meta-llama/llama-3.3-70b-instruct"
    mock_engine.decide.return_value = mock_dec_res
    mock_meta_gemini = MagicMock()
    mock_meta_gemini.provider = "Google Gemini API"
    mock_meta_openrouter = MagicMock()
    mock_meta_openrouter.provider = "OpenRouter API"
    mock_engine.model_registry.get_model.side_effect = lambda m: mock_meta_gemini if "gemini" in m else mock_meta_openrouter

    scheduler = ParallelTaskScheduler(decision_engine=mock_engine)
    tracker = RequestProviderTracker()

    subtask = SubTask(
        task_id="TASK-3",
        description="Design a secure backend architecture",
        category="general",
        assigned_model="gemini-3.5-flash",
        provider="Google Gemini API"
    )

    resp_gemini_503 = ResponseGenerationResponse(
        success=False,
        model_id="gemini-3.5-flash",
        provider="Google Gemini API",
        generated_text=None,
        latency_ms=50.0,
        execution_status="failed",
        error_message="Gemini API 503 UNAVAILABLE"
    )
    resp_openrouter_ok = ResponseGenerationResponse(
        success=True,
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider="OpenRouter API",
        generated_text="A secure architecture includes auth, Redis, Postgres, Docker and strict API protections.",
        finish_reason="STOP",
        latency_ms=120.0,
        execution_status="completed",
        usage=TokenUsage(input_tokens=1200, output_tokens=400, total_tokens=1600),
        cost=0.00064,
        cost_currency="USD",
        cost_source="configured_pricing_estimate"
    )

    scheduler.response_generator.generate_response = MagicMock(side_effect=[resp_gemini_503, resp_openrouter_ok])

    res = await scheduler._execute_single_subtask(
        subtask=subtask,
        completed_outputs={},
        execution_mode="online",
        provider_tracker=tracker
    )

    assert res.initial_model == "gemini-3.5-flash"
    assert res.assigned_model == "meta-llama/llama-3.3-70b-instruct"
    assert res.provider == "OpenRouter API"
    assert res.failover_used is True
    assert res.input_tokens == 1200
    assert res.output_tokens == 400
    assert res.total_tokens == 1600
    assert res.cost == 0.00064
    assert res.cost_source == "configured_pricing_estimate"

@pytest.mark.asyncio
async def test_subtask_failover_all_providers_fail():
    """Test genuine failure when all candidate providers fail."""
    mock_engine = MagicMock()
    mock_engine.decide.return_value = MagicMock(selected_model=None)
    mock_engine.model_registry.get_model.return_value = MagicMock(provider="Google Gemini API")

    scheduler = ParallelTaskScheduler(decision_engine=mock_engine)
    tracker = RequestProviderTracker()

    subtask = SubTask(
        task_id="TASK-2",
        description="Calculate orbit dynamics",
        category="mathematics",
        assigned_model="gemini-3.5-flash",
        provider="Google Gemini API"
    )

    resp_fail = ResponseGenerationResponse(
        success=False,
        model_id="gemini-3.5-flash",
        provider="Google Gemini API",
        generated_text=None,
        latency_ms=30.0,
        execution_status="failed",
        error_message="503 Service Unavailable"
    )

    scheduler.response_generator.generate_response = MagicMock(return_value=resp_fail)

    res = await scheduler._execute_single_subtask(
        subtask=subtask,
        completed_outputs={},
        execution_mode="online",
        provider_tracker=tracker
    )

    assert res.execution_success is False
    assert res.status == "FAILED"
    assert "failed" in res.error_message.lower()

@pytest.mark.asyncio
async def test_local_mode_failover_scoped_to_ollama():
    """Test local mode failover strictly uses local Ollama models."""
    mock_engine = MagicMock()
    mock_engine.decide.return_value = MagicMock(selected_model="gemma-3-4b")
    mock_engine.model_registry.get_model.return_value = MagicMock(provider="ollama")

    scheduler = ParallelTaskScheduler(decision_engine=mock_engine)
    tracker = RequestProviderTracker()

    subtask = SubTask(
        task_id="TASK-3",
        description="Write a python sort script",
        category="coding",
        assigned_model="qwen-coder-3b",
        provider="ollama"
    )

    resp_qwen_fail = ResponseGenerationResponse(
        success=False,
        model_id="qwen-coder-3b",
        provider="ollama",
        generated_text=None,
        latency_ms=40.0,
        execution_status="failed",
        error_message="Connection timed out on Ollama"
    )
    resp_gemma_ok = ResponseGenerationResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="ollama",
        generated_text="def sort_list(lst): return sorted(lst)",
        finish_reason="STOP",
        latency_ms=150.0,
        execution_status="completed"
    )

    scheduler.response_generator.generate_response = MagicMock(side_effect=[resp_qwen_fail, resp_gemma_ok])

    res = await scheduler._execute_single_subtask(
        subtask=subtask,
        completed_outputs={},
        execution_mode="local",
        provider_tracker=tracker
    )

    assert res.execution_success is True
    assert res.assigned_model == "gemma-3-4b"
    assert res.provider == "ollama"

@pytest.mark.asyncio
async def test_same_provider_cooldown_across_parallel_tasks():
    """Test that when Task 1 marks Gemini exhausted, Task 2 skips Gemini immediately."""
    mock_engine = MagicMock()
    mock_engine.decide.return_value = MagicMock(selected_model="mistral-small-latest")
    mock_meta_gemini = MagicMock(provider="Google Gemini API")
    mock_meta_mistral = MagicMock(provider="Mistral API")
    mock_engine.model_registry.get_model.side_effect = lambda m: mock_meta_gemini if "gemini" in m else mock_meta_mistral

    scheduler = ParallelTaskScheduler(decision_engine=mock_engine)
    tracker = RequestProviderTracker()

    # Pre-mark Gemini exhausted
    tracker.mark_provider_exhausted("Google Gemini API", "quota_exhausted")

    subtask2 = SubTask(
        task_id="TASK-2",
        description="Recommend hotels in Paris",
        category="explanation",
        assigned_model="gemini-3.5-flash",
        provider="Google Gemini API"
    )

    resp_mistral_ok = ResponseGenerationResponse(
        success=True,
        model_id="mistral-small-latest",
        provider="Mistral API",
        generated_text="Top hotels: Ritz, Le Meurice.",
        finish_reason="STOP",
        latency_ms=110.0,
        execution_status="completed"
    )

    scheduler.response_generator.generate_response = MagicMock(return_value=resp_mistral_ok)

    res = await scheduler._execute_single_subtask(
        subtask=subtask2,
        completed_outputs={},
        execution_mode="online",
        provider_tracker=tracker
    )

    # Verify Gemini was skipped due to cooldown and Mistral executed directly
    assert res.execution_success is True
    assert res.assigned_model == "mistral-small-latest"
    assert scheduler.response_generator.generate_response.call_count == 1
