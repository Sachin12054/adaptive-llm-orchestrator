import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.decision import DecisionResponse, CandidateScoreBreakdown, DecisionTrace
from app.schemas.response import ResponseGenerationResponse

def create_mock_decision(selected_model: str) -> DecisionResponse:
    return DecisionResponse(
        text="Test prompt",
        selected_model=selected_model,
        decision_score=0.95,
        policy="baseline_adaptive_policy",
        reasoning=[f"Selected {selected_model}"],
        total_pipeline_latency_ms=10.0,
        candidates=[
            CandidateScoreBreakdown(
                model_id=selected_model,
                provider="Test Provider",
                display_name=selected_model,
                eligible=True,
                candidate_score=0.95
            )
        ],
        decision_trace=DecisionTrace(
            decision_id="test_dec",
            intent="coding",
            is_ambiguous=False,
            complexity_level="medium",
            complexity_score=0.50,
            resource_summary={},
            selected_model=selected_model,
            decision_score=0.95,
            policy="baseline_adaptive_policy",
            decision_latency_ms=10.0,
            shadow_rl_decision={
                "proposed_action": "mistral-small-latest",
                "shadow_action": "mistral-small-latest"
            }
        ),
        shadow_rl_decision={
            "proposed_action": "mistral-small-latest",
            "shadow_action": "mistral-small-latest"
        }
    )

def test_baseline_selected_model_is_actually_executed():
    """Verify that the baseline policy selected model is EXACTLY the model passed to execution."""
    pipeline = OrchestrationPipeline()
    pipeline.decision_engine.decide = MagicMock(return_value=create_mock_decision("gemini-3.5-flash"))
    
    mock_gen_response = ResponseGenerationResponse(
        success=True,
        model_id="gemini-3.5-flash",
        provider="Google Gemini API",
        generated_text="Gemini generated response",
        finish_reason="STOP",
        latency_ms=100.0,
        execution_status="success"
    )
    pipeline.response_generator.generate_response = MagicMock(return_value=mock_gen_response)

    req = OrchestrationRequest(prompt="Test prompt", execution_mode="online")
    res = pipeline.run_pipeline(req)

    # Assert response generator was called with gemini-3.5-flash, NOT mistral-small-latest
    assert pipeline.response_generator.generate_response.called
    call_args = pipeline.response_generator.generate_response.call_args[0][0]
    assert call_args.selected_model == "gemini-3.5-flash"
    assert call_args.selected_model != "mistral-small-latest"

@pytest.mark.parametrize("target_model,expected_provider", [
    ("gemini-3.5-flash", "Google Gemini API"),
    ("mistral-small-latest", "Mistral API"),
    ("llama-3.3-70b-versatile", "Groq API"),
    ("meta-llama/llama-3.3-70b-instruct", "OpenRouter"),
    ("gemma-3-4b", "Local Ollama")
])
def test_all_providers_exact_dispatch(target_model, expected_provider):
    """Verify that Gemini, Mistral, Groq, OpenRouter, and Ollama each execute matching model and provider."""
    pipeline = OrchestrationPipeline()
    pipeline.decision_engine.decide = MagicMock(return_value=create_mock_decision(target_model))
    
    mock_gen_response = ResponseGenerationResponse(
        success=True,
        model_id=target_model,
        provider=expected_provider,
        generated_text=f"Response from {target_model}",
        finish_reason="STOP",
        latency_ms=150.0,
        execution_status="success"
    )
    pipeline.response_generator.generate_response = MagicMock(return_value=mock_gen_response)

    exec_mode = "online" if expected_provider != "Local Ollama" else "local"
    req = OrchestrationRequest(prompt="Explain routing", execution_mode=exec_mode)
    res = pipeline.run_pipeline(req)

    assert res.selected_model == target_model
    assert res.generation.model_id == target_model
    assert res.generation.provider == expected_provider

def test_fallback_when_gemini_fails():
    """Verify explicit fallback logging, re-evaluation, and execution of fallback model when Gemini fails."""
    pipeline = OrchestrationPipeline()
    
    # 1st decision returns gemini-3.5-flash, 2nd decision returns mistral-small-latest
    dec_gemini = create_mock_decision("gemini-3.5-flash")
    dec_mistral = create_mock_decision("mistral-small-latest")
    pipeline.decision_engine.decide = MagicMock(side_effect=[dec_gemini, dec_mistral])

    # 1st generation fails (Gemini error), 2nd generation succeeds (Mistral)
    failed_resp = ResponseGenerationResponse(
        success=False,
        model_id="gemini-3.5-flash",
        provider="Google Gemini API",
        generated_text=None,
        latency_ms=50.0,
        execution_status="failed",
        error_message="Google Gemini API HTTP 429 quota exhausted"
    )
    success_resp = ResponseGenerationResponse(
        success=True,
        model_id="mistral-small-latest",
        provider="Mistral API",
        generated_text="Mistral response text",
        latency_ms=120.0,
        execution_status="success"
    )
    pipeline.response_generator.generate_response = MagicMock(side_effect=[failed_resp, success_resp])

    req = OrchestrationRequest(prompt="Write code", execution_mode="online")
    res = pipeline.run_pipeline(req)

    assert res.selected_model == "mistral-small-latest"
    assert res.generation.model_id == "mistral-small-latest"

    # Verify response generator was called twice (first gemini, then mistral)
    assert pipeline.response_generator.generate_response.call_count == 2
    first_call_model = pipeline.response_generator.generate_response.call_args_list[0][0][0].selected_model
    second_call_model = pipeline.response_generator.generate_response.call_args_list[1][0][0].selected_model
    assert first_call_model == "gemini-3.5-flash"
    assert second_call_model == "mistral-small-latest"

def test_rl_shadow_mode_does_not_override_baseline():
    """Verify that RL proposing a different model in SHADOW mode does NOT override baseline policy selection."""
    pipeline = OrchestrationPipeline()
    
    # Decision Engine selects gemini-3.5-flash, but Shadow RL proposed mistral-small-latest
    dec_resp = create_mock_decision("gemini-3.5-flash")
    dec_resp.shadow_rl_decision = {
        "proposed_action": "mistral-small-latest",
        "predicted_reward": 0.98,
        "shadow_mode": True
    }
    pipeline.decision_engine.decide = MagicMock(return_value=dec_resp)

    mock_gen_response = ResponseGenerationResponse(
        success=True,
        model_id="gemini-3.5-flash",
        provider="Google Gemini API",
        generated_text="Gemini text",
        latency_ms=80.0,
        execution_status="success"
    )
    pipeline.response_generator.generate_response = MagicMock(return_value=mock_gen_response)

    req = OrchestrationRequest(prompt="Test RL shadow", execution_mode="online")
    res = pipeline.run_pipeline(req)

    # Production executed model MUST be gemini-3.5-flash, NOT mistral-small-latest
    assert res.selected_model == "gemini-3.5-flash"
    assert res.generation.model_id == "gemini-3.5-flash"
    assert res.decision.shadow_rl_decision["proposed_action"] == "mistral-small-latest"

def test_rl_shadow_mode_direction_two_mistral_baseline():
    """Verify that when baseline selects Mistral and Shadow RL proposes Gemini, production executes Mistral."""
    pipeline = OrchestrationPipeline()
    
    # Decision Engine selects mistral-small-latest, but Shadow RL proposed gemini-3.5-flash
    dec_resp = create_mock_decision("mistral-small-latest")
    dec_resp.shadow_rl_decision = {
        "proposed_action": "gemini-3.5-flash",
        "predicted_reward": 0.99,
        "shadow_mode": True
    }
    pipeline.decision_engine.decide = MagicMock(return_value=dec_resp)

    mock_gen_response = ResponseGenerationResponse(
        success=True,
        model_id="mistral-small-latest",
        provider="Mistral API",
        generated_text="Mistral text",
        latency_ms=90.0,
        execution_status="success"
    )
    pipeline.response_generator.generate_response = MagicMock(return_value=mock_gen_response)

    req = OrchestrationRequest(prompt="Test RL shadow direction 2", execution_mode="online")
    res = pipeline.run_pipeline(req)

    # Production executed model MUST be mistral-small-latest, NOT gemini-3.5-flash
    assert res.selected_model == "mistral-small-latest"
    assert res.generation.model_id == "mistral-small-latest"
    assert res.decision.shadow_rl_decision["proposed_action"] == "gemini-3.5-flash"
