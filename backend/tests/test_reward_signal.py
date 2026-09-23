import sys
import os
import pytest
import inspect
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.reward_signal import RewardSignal
from app.schemas.reward import RewardComputeRequest

client = TestClient(app)

def test_no_rl_training_or_model_selection_logic():
    service = RewardSignal()
    prohibited_methods = [
        "select_model", "choose_model", "rank_models", "best_model",
        "fallback_model", "route_request", "score_models",
        "generate_response", "regenerate", "retry_with_fallback",
        "train", "fit", "learn", "update_policy"
    ]
    for method in prohibited_methods:
        assert not hasattr(service, method)

def test_no_direct_gemini_sdk_import():
    import app.services.reward_signal as reward_module
    source_code = inspect.getsource(reward_module)
    assert "google.genai" not in source_code
    assert "google.generativeai" not in source_code

def test_reward_formula_is_correct():
    """
    Standard example:
    0.30 * 1.0  = 0.30
    0.20 * 0.4  = 0.08
    0.25 * 0.5  = 0.125
    0.15 * 1.0  = 0.15
    0.10 * 1.0  = 0.10
    Total = 0.7550
    """
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        winning_score=0.82,
        execution_success=True,
        execution_status="completed",
        generated_text="Paris is the capital of France.",
        verification_status="VERIFIED_BASELINE",
        verified=True,
        response_present=True,
        structural_quality_score=1.0,
        completeness_score=0.4,
        relevance_score=0.5,
        factual_verification_status="not_verified"
    )

    res = service.compute_reward(req)

    assert res.success is True
    assert res.reward == pytest.approx(0.7550)
    assert res.reward_status == "completed_verified"
    assert res.reward_breakdown.quality.contribution == pytest.approx(0.30)
    assert res.reward_breakdown.completeness.contribution == pytest.approx(0.08)
    assert res.reward_breakdown.relevance.contribution == pytest.approx(0.125)
    assert res.reward_breakdown.verification.contribution == pytest.approx(0.15)
    assert res.reward_breakdown.execution.contribution == pytest.approx(0.10)

def test_reward_is_clamped_to_zero_one():
    service = RewardSignal()
    # High score inputs
    req_max = RewardComputeRequest(
        prompt="Test prompt",
        selected_model="gemini-3.5-flash",
        execution_success=True,
        execution_status="completed",
        generated_text="Valid long response text for testing clamping",
        verification_status="VERIFIED_BASELINE",
        verified=True,
        response_present=True,
        structural_quality_score=1.0,
        completeness_score=1.0,
        relevance_score=1.0,
        factual_verification_status="not_verified"
    )
    res_max = service.compute_reward(req_max)
    assert res_max.reward <= 1.0000

    # Low score inputs
    req_min = RewardComputeRequest(
        prompt="Test prompt",
        selected_model="gemini-3.5-flash",
        execution_success=False,
        execution_status="failed",
        generated_text=None,
        verification_status="NOT_VERIFIABLE",
        verified=False,
        response_present=False,
        structural_quality_score=0.0,
        completeness_score=0.0,
        relevance_score=0.0,
        factual_verification_status="not_verified"
    )
    res_min = service.compute_reward(req_min)
    assert res_min.reward >= 0.0000

def test_successful_verified_response():
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="Explain quantum entanglement.",
        selected_model="gemini-2.5-pro",
        execution_success=True,
        execution_status="completed",
        generated_text="Quantum entanglement is a physical phenomenon...",
        verification_status="VERIFIED_BASELINE",
        verified=True,
        response_present=True,
        structural_quality_score=1.0,
        completeness_score=0.8,
        relevance_score=0.9,
        factual_verification_status="not_verified"
    )
    res = service.compute_reward(req)
    assert res.success is True
    assert res.reward > 0.80
    assert res.reward_status == "completed_verified"

def test_failed_execution():
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        execution_success=False,
        execution_status="failed",
        generated_text=None,
        verification_status="NOT_VERIFIABLE",
        verified=False,
        response_present=False,
        structural_quality_score=0.0,
        completeness_score=0.0,
        relevance_score=0.0,
        factual_verification_status="not_verified"
    )
    res = service.compute_reward(req)
    assert res.reward == 0.0000
    assert res.reward_status == "failed_execution"
    assert res.reward_breakdown.execution.contribution == 0.0

def test_not_configured_execution():
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        execution_success=False,
        execution_status="not_configured",
        generated_text=None,
        verification_status="NOT_VERIFIABLE",
        verified=False,
        response_present=False,
        structural_quality_score=0.0,
        completeness_score=0.0,
        relevance_score=0.0,
        factual_verification_status="not_verified"
    )
    res = service.compute_reward(req)
    assert res.reward == 0.0000
    assert res.reward_status == "failed_execution"

def test_not_verifiable_response():
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        execution_success=True,
        execution_status="completed",
        generated_text="   ",
        verification_status="EMPTY_RESPONSE",
        verified=False,
        response_present=False,
        structural_quality_score=0.0,
        completeness_score=0.0,
        relevance_score=0.0,
        factual_verification_status="not_verified"
    )
    res = service.compute_reward(req)
    assert res.reward == 0.1000  # Only execution component credited
    assert res.reward_breakdown.execution.contribution == 0.10

def test_factual_verification_not_assumed():
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        execution_success=True,
        execution_status="completed",
        generated_text="Paris",
        verification_status="VERIFIED_BASELINE",
        verified=True,
        response_present=True,
        structural_quality_score=1.0,
        completeness_score=0.5,
        relevance_score=0.5,
        factual_verification_status="not_verified"
    )
    res = service.compute_reward(req)
    assert res.factual_verification_status == "not_verified"
    assert any("Factual correctness was not established" in r for r in res.reasoning)

def test_empty_prompt_rejection():
    service = RewardSignal()
    req = RewardComputeRequest(
        prompt="   ",
        selected_model="gemini-3.5-flash",
        execution_success=True,
        execution_status="completed",
        generated_text="Paris",
        verification_status="VERIFIED_BASELINE",
        verified=True,
        response_present=True,
        structural_quality_score=1.0,
        completeness_score=0.5,
        relevance_score=0.5,
        factual_verification_status="not_verified"
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        service.compute_reward(req)

def test_api_reward_status_endpoint():
    response = client.get("/api/reward/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "reward_signal"
    assert data["status"] == "ready"
    assert data["learning_active"] is False

def test_api_reward_compute_endpoint():
    payload = {
        "prompt": "What is the capital of France?",
        "selected_model": "gemini-3.5-flash",
        "winning_score": 0.82,
        "execution_success": True,
        "execution_status": "completed",
        "generated_text": "Paris is the capital of France.",
        "verification_status": "VERIFIED_BASELINE",
        "verified": True,
        "response_present": True,
        "structural_quality_score": 1.0,
        "completeness_score": 0.4,
        "relevance_score": 0.5,
        "factual_verification_status": "not_verified"
    }
    response = client.post("/api/reward/compute", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["reward"] == pytest.approx(0.7550)
    assert data["selected_model"] == "gemini-3.5-flash"
    assert "reward_breakdown" in data
    assert data["reward_breakdown"]["quality"]["contribution"] == pytest.approx(0.30)
    assert data["reward_breakdown"]["completeness"]["contribution"] == pytest.approx(0.08)
    assert data["reward_breakdown"]["relevance"]["contribution"] == pytest.approx(0.125)
    assert data["reward_breakdown"]["verification"]["contribution"] == pytest.approx(0.15)
    assert data["reward_breakdown"]["execution"]["contribution"] == pytest.approx(0.10)
