"""
Unit & Integration Test Suite for RUN 6 RL Experience Collection Pipeline.
Verifies canonical action mapping, selected vs executed action separation,
propensity logging, BGE-M3 masking, fallback action recording, reward attribution,
cost telemetry, experience buffer persistence, and controlled exploration mode.
"""
import os
import json
import time
import pytest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.schemas.experience import ExperienceRecord
from app.schemas.decision import (
    DecisionResponse,
    DecisionTrace,
    CandidateScoreBreakdown
)
from app.schemas.orchestration import OrchestrationResponse
from app.schemas.response import ResponseGenerationResponse, TokenUsage
from app.schemas.verification import VerificationResponse
from app.schemas.reward import RewardComputeResponse, RewardBreakdown, ComponentContribution
from app.services.experience_buffer import (
    ExperienceBufferService,
    ACTION_MAP,
    REVERSE_ACTION_MAP,
    MODEL_ALIASES,
    resolve_canonical_action
)
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine


@pytest.fixture
def tmp_buffer(tmp_path):
    """Fixture providing a clean ExperienceBufferService backed by a temporary file."""
    persistence_file = str(tmp_path / "test_experience_buffer.jsonl")
    buffer_service = ExperienceBufferService(capacity=100, persistence_path=persistence_file)
    buffer_service.clear()
    yield buffer_service
    buffer_service.clear()


def build_mock_orchestration_response(
    selected_model="gemma-3-4b",
    executed_model="gemma-3-4b",
    provider="ollama",
    reward=0.85,
    fallback_used=False,
    fallback_reason=None,
    cost=0.0,
    cost_source="zero_local",
    candidate_probs=None,
    propensity_prob=None
):
    """Helper to construct a complete OrchestrationResponse for unit tests."""
    dec_trace = DecisionTrace(
        decision_id="dec-test-1234",
        intent="factual",
        is_ambiguous=False,
        complexity_level="low",
        complexity_score=0.15,
        resource_summary={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 50.0, "gpu_available": False},
        selected_model=executed_model,
        decision_score=0.80,
        policy="rl_contextual_bandit_policy",
        shadow_rl_decision={
            "production_policy": "rl_contextual_bandit_policy",
            "rl_selected_model": selected_model,
            "executed_model": executed_model,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "production_override": not fallback_used
        },
        candidate_action_probabilities=candidate_probs,
        propensity_probability=propensity_prob,
        decision_latency_ms=12.5
    )

    dec_res = DecisionResponse(
        text="Test prompt for experience collection",
        selected_model=executed_model,
        decision_score=0.80,
        policy="rl_contextual_bandit_policy",
        reasoning=["Test reasoning line"],
        candidates=[],
        decision_trace=dec_trace,
        shadow_rl_decision=dec_trace.shadow_rl_decision,
        total_pipeline_latency_ms=15.0
    )

    gen_res = ResponseGenerationResponse(
        success=True,
        model_id=executed_model,
        provider=provider,
        generated_text="Test generated answer text.",
        finish_reason="STOP",
        latency_ms=150.0,
        usage=TokenUsage(input_tokens=20, output_tokens=30, total_tokens=50),
        cost=cost,
        cost_currency="USD",
        cost_source=cost_source,
        execution_status="completed",
        error_message=None
    )

    ver_res = VerificationResponse(
        verified=True,
        verification_status="verified_baseline",
        prompt="Test prompt for experience collection",
        selected_model=executed_model,
        response_present=True,
        relevance_score=1.0,
        completeness_score=1.0,
        structural_quality_score=1.0,
        factual_verification_status="not_verified",
        issues=[],
        verification_reasoning=["Verification passed."],
        verification_latency_ms=1.0
    )

    dummy_cc = ComponentContribution(score=1.0, weight=0.2, contribution=0.2)
    reward_breakdown = RewardBreakdown(
        quality=dummy_cc, completeness=dummy_cc, relevance=dummy_cc, verification=dummy_cc, execution=dummy_cc
    )

    reward_res = RewardComputeResponse(
        success=True,
        selected_model=executed_model,
        reward=reward,
        reward_breakdown=reward_breakdown,
        reward_status="completed_verified",
        reasoning=["Reward calculated."],
        latency_ms=1.0
    )

    return OrchestrationResponse(
        run_id="run-test-1",
        success=True,
        prompt="Test prompt for experience collection",
        selected_model=selected_model,
        decision_score=0.80,
        decision=dec_res,
        generation=gen_res,
        verification=ver_res,
        reward=reward_res,
        pipeline_latency_ms=166.0
    )


# ---------------------------------------------------------------------------
# Test Case 1: Canonical Action Mapping for All 8 Models
# ---------------------------------------------------------------------------
def test_canonical_action_mapping_all_eight_models():
    expected = {
        "gemma-3-4b": 0,
        "qwen-coder-3b": 1,
        "deepseek-r1-7b": 2,
        "gemini-3.5-flash": 3,
        "mistral-small-latest": 4,
        "llama-3.3-70b-versatile": 5,
        "meta-llama/llama-3.3-70b-instruct": 6,
        "BAAI/bge-m3": 7,
    }
    for model_id, expected_idx in expected.items():
        act_idx, canon_id = resolve_canonical_action(model_id)
        assert act_idx == expected_idx, f"Model '{model_id}' expected action {expected_idx}, got {act_idx}"
        assert canon_id == model_id


# ---------------------------------------------------------------------------
# Test Case 2: Cloud Action Mapping & Alias Resolution
# ---------------------------------------------------------------------------
def test_cloud_action_mapping_and_aliases():
    aliases_to_test = [
        ("gemini-2.0-flash", 3, "gemini-3.5-flash"),
        ("gemini-2.5-flash", 3, "gemini-3.5-flash"),
        ("mistral-small", 4, "mistral-small-latest"),
        ("groq-llama-3.3-70b", 5, "llama-3.3-70b-versatile"),
        ("openrouter-llama-3.3-70b", 6, "meta-llama/llama-3.3-70b-instruct"),
    ]
    for raw_name, expected_act, expected_canon in aliases_to_test:
        act_idx, canon_id = resolve_canonical_action(raw_name)
        assert act_idx == expected_act, f"Alias '{raw_name}' expected action {expected_act}, got {act_idx}"
        assert canon_id == expected_canon, f"Alias '{raw_name}' expected canonical '{expected_canon}', got '{canon_id}'"


# ---------------------------------------------------------------------------
# Test Case 3: Action != -1 for Valid Mapped Executions
# ---------------------------------------------------------------------------
def test_valid_mapped_executions_never_produce_minus_one(tmp_buffer):
    models = ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b", "gemini-3.5-flash", "mistral-small-latest", "llama-3.3-70b-versatile"]
    for m in models:
        orch_res = build_mock_orchestration_response(selected_model=m, executed_model=m)
        rec = tmp_buffer.record_from_orchestration(orch_res)
        assert rec.action >= 0 and rec.action <= 6, f"Valid model '{m}' produced invalid action {rec.action}"
        assert rec.action == ACTION_MAP[m]
        assert rec.is_valid_rl_sample is True


# ---------------------------------------------------------------------------
# Test Case 4: Selected vs Executed Action Separation (Failover Scenario)
# ---------------------------------------------------------------------------
def test_selected_vs_executed_action_separation(tmp_buffer):
    orch_res = build_mock_orchestration_response(
        selected_model="gemini-3.5-flash",
        executed_model="gemma-3-4b",
        provider="ollama",
        fallback_used=True,
        fallback_reason="Google Gemini API rate_limit"
    )

    rec = tmp_buffer.record_from_orchestration(orch_res)

    assert rec.rl_selected_model == "gemini-3.5-flash"
    assert rec.rl_selected_action == 3
    assert rec.executed_model == "gemma-3-4b"
    assert rec.executed_action == 0
    assert rec.action == 0  # Action stored is executed action!
    assert rec.action_model_id == "gemma-3-4b"
    assert rec.metadata["fallback_used"] is True
    assert "rate_limit" in rec.metadata["fallback_reason"]


# ---------------------------------------------------------------------------
# Test Case 5: Propensity Probability Calculation
# ---------------------------------------------------------------------------
def test_propensity_probability_calculation(tmp_buffer):
    candidate_probs = {
        "gemma-3-4b": 0.82857,
        "qwen-coder-3b": 0.02857,
        "deepseek-r1-7b": 0.02857,
        "gemini-3.5-flash": 0.02857,
        "mistral-small-latest": 0.02857,
        "llama-3.3-70b-versatile": 0.02857,
        "meta-llama/llama-3.3-70b-instruct": 0.02857,
        "BAAI/bge-m3": 0.0
    }
    orch_res = build_mock_orchestration_response(
        selected_model="gemma-3-4b",
        executed_model="gemma-3-4b",
        candidate_probs=candidate_probs,
        propensity_prob=0.82857
    )

    rec = tmp_buffer.record_from_orchestration(orch_res)

    assert rec.propensity_probability == pytest.approx(0.82857, abs=1e-4)
    assert rec.candidate_action_probabilities["gemma-3-4b"] == pytest.approx(0.82857, abs=1e-4)
    assert rec.candidate_action_probabilities["BAAI/bge-m3"] == 0.0
    assert rec.propensity_available is True
    assert sum(rec.candidate_action_probabilities.values()) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Test Case 6: BGE-M3 (Action 7) Masking
# ---------------------------------------------------------------------------
def test_bge_m3_action_seven_masking():
    engine = AdaptiveDecisionEngine()
    models = engine.model_registry.list_models()
    bge_m3_meta = next((m for m in models if m.model_id == "BAAI/bge-m3"), None)
    assert bge_m3_meta is not None, "BAAI/bge-m3 must exist in model registry"

    selected_model, _, breakdowns, _ = engine.baseline_policy.evaluate_candidates(
        prompt="Write a Python script",
        intent_info={"intent": "coding"},
        complexity_info={"complexity_score": 0.40},
        resource_info={},
        candidate_models=models
    )

    bge_breakdown = next((c for c in breakdowns if c.model_id == "BAAI/bge-m3"), None)
    assert bge_breakdown is not None
    assert bge_breakdown.eligible is False, "BGE-M3 must be marked ineligible for text generation"
    assert selected_model != "BAAI/bge-m3", "BGE-M3 must never be selected for text generation"


# ---------------------------------------------------------------------------
# Test Case 7: Provider Unavailable Handling
# ---------------------------------------------------------------------------
def test_provider_unavailable_handling():
    engine = AdaptiveDecisionEngine()
    with patch.object(settings, "GEMINI_API_KEY", ""):
        models = engine.model_registry.list_models()
        gemini_meta = next((m for m in models if m.model_id == "gemini-3.5-flash"), None)
        assert gemini_meta is not None
        assert gemini_meta.available is False
        assert gemini_meta.configuration_status == "not_configured"


# ---------------------------------------------------------------------------
# Test Case 8: Fallback Action Recording
# ---------------------------------------------------------------------------
def test_fallback_action_recording(tmp_buffer):
    orch_res = build_mock_orchestration_response(
        selected_model="deepseek-r1-7b",
        executed_model="qwen-coder-3b",
        provider="ollama",
        fallback_used=True,
        fallback_reason="RL selected model 'deepseek-r1-7b' is ineligible or unconfigured."
    )

    rec = tmp_buffer.record_from_orchestration(orch_res)

    assert rec.metadata["fallback_used"] is True
    assert "ineligible or unconfigured" in rec.metadata["fallback_reason"]
    assert rec.rl_selected_model == "deepseek-r1-7b"
    assert rec.executed_model == "qwen-coder-3b"


# ---------------------------------------------------------------------------
# Test Case 9: Reward Attribution to Actual Executed Outcome
# ---------------------------------------------------------------------------
def test_reward_attribution_to_executed_outcome(tmp_buffer):
    orch_res = build_mock_orchestration_response(
        selected_model="gemini-3.5-flash",
        executed_model="gemma-3-4b",
        reward=0.92,
        fallback_used=True
    )

    rec = tmp_buffer.record_from_orchestration(orch_res)

    assert rec.reward == 0.92
    assert rec.action == 0
    assert rec.action_model_id == "gemma-3-4b"


# ---------------------------------------------------------------------------
# Test Case 10: Cost Telemetry Preservation
# ---------------------------------------------------------------------------
def test_cost_telemetry_preservation(tmp_buffer):
    orch_local = build_mock_orchestration_response(executed_model="gemma-3-4b", cost=0.0, cost_source="zero_local")
    rec_local = tmp_buffer.record_from_orchestration(orch_local)
    assert rec_local.metadata["cost"] == 0.0
    assert rec_local.metadata["cost_source"] == "zero_local"

    orch_cloud = build_mock_orchestration_response(
        executed_model="gemini-3.5-flash",
        provider="Google Gemini API",
        cost=0.0025,
        cost_source="configured_pricing_estimate"
    )
    rec_cloud = tmp_buffer.record_from_orchestration(orch_cloud)
    assert rec_cloud.metadata["cost"] == 0.0025
    assert rec_cloud.metadata["cost_source"] == "configured_pricing_estimate"

    metrics = tmp_buffer.get_cost_metrics()
    assert metrics["total_api_cost"] == 0.0025
    assert metrics["local_requests_count"] == 1
    assert metrics["cloud_requests_count"] == 1


# ---------------------------------------------------------------------------
# Test Case 11: Experience Buffer Persistence and Reloading
# ---------------------------------------------------------------------------
def test_experience_buffer_persistence(tmp_path):
    persist_file = str(tmp_path / "persistence_test_buffer.jsonl")

    buf1 = ExperienceBufferService(capacity=100, persistence_path=persist_file)
    buf1.clear()

    orch_res = build_mock_orchestration_response(selected_model="gemma-3-4b", executed_model="gemma-3-4b")
    rec_saved = buf1.record_from_orchestration(orch_res)

    ExperienceBufferService._instance = None
    buf2 = ExperienceBufferService(capacity=100, persistence_path=persist_file)

    assert len(buf2._buffer) == 1
    rec_loaded = buf2._buffer[0]
    assert rec_loaded.experience_id == rec_saved.experience_id
    assert rec_loaded.action == 0
    assert rec_loaded.executed_model == "gemma-3-4b"

    buf2.clear()


# ---------------------------------------------------------------------------
# Test Case 12: Controlled Exploration Mode Toggling via Settings
# ---------------------------------------------------------------------------
def test_controlled_exploration_mode_toggling():
    engine = AdaptiveDecisionEngine()
    from app.schemas.decision import DecisionRequest
    dec_req = DecisionRequest(text="What is the capital of France?", execution_mode="local")

    # 1. Normal mode (RL_DATA_COLLECTION_MODE = False)
    with patch.object(settings, "RL_DATA_COLLECTION_MODE", False):
        res_normal = engine.decide(dec_req)
        assert "_exploration" not in res_normal.policy
        assert res_normal.decision_trace.propensity_probability == 1.0

    # 2. Exploration mode (RL_DATA_COLLECTION_MODE = True)
    with patch.object(settings, "RL_DATA_COLLECTION_MODE", True), \
         patch.object(settings, "EXPLORATION_EPSILON", 0.20):
        res_expl = engine.decide(dec_req)
        assert "_exploration" in res_expl.policy
        trace = res_expl.decision_trace
        assert trace.candidate_action_probabilities is not None
        assert "gemma-3-4b" in trace.candidate_action_probabilities
        assert trace.candidate_action_probabilities["BAAI/bge-m3"] == 0.0
        assert trace.propensity_probability > 0.0
