import pytest
import os
import json
import hashlib
import shutil
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.services.policies.baseline_policy import BaselineAdaptivePolicy
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP
from app.schemas.decision import DecisionRequest, DecisionResponse, DecisionTrace
from app.schemas.provider import ProviderGenerationResponse, TokenUsage
from app.schemas.response import ResponseGenerationResponse
from app.schemas.orchestration import OrchestrationResponse
from app.schemas.verification import VerificationResponse
from app.schemas.reward import RewardComputeResponse, RewardBreakdown, ComponentContribution
from app.schemas.complex import ComplexTaskPlan, SubTask

client = TestClient(app)

def test_rl_policy_artifact_loading():
    """Verify RLContextualBanditPolicy loads v2.0.0 weights (K=8, D=12)."""
    policy = RLContextualBanditPolicy()
    assert policy.load_policy() is True
    assert policy.weights is not None
    assert policy.bias is not None
    assert policy.weights.shape == (8, 12)
    assert policy.bias.shape == (8,)
    assert policy.state_dim == 12
    assert policy.action_dim == 8

def test_embedding_model_action_masking():
    """Verify Action 7 (BAAI/bge-m3) is masked and cannot be selected for text generation."""
    policy = RLContextualBanditPolicy()
    policy.load_policy()
    
    class DummyModel:
        def __init__(self, model_id, model_type="llm", available=True, config="configured"):
            self.model_id = model_id
            self.model_type = model_type
            self.available = available
            self.configuration_status = config
            self.provider = "ollama"
            self.display_name = model_id

    candidates = [
        DummyModel("BAAI/bge-m3", model_type="embedding"),
        DummyModel("gemma-3-4b", model_type="llm")
    ]
    
    state_vec = [0.1, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.0, 0.8]
    shadow_dec = policy.predict_shadow_decision(state_vec, None, candidates)
    
    assert shadow_dec.proposed_model != "BAAI/bge-m3"
    assert shadow_dec.action_index != 7

def test_production_rl_decision_engine_routing():
    """Verify AdaptiveDecisionEngine routes production requests via RLContextualBanditPolicy."""
    policy = RLContextualBanditPolicy()
    policy.load_policy()
    engine = AdaptiveDecisionEngine(policy=policy)
    req = DecisionRequest(text="Explain machine learning algorithms", execution_mode="local")
    dec = engine.decide(req)
    
    assert dec.policy == "rl_contextual_bandit_policy"
    assert dec.selected_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    assert dec.shadow_rl_decision is not None
    assert dec.shadow_rl_decision.get("production_policy") == "rl_contextual_bandit_policy"
    assert dec.shadow_rl_decision.get("fallback_used") is False

def test_fallback_gate_when_rl_model_corrupted_or_missing(tmp_path):
    """Verify engine falls back cleanly to BaselineAdaptivePolicy when RL weights are missing/corrupted."""
    default_policy = RLContextualBanditPolicy()
    orig_path = default_policy.model_path

    try:
        corrupted_policy_path = str(tmp_path / "corrupted_rl.json")
        with open(corrupted_policy_path, "w") as f:
            f.write('{"state_dim": 99}')  # Corrupted state dimension
            
        policy = RLContextualBanditPolicy(model_path=corrupted_policy_path)
        policy.load_policy()
        engine = AdaptiveDecisionEngine(policy=policy)
        
        req = DecisionRequest(text="Write a Python script to sort a list", execution_mode="local")
        dec = engine.decide(req)
        
        assert dec.selected_model is not None
        assert dec.policy == "baseline_adaptive_policy"
        assert dec.shadow_rl_decision.get("fallback_used") is True
    finally:
        # Restore default policy path and valid weights
        default_policy.model_path = orig_path
        default_policy.load_policy()


def test_missing_rl_artifact_falls_back_to_baseline(tmp_path):
    RLContextualBanditPolicy._instance = None


@pytest.mark.parametrize(
    "artifact_data",
    [
        {
            "policy_name": "rl_contextual_bandit_policy",
            "version": "2.0.0",
            "state_dim": 12,
            "action_map": ACTION_MAP,
            "weights": [[0.0] * 12] * 7,
            "bias": [0.0] * 7,
        },
        {
            "policy_name": "rl_contextual_bandit_policy",
            "version": "invalid",
            "state_dim": 12,
            "action_map": ACTION_MAP,
            "weights": [[0.0] * 12] * 8,
            "bias": [0.0] * 8,
        },
    ],
    ids=["invalid_dimensions", "invalid_version"],
)
def test_invalid_rl_artifact_falls_back_to_baseline(tmp_path, artifact_data):
    RLContextualBanditPolicy._instance = None
    artifact_path = tmp_path / "invalid_rl.json"
    artifact_path.write_text(json.dumps(artifact_data), encoding="utf-8")

    policy = RLContextualBanditPolicy(model_path=str(artifact_path))
    engine = AdaptiveDecisionEngine(policy=policy)
    decision = engine.decide(DecisionRequest(text="Write a Python script to sort a list", execution_mode="local"))

    assert decision.policy == "baseline_adaptive_policy"
    assert decision.shadow_rl_decision.get("fallback_used") is True
    assert decision.shadow_rl_decision.get("fallback_reason")
    RLContextualBanditPolicy._instance = None
    missing_path = str(tmp_path / "missing_rl.json")
    policy = RLContextualBanditPolicy(model_path=missing_path)
    engine = AdaptiveDecisionEngine(policy=policy)

    dec = engine.decide(DecisionRequest(text="Write a Python script to sort a list", execution_mode="local"))

    assert dec.policy == "baseline_adaptive_policy"
    assert dec.shadow_rl_decision.get("fallback_used") is True
    assert dec.shadow_rl_decision.get("fallback_reason")
    RLContextualBanditPolicy._instance = None

def test_rollback_to_baseline_config():
    """Verify setting PRODUCTION_POLICY='baseline' switches primary decision authority to BaselineAdaptivePolicy."""
    baseline_policy = BaselineAdaptivePolicy()
    engine = AdaptiveDecisionEngine(policy=baseline_policy)
    
    req = DecisionRequest(text="Calculate prime numbers", execution_mode="local")
    dec = engine.decide(req)
    
    assert dec.policy == "baseline_adaptive_policy"
    assert dec.selected_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]

def test_experience_buffer_records_production_rl_telemetry(tmp_path):
    """Verify experience records capture production RL telemetry without executing online model fitting."""
    buf = ExperienceBufferService(persistence_path=str(tmp_path / "telemetry_buffer.jsonl"))
    
    dummy_trace = DecisionTrace(
        decision_id="dec_test",
        intent="CODE_GEN",
        is_ambiguous=False,
        complexity_level="SIMPLE",
        complexity_score=0.2,
        resource_summary={"vram_used_gb": 4.0},
        selected_model="gemma-3-4b",
        decision_score=0.85,
        policy="rl_contextual_bandit_policy",
        execution_mode="local",
        decision_latency_ms=4.0
    )
    
    dummy_decision = DecisionResponse(
        text="Write python code",
        selected_model="gemma-3-4b",
        decision_score=0.85,
        policy="rl_contextual_bandit_policy",
        reasoning=["RL selected gemma-3-4b"],
        candidates=[],
        decision_trace=dummy_trace,
        shadow_rl_decision={
            "production_policy": "rl_contextual_bandit_policy",
            "rl_selected_model": "gemma-3-4b",
            "executed_model": "gemma-3-4b",
            "fallback_used": False,
            "fallback_reason": None
        },
        intent_info={"primary_intent": "CODE_GEN"},
        complexity_info={"assessed_complexity": "SIMPLE"},
        total_pipeline_latency_ms=8.0
    )

    dummy_breakdown = RewardBreakdown(
        quality=ComponentContribution(score=0.9, weight=0.3, contribution=0.27),
        completeness=ComponentContribution(score=0.9, weight=0.2, contribution=0.18),
        relevance=ComponentContribution(score=0.9, weight=0.25, contribution=0.225),
        verification=ComponentContribution(score=1.0, weight=0.15, contribution=0.15),
        execution=ComponentContribution(score=1.0, weight=0.10, contribution=0.10)
    )

    orch_res = OrchestrationResponse(
        run_id="run_rl_prod_1",
        success=True,
        prompt="Write python code",
        selected_model="gemma-3-4b",
        decision_score=0.85,
        decision=dummy_decision,
        generation=ResponseGenerationResponse(
            provider="ollama",
            model_id="gemma-3-4b",
            generated_text="print('test')",
            execution_status="completed",
            cost=0.0,
            cost_source="zero_local",
            usage=TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30),
            latency_ms=100.0,
            success=True
        ),
        verification=VerificationResponse(
            verified=True,
            prompt="Write python code",
            selected_model="gemma-3-4b",
            response_present=True,
            relevance_score=0.9,
            completeness_score=0.9,
            structural_quality_score=0.9,
            verification_reasoning=["passed"],
            verified_model="gemma-3-4b",
            verification_status="PASSED",
            quality_score=0.9,
            verification_latency_ms=10.0,
            checks_passed=["syntax"]
        ),
        reward=RewardComputeResponse(
            success=True,
            selected_model="gemma-3-4b",
            reward=0.9,
            reward_breakdown=dummy_breakdown,
            reward_status="COMPUTED",
            reasoning=["passed"],
            latency_ms=5.0
        ),
        pipeline_latency_ms=115.0
    )

    rec = buf.record_from_orchestration(orch_res)
    assert rec.metadata["production_policy"] == "rl_contextual_bandit_policy"
    assert rec.metadata["rl_selected_model"] == "gemma-3-4b"
    assert rec.metadata["executed_model"] == "gemma-3-4b"
    assert rec.metadata["fallback_used"] is False

def test_rl_status_and_train_api_endpoints(tmp_path):
    """Verify /api/rl/status and /api/rl/train API endpoints work cleanly."""
    policy = RLContextualBanditPolicy()
    policy.load_policy()
    
    # Reload experience buffer from disk to ensure valid sample count
    canonical_buffer_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "rl", "experience_buffer.jsonl")))
    canonical_snapshot = canonical_buffer_path.read_bytes()
    buf = ExperienceBufferService(persistence_path=str(tmp_path / "api_buffer.jsonl"))
    buf._load_from_persistence()

    res_status = client.get("/api/rl/status")
    assert res_status.status_code == 200
    data_s = res_status.json()
    assert data_s["policy_loaded"] is True
    assert data_s["policy_name"] == "rl_contextual_bandit_policy"
    assert data_s["policy_version"] == "2.0.0"

    production_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "rl", "models", "rl_contextual_bandit_policy.json"))
    with open(production_path, "rb") as artifact_file:
        hash_before = hashlib.sha256(artifact_file.read()).hexdigest()

    temp_artifact = tmp_path / "test_trained_policy.json"
    temp_buffer_path = tmp_path / "api_buffer.jsonl"
    shutil.copyfile(canonical_buffer_path, temp_buffer_path)
    with patch("app.api.routes.rl.ExperienceBufferService", lambda: ExperienceBufferService(persistence_path=str(tmp_path / "api_buffer.jsonl"))), \
         patch("app.api.routes.rl.PolicyTrainer", lambda: __import__("rl.policy_trainer", fromlist=["PolicyTrainer"]).PolicyTrainer(output_path=str(temp_artifact), experience_buffer=ExperienceBufferService(persistence_path=str(tmp_path / "api_buffer.jsonl")))):
        res_train = client.post("/api/rl/train", json={"minimum_samples": 5, "epochs": 10, "learning_rate": 0.01})

    assert res_train.status_code == 200
    data_t = res_train.json()
    assert data_t["success"] is True
    assert data_t["status"] == "completed"
    assert data_t["final_mse"] is not None
    assert temp_artifact.exists()
    with open(production_path, "rb") as artifact_file:
        hash_after = hashlib.sha256(artifact_file.read()).hexdigest()
    assert hash_before == hash_after
    assert canonical_buffer_path.read_bytes() == canonical_snapshot
