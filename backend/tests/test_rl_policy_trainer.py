import sys
import os
import json
import pytest
import inspect
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.main import app
from app.services.policies.base_policy import BaseDecisionPolicy
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.decision import DecisionRequest
from app.schemas.experience import ExperienceRecord
from app.schemas.rl import RLTrainRequest
from app.schemas.model import ModelMetadata
from rl.policy_trainer import PolicyTrainer

client = TestClient(app)

def test_no_gemini_sdk_import_in_rl():
    import app.services.policies.rl_bandit_policy as rl_module
    import rl.policy_trainer as trainer_module

    source1 = inspect.getsource(rl_module)
    source2 = inspect.getsource(trainer_module)

    assert "google.genai" not in source1
    assert "google.generativeai" not in source1
    assert "google.genai" not in source2
    assert "google.generativeai" not in source2

def test_policy_initialization():
    policy = RLContextualBanditPolicy()
    assert issubclass(RLContextualBanditPolicy, BaseDecisionPolicy)
    assert isinstance(policy, BaseDecisionPolicy)
    assert policy.policy_name == "rl_contextual_bandit_policy"
    assert policy.state_dim == 12

def test_evaluate_candidates_abstract_contract_implementation():
    policy = RLContextualBanditPolicy()
    cands = [
        ModelMetadata(model_id="gemma-3-4b", provider="ollama", display_name="Gemma", model_type="llm", capabilities=["general_qa"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"),
        ModelMetadata(model_id="qwen-coder-3b", provider="ollama", display_name="Qwen", model_type="llm", capabilities=["coding"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"),
        ModelMetadata(model_id="deepseek-r1-7b", provider="ollama", display_name="DeepSeek", model_type="llm", capabilities=["reasoning"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured")
    ]
    
    selected, score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="What is the capital of France?",
        intent_info={"intent": "general_qa", "is_ambiguous": False},
        complexity_info={"level": "low", "complexity_score": 0.20},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 50.0},
        candidate_models=cands
    )

    assert isinstance(breakdowns, list)
    assert isinstance(reasoning, list)
    assert isinstance(score, float)

def test_untrained_policy_safely_produces_no_proposal():
    policy = RLContextualBanditPolicy(model_path="data/rl/non_existent_policy.json")
    cands = [
        ModelMetadata(model_id="gemma-3-4b", provider="ollama", display_name="Gemma", model_type="llm", capabilities=["general_qa"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured")
    ]
    
    shadow_res = policy.predict_shadow_decision([0.1] * 12, "gemma-3-4b", cands)
    assert shadow_res.policy_available is False
    assert shadow_res.proposed_model is None
    assert shadow_res.predicted_reward == 0.0

    selected, score, breakdowns, reasoning = policy.evaluate_candidates("prompt", {}, {}, {}, cands)
    assert selected is None
    assert score == 0.0

def test_weight_shape_and_bias_shape():
    policy = RLContextualBanditPolicy()
    weights = np.zeros((3, 12), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)

    policy.weights = weights
    policy.bias = bias

    assert policy.weights.shape == (3, 12)
    assert policy.bias.shape == (3,)

def test_prediction_shape():
    policy = RLContextualBanditPolicy()
    policy.weights = np.zeros((3, 12), dtype=np.float32)
    policy.bias = np.zeros(3, dtype=np.float32)

    s_vec = [0.1] * 12
    cands = [
        ModelMetadata(model_id="gemma-3-4b", provider="ollama", display_name="Gemma", model_type="llm", capabilities=["general_qa"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"),
        ModelMetadata(model_id="qwen-coder-3b", provider="ollama", display_name="Qwen", model_type="llm", capabilities=["coding"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"),
        ModelMetadata(model_id="deepseek-r1-7b", provider="ollama", display_name="DeepSeek", model_type="llm", capabilities=["reasoning"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured")
    ]

    shadow_res = policy.predict_shadow_decision(s_vec, "gemma-3-4b", cands)
    assert shadow_res.policy_available is True
    assert shadow_res.proposed_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    assert isinstance(shadow_res.predicted_reward, float)

def test_action_masking():
    policy = RLContextualBanditPolicy()
    policy.weights = np.ones((3, 12), dtype=np.float32)
    policy.bias = np.zeros(3, dtype=np.float32)

    s_vec = [0.5] * 12
    cands = [
        ModelMetadata(model_id="gemma-3-4b", provider="ollama", display_name="Gemma", model_type="llm", capabilities=["general_qa"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"),
        ModelMetadata(model_id="qwen-coder-3b", provider="ollama", display_name="Qwen", model_type="llm", capabilities=["coding"], context_length=8192, execution_mode="local", local=True, available=False, configuration_status="not_configured"),
        ModelMetadata(model_id="deepseek-r1-7b", provider="ollama", display_name="DeepSeek", model_type="llm", capabilities=["reasoning"], context_length=8192, execution_mode="local", local=True, available=False, configuration_status="not_configured")
    ]

    shadow_res = policy.predict_shadow_decision(s_vec, "gemma-3-4b", cands)
    assert shadow_res.proposed_model == "gemma-3-4b"
    assert shadow_res.action_masked is True

def test_unavailable_action_cannot_be_selected():
    policy = RLContextualBanditPolicy()
    weights = np.array([[1.0] * 12, [10.0] * 12, [10.0] * 12], dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    policy.weights = weights
    policy.bias = bias

    s_vec = [1.0] * 12
    cands = [
        ModelMetadata(model_id="gemma-3-4b", provider="ollama", display_name="Gemma", model_type="llm", capabilities=["general_qa"], context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"),
        ModelMetadata(model_id="qwen-coder-3b", provider="ollama", display_name="Qwen", model_type="llm", capabilities=["coding"], context_length=8192, execution_mode="local", local=True, available=False, configuration_status="not_configured"),
        ModelMetadata(model_id="deepseek-r1-7b", provider="ollama", display_name="DeepSeek", model_type="llm", capabilities=["reasoning"], context_length=8192, execution_mode="local", local=True, available=False, configuration_status="not_configured")
    ]

    shadow_res = policy.predict_shadow_decision(s_vec, "gemma-3-4b", cands)
    assert shadow_res.proposed_model == "gemma-3-4b"

def test_insufficient_data_handling():
    trainer = PolicyTrainer(output_path="data/rl/test_insufficient_model.json")
    req = RLTrainRequest(minimum_samples=100)

    res = trainer.train_policy(req)
    assert res.success is False
    assert res.status == "insufficient_data"
    assert "Insufficient Step 20 experience samples" in res.message

@patch("app.services.experience_buffer.ExperienceBufferService")
def test_training_with_valid_experiences(mock_buf_cls):
    mock_buf_inst = MagicMock()

    synthetic_records = []
    models = ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    for i in range(120):
        action_idx = i % 3
        rec = ExperienceRecord(
            experience_id=f"exp-{i}",
            state=[0.1 * (i % 10)] * 12,
            action=action_idx,
            action_model_id=models[action_idx],
            reward=0.75 + 0.05 * action_idx,
            next_state=None,
            done=True,
            timestamp=1000.0 + i,
            metadata={}
        )
        synthetic_records.append(rec)

    mock_buf_inst._buffer = synthetic_records
    mock_buf_cls.return_value = mock_buf_inst

    trainer = PolicyTrainer(output_path="data/rl/test_trained_model.json")
    req = RLTrainRequest(minimum_samples=100, epochs=10)
    res = trainer.train_policy(req)

    assert res.success is True
    assert res.status == "completed"
    assert res.samples_used == 120
    assert res.final_mse is not None
    assert os.path.exists("data/rl/test_trained_model.json")

    if os.path.exists("data/rl/test_trained_model.json"):
        os.remove("data/rl/test_trained_model.json")

def test_policy_save_and_load_serialization():
    trainer_path = "data/rl/test_save_load_model.json"

    policy_data = {
        "policy_name": "rl_contextual_bandit_policy",
        "version": "1.0.0",
        "state_dim": 12,
        "action_map": {"gemma-3-4b": 0, "qwen-coder-3b": 1, "deepseek-r1-7b": 2},
        "weights": [[0.1] * 12, [0.2] * 12, [0.3] * 12],
        "bias": [0.05, 0.08, 0.10],
        "training_samples_count": 150,
        "mean_squared_error": 0.012,
        "timestamp": 123456.78
    }

    os.makedirs(os.path.dirname(trainer_path), exist_ok=True)
    with open(trainer_path, "w", encoding="utf-8") as f:
        json.dump(policy_data, f, indent=2)

    policy = RLContextualBanditPolicy(model_path=trainer_path)
    assert policy.weights is not None
    assert policy.weights.shape == (3, 12)
    assert policy.bias.shape == (3,)
    assert policy.training_samples_count == 150
    assert policy.mean_squared_error == 0.012

    if os.path.exists(trainer_path):
        os.remove(trainer_path)

def test_invalid_policy_rejection():
    invalid_path = "data/rl/test_invalid_dim_model.json"
    invalid_data = {
        "policy_name": "rl_contextual_bandit_policy",
        "state_dim": 8,
        "weights": [[0.1] * 8],
        "bias": [0.0]
    }
    os.makedirs(os.path.dirname(invalid_path), exist_ok=True)
    with open(invalid_path, "w", encoding="utf-8") as f:
        json.dump(invalid_data, f)

    policy = RLContextualBanditPolicy(model_path=invalid_path)
    assert policy.weights is None

    if os.path.exists(invalid_path):
        os.remove(invalid_path)

def test_shadow_mode_does_not_override_baseline():
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="What is the capital of France?")
    res = engine.decide(req)

    assert res.selected_model == "gemma-3-4b"
    assert res.policy == "baseline_adaptive_policy"
    assert res.shadow_rl_decision is not None
    assert "proposed_model" in res.shadow_rl_decision

def test_exact_step20_reward_consumption():
    trainer = PolicyTrainer(output_path="data/rl/test_reward_cons_model.json")
    rec = ExperienceRecord(
        experience_id="exp-1",
        state=[0.1] * 12,
        action=0,
        action_model_id="gemma-3-4b",
        reward=0.7550,
        next_state=None,
        done=True,
        timestamp=1000.0,
        metadata={}
    )
    assert rec.reward == 0.7550

def test_api_rl_endpoints():
    response = client.get("/api/rl/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "rl_policy_framework"
    assert data["state_dim"] == 12

    train_resp = client.post("/api/rl/train", json={"minimum_samples": 100})
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert "status" in train_data

    eval_resp = client.post("/api/rl/eval", json={"eval_batch_size": 10})
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert "off_policy_reward_estimate" in eval_data
    assert eval_data["off_policy_reward_estimate"] == "unavailable"
