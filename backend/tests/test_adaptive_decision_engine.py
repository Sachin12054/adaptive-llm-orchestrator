import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.policies.baseline_policy import BaselineAdaptivePolicy
from app.schemas.decision import DecisionRequest
from app.schemas.model import ModelMetadata

client = TestClient(app)

def test_embedding_model_rejection_in_candidates():
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="What is the capital of France?")
    res = engine.decide(req)

    bge_breakdown = next((c for c in res.candidates if c.model_id == "BAAI/bge-m3"), None)
    if bge_breakdown is not None:
        assert bge_breakdown.eligible is False

def test_policy_abstraction_contract():
    policy = BaselineAdaptivePolicy()
    assert policy.policy_name == "baseline_adaptive_policy"

    forbidden_attrs = ["reward_function", "q_table", "policy_network", "train_step"]
    for attr in forbidden_attrs:
        assert not hasattr(policy, attr)

@patch("app.services.model_registry.ModelRegistry.list_models")
def test_capability_matching_and_complexity_influence(mock_list_models):
    gemma_meta = ModelMetadata(
        model_id="gemma-3-4b",
        provider="ollama",
        display_name="Gemma 3 4B",
        model_type="llm",
        capabilities=["general_qa", "explanation", "factual", "conversational"],
        context_length=8192,
        execution_mode="local",
        local=True,
        available=True,
        configuration_status="configured",
        metadata_source="test"
    )
    coder_meta = ModelMetadata(
        model_id="qwen-coder-3b",
        provider="ollama",
        display_name="Qwen Coder 3B",
        model_type="llm",
        capabilities=["coding", "explanation", "general_qa"],
        context_length=8192,
        execution_mode="local",
        local=True,
        available=True,
        configuration_status="configured",
        metadata_source="test"
    )
    deepseek_meta = ModelMetadata(
        model_id="deepseek-r1-7b",
        provider="ollama",
        display_name="DeepSeek R1 7B",
        model_type="llm",
        capabilities=["reasoning", "mathematics", "coding"],
        context_length=8192,
        execution_mode="local",
        local=True,
        available=True,
        configuration_status="configured",
        metadata_source="test"
    )
    mock_list_models.return_value = [gemma_meta, coder_meta, deepseek_meta]

    engine = AdaptiveDecisionEngine(policy=BaselineAdaptivePolicy())

    # Simple factual explanation prompt -> expect gemma-3-4b winning
    res_low = engine.decide(DecisionRequest(text="What is Python?"))
    assert res_low.selected_model == "gemma-3-4b"

    # Coding prompt -> expect qwen-coder-3b winning
    prompt_code = "Write Python code to read a CSV using pandas."
    res_code = engine.decide(DecisionRequest(text=prompt_code))
    assert res_code.selected_model == "qwen-coder-3b"

def test_resource_aware_gpu_vram_fit():
    policy = BaselineAdaptivePolicy()
    
    gemma = ModelMetadata(
        model_id="gemma-3-4b", provider="ollama", display_name="Gemma 3 4B", model_type="llm",
        capabilities=["general_qa", "explanation"], available=True, configuration_status="configured",
        execution_mode="local", local=True, metadata_source="test"
    )
    deepseek = ModelMetadata(
        model_id="deepseek-r1-7b", provider="ollama", display_name="DeepSeek R1 7B", model_type="llm",
        capabilities=["reasoning", "mathematics"], available=True, configuration_status="configured",
        execution_mode="local", local=True, metadata_source="test"
    )

    # Case A: Low free VRAM (3.2 GB) -> DeepSeek (requires 4.5GB) is penalized
    res_low_vram = {
        "gpu_available": True,
        "gpu_free_vram_gb": 3.2,
        "gpu_total_vram_gb": 4.0,
        "memory_utilization_percent": 50.0,
        "cpu_utilization_percent": 20.0
    }

    gemma_res_score = policy._calculate_resource_fit_score(res_low_vram, gemma)
    deepseek_res_score = policy._calculate_resource_fit_score(res_low_vram, deepseek)

    assert gemma_res_score == 1.00
    assert deepseek_res_score == 0.40

    # Case B: High free VRAM (8.0 GB) -> DeepSeek fits comfortably
    res_high_vram = {
        "gpu_available": True,
        "gpu_free_vram_gb": 8.0,
        "gpu_total_vram_gb": 12.0,
        "memory_utilization_percent": 30.0,
        "cpu_utilization_percent": 15.0
    }

    deepseek_high_score = policy._calculate_resource_fit_score(res_high_vram, deepseek)
    assert deepseek_high_score == 1.00

def test_decision_trace_generation():
    engine = AdaptiveDecisionEngine()
    res = engine.decide(DecisionRequest(text="What is 25 multiplied by 4?"))

    assert res.decision_trace is not None
    assert res.decision_trace.decision_id.startswith("dec-")
    assert res.decision_trace.policy == "rl_contextual_bandit_policy"
    assert "cpu_utilization_percent" in res.decision_trace.resource_summary
    assert "gpu_free_vram_gb" in res.decision_trace.resource_summary

def test_no_gemini_api_call_in_decision_engine():
    engine = AdaptiveDecisionEngine()
    assert not hasattr(engine, "gemini_provider")

def test_api_decision_status_endpoint():
    response = client.get("/api/decision/status")
    assert response.status_code == 200
    data = response.json()
    assert data["policy"] == "rl_contextual_bandit_policy"
    assert "registered_models_count" in data
    assert "executable_candidates_count" in data

def test_api_decision_decide_endpoint():
    payload = {"text": "What is the capital of France?"}
    response = client.post("/api/decision/decide", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "selected_model" in data
    assert "decision_score" in data
    assert "reasoning" in data
    assert "candidates" in data
    assert "decision_trace" in data

def test_candidate_score_formula_is_correct():
    from types import SimpleNamespace

    policy = BaselineAdaptivePolicy()

    model = SimpleNamespace(
        model_id="qwen-coder-3b",
        model_type="llm",
        available=True,
        configuration_status="configured",
        capabilities=["coding"],
        display_name="Qwen Coder 3B",
        context_length=8192,
    )

    selected_model, winning_score, breakdowns, reasoning = (
        policy.evaluate_candidates(
            prompt="Write a Python function to reverse a linked list.",
            intent_info={
                "intent": "coding",
                "is_ambiguous": False,
            },
            complexity_info={
                "level": "medium",
                "complexity_score": 0.50,
            },
            resource_info={
                "memory_utilization_percent": 50.0,
                "cpu_utilization_percent": 20.0,
                "gpu_available": True,
                "gpu_free_vram_gb": 3.5
            },
            candidate_models=[model],
        )
    )

    candidate = breakdowns[0]

    expected_score = round(
        0.30 * candidate.capability_score
        + 0.30 * candidate.complexity_fit_score
        + 0.15 * candidate.resource_fit_score
        + 0.15 * candidate.context_fit_score
        + 0.10 * candidate.cost_fit_score,
        4,
    )

    assert selected_model == "qwen-coder-3b"
    assert candidate.eligible is True
    assert candidate.candidate_score == pytest.approx(expected_score)
    assert winning_score == pytest.approx(expected_score)
