import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.model_registry import ModelRegistry
from app.services.experience_buffer import ACTION_MAP, REVERSE_ACTION_MAP
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

def test_exactly_eight_actions_exist():
    assert len(ACTION_MAP) == 8
    assert len(REVERSE_ACTION_MAP) == 8

def test_every_model_registry_model_has_action_index():
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) == 8
    for m in models:
        assert m.model_id in ACTION_MAP
        idx = ACTION_MAP[m.model_id]
        assert 0 <= idx < 8

def test_every_action_maps_to_single_model():
    for i in range(8):
        assert i in REVERSE_ACTION_MAP
        model_id = REVERSE_ACTION_MAP[i]
        assert ACTION_MAP[model_id] == i

def test_no_duplicate_model_ids_or_missing_models():
    unique_models = set(ACTION_MAP.keys())
    assert len(unique_models) == 8
    unique_indices = set(ACTION_MAP.values())
    assert unique_indices == set(range(8))

def test_weights_matrix_shape_is_8x12():
    policy = RLContextualBanditPolicy()
    assert policy.weights.shape == (8, 12)

def test_bias_vector_shape_is_8():
    policy = RLContextualBanditPolicy()
    assert policy.bias.shape == (8,)

def test_every_predicted_action_maps_to_valid_model():
    policy = RLContextualBanditPolicy()
    registry = ModelRegistry()
    models = registry.list_models()
    
    dummy_state = [0.1] * 12
    shadow_dec = policy.predict_shadow_decision(dummy_state, baseline_selected_model="gemini-3.5-flash", candidate_models=models)
    
    assert shadow_dec.action_index >= 0
    assert shadow_dec.action_index < 8
    assert shadow_dec.proposed_model in ACTION_MAP

def test_rl_prediction_never_returns_invalid_action_index():
    policy = RLContextualBanditPolicy()
    registry = ModelRegistry()
    models = registry.list_models()
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        rand_state = list(rng.uniform(0.0, 1.0, 12))
        dec = policy.predict_shadow_decision(rand_state, baseline_selected_model="gemini-3.5-flash", candidate_models=models)
        if dec.action_index != -1:
            assert 0 <= dec.action_index < 8
            assert dec.proposed_model == REVERSE_ACTION_MAP[dec.action_index]

def test_rl_cannot_override_production():
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="What is quantum mechanics?")
    res = pipeline.run_pipeline(req)
    
    # Invariant: executed model matches selected model and policy is production RL
    assert res.selected_model == res.decision.selected_model
    assert res.decision.policy == "rl_contextual_bandit_policy"
