import pytest
import os
import json
from app.schemas.decision import DecisionRequest
from app.schemas.orchestration import OrchestrationRequest
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.policies.baseline_policy import BaselineAdaptivePolicy
from app.services.providers.online_provider_manager import OnlineProviderManager
from app.services.orchestration_pipeline import OrchestrationPipeline

def test_single_production_routing_authority_invariant():
    """Explicit test proving RLContextualBanditPolicy is the primary production decision authority with BaselineAdaptivePolicy fallback."""
    engine = AdaptiveDecisionEngine()
    
    # 1. Local Mode Decision
    req_local = DecisionRequest(text="Write a Python function to split a string.", execution_mode="local")
    dec_local = engine.decide(req_local)
    
    assert dec_local.policy == "rl_contextual_bandit_policy"
    assert dec_local.selected_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    
    # 2. Explicit Baseline policy instance evaluation
    engine_baseline = AdaptiveDecisionEngine(policy=BaselineAdaptivePolicy())
    dec_baseline = engine_baseline.decide(req_local)
    assert dec_baseline.policy == "baseline_adaptive_policy"

def test_fallback_triggers_policy_reevaluation():
    """Verifies that provider failure triggers candidate exclusion and re-evaluation."""
    engine = AdaptiveDecisionEngine()
    req_1 = DecisionRequest(text="FastAPI microservice code", execution_mode="online")
    dec_1 = engine.decide(req_1)
    
    first_selected = dec_1.selected_model
    assert first_selected is not None
    
    # Simulate exclusion of failed candidate
    req_2 = DecisionRequest(text="FastAPI microservice code", execution_mode="online", excluded_models=[first_selected])
    dec_2 = engine.decide(req_2)
    
    second_selected = dec_2.selected_model
    assert second_selected != first_selected
    assert second_selected not in [first_selected]

def test_shadow_rl_does_not_override_production():
    """Verifies RLContextualBanditPolicy is production decision authority with fallback_used tracking."""
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="Analyze sorting algorithms.", execution_mode="local")
    dec = engine.decide(req)
    
    assert dec.policy == "rl_contextual_bandit_policy"
    assert dec.shadow_rl_decision is not None
    assert dec.shadow_rl_decision.get("fallback_used") is False
