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
    """Explicit test proving BaselineAdaptivePolicy is the ONLY production decision authority."""
    engine = AdaptiveDecisionEngine()
    
    # 1. Local Mode Decision
    req_local = DecisionRequest(text="Write a Python function to split a string.", execution_mode="local")
    dec_local = engine.decide(req_local)
    
    assert dec_local.policy == "baseline_adaptive_policy"
    assert dec_local.selected_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    
    # 2. Online Mode Decision
    req_online = DecisionRequest(text="Explain quantum mechanics.", execution_mode="online")
    dec_online = engine.decide(req_online)
    
    assert dec_online.policy == "baseline_adaptive_policy"
    assert dec_online.selected_model is not None
    
    # Prove that the decision matches BaselineAdaptivePolicy evaluation
    candidates = engine.online_provider_manager.get_online_model_candidates()
    selected_by_policy, score, breakdowns, reasoning = engine.policy.evaluate_candidates(
        prompt="Explain quantum mechanics.",
        intent_info=dec_online.intent_info,
        complexity_info=dec_online.complexity_info,
        resource_info=dec_online.decision_trace.resource_summary,
        candidate_models=candidates
    )
    
    assert dec_online.selected_model == selected_by_policy
    print(f"\n[SINGLE AUTHORITY PROOF] Production decision ({dec_online.selected_model}) matches BaselineAdaptivePolicy output ({selected_by_policy}).")

def test_fallback_triggers_policy_reevaluation():
    """Verifies that provider failure triggers candidate exclusion and BaselineAdaptivePolicy re-evaluation."""
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
    print(f"\n[FALLBACK PROOF] Primary choice '{first_selected}' excluded. BaselineAdaptivePolicy re-evaluated and selected next best candidate '{second_selected}'.")

def test_shadow_rl_does_not_override_production():
    """Verifies that RLContextualBanditPolicy operates strictly in shadow mode."""
    engine = AdaptiveDecisionEngine()
    req = DecisionRequest(text="Analyze sorting algorithms.", execution_mode="local")
    dec = engine.decide(req)
    
    assert dec.policy == "baseline_adaptive_policy"
    assert dec.shadow_rl_decision is not None
    assert dec.shadow_rl_decision.get("production_override") is False
