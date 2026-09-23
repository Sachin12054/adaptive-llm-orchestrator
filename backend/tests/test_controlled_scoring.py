import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.model import ModelMetadata
from app.services.policies.baseline_policy import BaselineAdaptivePolicy

def test_controlled_scoring_mistral_wins():
    """Verify that baseline_adaptive_policy selects Mistral when Mistral has the highest capability match."""
    policy = BaselineAdaptivePolicy()
    
    candidates = [
        ModelMetadata(
            model_id="gemini-3.5-flash",
            provider="Google Gemini API",
            display_name="Gemini 3.5 Flash",
            model_type="llm",
            capabilities=["general_qa", "reasoning"],
            context_length=1000000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        ),
        ModelMetadata(
            model_id="mistral-small-latest",
            provider="Mistral API",
            display_name="Mistral Small Latest",
            model_type="llm",
            capabilities=["translation", "general_qa", "explanation"],
            context_length=32000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        )
    ]

    selected, score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="Translate this text to Spanish",
        intent_info={"intent": "translation", "is_ambiguous": False},
        complexity_info={"level": "medium", "complexity_score": 0.50},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
        candidate_models=candidates
    )

    assert selected == "mistral-small-latest"
    assert selected != "gemini-3.5-flash"

def test_controlled_scoring_groq_wins():
    """Verify that baseline_adaptive_policy selects Groq when Groq has the highest capability match."""
    policy = BaselineAdaptivePolicy()
    
    candidates = [
        ModelMetadata(
            model_id="gemini-3.5-flash",
            provider="Google Gemini API",
            display_name="Gemini 3.5 Flash",
            model_type="llm",
            capabilities=["general_qa", "reasoning"],
            context_length=1000000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        ),
        ModelMetadata(
            model_id="llama-3.3-70b-versatile",
            provider="Groq API",
            display_name="Groq LLaMA 3.3 70B",
            model_type="llm",
            capabilities=["mathematics", "general_qa"],
            context_length=128000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        )
    ]

    selected, score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="Solve 2x + 5 = 15 for x",
        intent_info={"intent": "mathematics", "is_ambiguous": False},
        complexity_info={"level": "medium", "complexity_score": 0.50},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
        candidate_models=candidates
    )

    assert selected == "llama-3.3-70b-versatile"
    assert selected != "gemini-3.5-flash"

def test_controlled_scoring_openrouter_wins():
    """Verify that baseline_adaptive_policy selects OpenRouter when OpenRouter has the highest complexity score for high complexity task."""
    policy = BaselineAdaptivePolicy()
    
    candidates = [
        ModelMetadata(
            model_id="gemini-3.5-flash",
            provider="Google Gemini API",
            display_name="Gemini 3.5 Flash",
            model_type="llm",
            capabilities=["general_qa"],
            context_length=1000000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        ),
        ModelMetadata(
            model_id="meta-llama/llama-3.3-70b-instruct",
            provider="OpenRouter API",
            display_name="OpenRouter LLaMA 3.3 70B",
            model_type="llm",
            capabilities=["reasoning", "general_qa"],
            context_length=128000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        )
    ]

    selected, score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="Prove by induction that sum of n integers is n(n+1)/2",
        intent_info={"intent": "reasoning", "is_ambiguous": False},
        complexity_info={"level": "high", "complexity_score": 0.90},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
        candidate_models=candidates
    )

    assert selected == "meta-llama/llama-3.3-70b-instruct"
    assert selected != "gemini-3.5-flash"

def test_controlled_scoring_gemini_wins():
    """Verify that baseline_adaptive_policy selects Gemini when Gemini has the highest capability match for summarization."""
    policy = BaselineAdaptivePolicy()
    
    candidates = [
        ModelMetadata(
            model_id="gemini-3.5-flash",
            provider="Google Gemini API",
            display_name="Gemini 3.5 Flash",
            model_type="llm",
            capabilities=["summarization", "general_qa"],
            context_length=1000000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        ),
        ModelMetadata(
            model_id="mistral-small-latest",
            provider="Mistral API",
            display_name="Mistral Small Latest",
            model_type="llm",
            capabilities=["translation", "coding"],
            context_length=32000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        )
    ]

    selected, score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="Summarize this long report...",
        intent_info={"intent": "summarization", "is_ambiguous": False},
        complexity_info={"level": "medium", "complexity_score": 0.50},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
        candidate_models=candidates
    )

    assert selected == "gemini-3.5-flash"
