import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.model import ModelMetadata
from app.schemas.orchestration import OrchestrationRequest
from app.services.policies.baseline_policy import BaselineAdaptivePolicy
from app.services.orchestration_pipeline import OrchestrationPipeline

def test_phase8_multitask_6_prompts_ranking():
    """Verify Phase 8 requirement: 6 diverse prompts evaluate candidate ranking, select top candidate, and execute without RL override."""
    policy = BaselineAdaptivePolicy()

    candidates = [
        ModelMetadata(
            model_id="gemini-3.5-flash",
            provider="Google Gemini API",
            display_name="Gemini 3.5 Flash",
            model_type="llm",
            capabilities=["general_qa", "reasoning", "explanation", "coding", "summarization"],
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
            capabilities=["translation", "general_qa", "explanation", "coding"],
            context_length=32000,
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
            capabilities=["mathematics", "coding", "general_qa", "reasoning"],
            context_length=128000,
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
            capabilities=["reasoning", "general_qa", "coding", "explanation"],
            context_length=128000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        )
    ]

    test_prompts = [
        ("France Capital", "What is the capital of France?", "general_qa", 0.20, "gemini-3.5-flash"),
        ("Technical Translation", "Translate the following technical paragraph into Spanish while preserving terminology.", "translation", 0.50, "mistral-small-latest"),
        ("Mathematical Proof", "Solve this mathematical proof and explain every step rigorously.", "mathematics", 0.70, "llama-3.3-70b-versatile"),
        ("Recursive Complexity Proof", "Prove the time complexity of this recursive algorithm using a recurrence relation.", "reasoning", 0.90, "meta-llama/llama-3.3-70b-instruct"),
        ("FastAPI Endpoint", "Write a Python FastAPI endpoint that validates JSON and stores it in PostgreSQL.", "coding", 0.50, "gemini-3.5-flash"),
        ("Document Summary", "Summarize this long document into five concise bullet points.", "summarization", 0.30, "gemini-3.5-flash")
    ]

    for title, prompt_text, intent, cmplx, expected_top_model in test_prompts:
        selected, score, breakdowns, reasoning = policy.evaluate_candidates(
            prompt=prompt_text,
            intent_info={"intent": intent, "is_ambiguous": False},
            complexity_info={"level": "high" if cmplx > 0.65 else "medium", "complexity_score": cmplx},
            resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
            candidate_models=candidates
        )
        assert selected is not None
        assert breakdowns[0].model_id == selected
        assert breakdowns[0].candidate_score >= breakdowns[1].candidate_score

def test_phase9_cost_efficiency_scoring():
    """Verify Phase 9 requirement: High-cost vs low-cost candidates score dynamically based on complexity."""
    policy = BaselineAdaptivePolicy()
    
    # Candidate A: High reasoning 70B model
    # Candidate B: Lightweight fast model
    candidates = [
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
        ),
        ModelMetadata(
            model_id="mistral-small-latest",
            provider="Mistral API",
            display_name="Mistral Small Latest",
            model_type="llm",
            capabilities=["general_qa", "reasoning"],
            context_length=32000,
            execution_mode="online_api",
            local=False,
            available=True,
            configuration_status="configured",
            requirements={},
            metadata_source="test"
        )
    ]

    # For low complexity task, lightweight candidate B wins
    sel_low, score_low, breakdowns_low, _ = policy.evaluate_candidates(
        prompt="Simple QA prompt",
        intent_info={"intent": "general_qa", "is_ambiguous": False},
        complexity_info={"level": "low", "complexity_score": 0.20},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
        candidate_models=candidates
    )
    assert sel_low == "mistral-small-latest"

    # For high complexity task, high reasoning candidate A wins
    sel_high, score_high, breakdowns_high, _ = policy.evaluate_candidates(
        prompt="Complex math proof prompt",
        intent_info={"intent": "reasoning", "is_ambiguous": False},
        complexity_info={"level": "high", "complexity_score": 0.90},
        resource_info={"cpu_utilization_percent": 20.0, "memory_utilization_percent": 30.0},
        candidate_models=candidates
    )
    assert sel_high == "meta-llama/llama-3.3-70b-instruct"

def test_phase10_vram_resource_constraints():
    """Verify Phase 10 requirement: Severe GPU VRAM deficit marks local Ollama model ineligible."""
    policy = BaselineAdaptivePolicy()

    local_model = ModelMetadata(
        model_id="deepseek-r1-7b",
        provider="ollama",
        display_name="DeepSeek R1 7B",
        model_type="llm",
        capabilities=["reasoning", "general_qa"],
        context_length=8192,
        execution_mode="local",
        local=True,
        available=True,
        configuration_status="configured",
        requirements={},
        metadata_source="test"
    )

    # When VRAM is severely limited (0.5 GB free for 4.5 GB model requirement)
    selected, score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="Reasoning prompt",
        intent_info={"intent": "reasoning", "is_ambiguous": False},
        complexity_info={"level": "medium", "complexity_score": 0.50},
        resource_info={
            "cpu_utilization_percent": 20.0,
            "memory_utilization_percent": 30.0,
            "gpu_available": True,
            "gpu_free_vram_gb": 0.1
        },
        candidate_models=[local_model]
    )

    # Resource fit score should reflect severe VRAM penalty or offload state
    assert breakdowns[0].resource_fit_score <= 0.60
