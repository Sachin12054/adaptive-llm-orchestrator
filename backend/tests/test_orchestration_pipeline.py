import sys
import os
import pytest
import inspect
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest
from app.schemas.model_manager import ModelExecutionResponse
from app.schemas.provider import TokenUsage
from app.schemas.model import ModelMetadata

client = TestClient(app)

def test_no_prohibited_methods_in_pipeline():
    pipeline = OrchestrationPipeline()
    prohibited_methods = [
        "select_model", "choose_model", "rank_models", "best_model",
        "fallback_model", "route_request", "score_models",
        "train", "fit", "learn", "update_policy"
    ]
    for method in prohibited_methods:
        assert not hasattr(pipeline, method)

def test_no_direct_gemini_sdk_import():
    import app.services.orchestration_pipeline as pipe_module
    source_code = inspect.getsource(pipe_module)
    assert "google.genai" not in source_code
    assert "google.generativeai" not in source_code

def test_empty_prompt_rejection():
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="   ")
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.run_pipeline(req)

@patch("app.services.model_manager.ModelManager.execute")
def test_configured_pipeline_execution(mock_mm_execute):
    mock_mm_execute.return_value = ModelExecutionResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="ollama",
        generated_text="Paris is the capital of France.",
        finish_reason="STOP",
        latency_ms=120.0,
        usage=TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15),
        execution_status="completed",
        error_message=None
    )

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="What is the capital of France?")
    res = pipeline.run_pipeline(req)

    assert res.prompt == "What is the capital of France?"
    assert res.selected_model == "gemma-3-4b"
    assert res.decision_score > 0.70
    assert res.generation.execution_status == "completed"
    assert res.generation.generated_text is not None
    assert "Paris" in res.generation.generated_text
    assert res.verification.verified is True
    assert res.verification.verification_status == "verified_baseline"
    assert res.reward.reward > 0.70
    assert res.pipeline_latency_ms > 0.0

@patch("app.services.model_registry.ModelRegistry.list_models")
@patch("app.services.model_manager.ModelManager.execute")
def test_controlled_mocked_success_e2e_pipeline(mock_mm_execute, mock_list_models):
    gemma_meta = ModelMetadata(
        model_id="gemma-3-4b",
        provider="ollama",
        display_name="Gemma 3 4B",
        model_type="llm",
        capabilities=["general_qa", "explanation", "coding", "summarization"],
        context_length=8192,
        execution_mode="local",
        local=True,
        available=True,
        configuration_status="configured",
        metadata_source="test"
    )
    mock_list_models.return_value = [gemma_meta]

    mock_mm_execute.return_value = ModelExecutionResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="ollama",
        generated_text="Paris is the capital of France.",
        finish_reason="STOP",
        latency_ms=120.0,
        usage=TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15),
        execution_status="completed",
        error_message=None
    )

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="What is the capital of France?")
    res = pipeline.run_pipeline(req)

    assert res.success is True
    assert res.prompt == "What is the capital of France?"
    assert res.selected_model == "gemma-3-4b"
    assert res.decision_score > 0.70
    assert res.generation.generated_text == "Paris is the capital of France."
    assert res.generation.execution_status == "completed"
    assert res.verification.verified is True
    assert res.verification.verification_status == "verified_baseline"
    
    # Exact Step 18 reward calculation for baseline verified response
    assert res.reward.reward == pytest.approx(0.7550)
    assert res.reward.reward_status == "completed_verified"
    assert res.pipeline_latency_ms > 0.0

def test_api_orchestration_status_endpoint():
    response = client.get("/api/orchestrate/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "e2e_orchestrator"
    assert "status" in data
    assert "active_policy" in data

@patch("app.services.model_manager.ModelManager.execute")
def test_api_orchestration_route_execution(mock_mm_execute):
    mock_mm_execute.return_value = ModelExecutionResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="ollama",
        generated_text="Paris is the capital of France.",
        finish_reason="STOP",
        latency_ms=120.0,
        usage=TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15),
        execution_status="completed",
        error_message=None
    )

    payload = {"prompt": "What is the capital of France?"}
    response = client.post("/api/orchestrate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == "What is the capital of France?"
    assert data["selected_model"] == "gemma-3-4b"
    assert data["generation"]["execution_status"] == "completed"
    assert data["generation"]["generated_text"] is not None
    assert "Paris" in data["generation"]["generated_text"]
    assert data["verification"]["verified"] is True
    assert data["reward"]["reward"] > 0.70
