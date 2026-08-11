import sys
import os
import pytest
import inspect
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.response_generator import ResponseGenerator
from app.schemas.response import ResponseGenerationRequest
from app.schemas.model_manager import ModelExecutionResponse
from app.schemas.provider import TokenUsage

client = TestClient(app)

def test_architectural_boundaries_no_model_selection():
    generator = ResponseGenerator()
    forbidden_attrs = [
        "select_model", "choose_model", "rank_models",
        "best_model", "fallback_model", "route_request", "score_models"
    ]
    for attr in forbidden_attrs:
        assert not hasattr(generator, attr)

def test_architectural_boundaries_no_direct_gemini_sdk_import():
    import app.services.response_generator as rg_module
    source_code = inspect.getsource(rg_module)
    assert "google.genai" not in source_code
    assert "google.generativeai" not in source_code

def test_empty_prompt_rejection():
    generator = ResponseGenerator()
    req = ResponseGenerationRequest(prompt="   ", selected_model="gemma-3-4b")
    with pytest.raises(ValueError, match="cannot be empty"):
        generator.generate_response(req)

@patch("app.services.model_manager.ModelManager.execute")
def test_successful_mocked_response_generation(mock_mm_execute):
    mock_mm_execute.return_value = ModelExecutionResponse(
        success=True,
        model_id="gemma-3-4b",
        provider="ollama",
        generated_text="Paris is the capital of France.",
        finish_reason="STOP",
        latency_ms=100.0,
        usage=TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15),
        execution_status="completed",
        error_message=None
    )

    generator = ResponseGenerator()
    req = ResponseGenerationRequest(prompt="What is the capital of France?", selected_model="gemma-3-4b")
    res = generator.generate_response(req)

    assert res.success is True
    assert res.model_id == "gemma-3-4b"
    assert res.provider == "ollama"
    assert res.generated_text == "Paris is the capital of France."
    assert res.finish_reason == "STOP"
    assert res.usage.total_tokens == 15
    assert res.execution_status == "completed"

    mock_mm_execute.assert_called_once()
    called_arg = mock_mm_execute.call_args[0][0]
    assert called_arg.model_id == "gemma-3-4b"
    assert called_arg.prompt == "What is the capital of France?"

def test_unsupported_model_handling():
    generator = ResponseGenerator()
    req = ResponseGenerationRequest(prompt="What is the capital of France?", selected_model="unsupported-model-xyz")
    res = generator.generate_response(req)

    assert res.success is False
    assert res.execution_status == "unsupported_model"
    assert "not registered" in res.error_message.lower()

def test_api_response_status_endpoint():
    response = client.get("/api/response/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "response_generator"
    assert "status" in data
    assert "model_manager_available" in data
