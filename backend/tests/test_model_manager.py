import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.model_manager import ModelManager
from app.schemas.model_manager import ModelExecutionRequest
from app.schemas.provider import ProviderGenerationResponse, TokenUsage

client = TestClient(app)

def test_no_routing_and_no_fallback_methods():
    manager = ModelManager()
    forbidden_attrs = [
        "select_model", "choose_model", "rank_models",
        "best_model", "route_request", "fallback_model", "adaptive_route"
    ]
    for attr in forbidden_attrs:
        assert not hasattr(manager, attr)

def test_embedding_model_rejection():
    manager = ModelManager()
    req = ModelExecutionRequest(model_id="BAAI/bge-m3", prompt="Hello")
    res = manager.execute(req)
    assert res.success is False
    assert res.execution_status == "unsupported_model"
    assert "embedding model" in res.error_message.lower()

def test_unregistered_model_rejection():
    manager = ModelManager()
    req = ModelExecutionRequest(model_id="unregistered-model-xyz", prompt="Hello")
    res = manager.execute(req)
    assert res.success is False
    assert res.execution_status == "unsupported_model"
    assert "not registered" in res.error_message.lower()

@patch("app.services.providers.ollama_provider.OllamaProvider.generate")
@patch("app.services.model_registry.ModelRegistry.get_model")
def test_valid_model_execution_contract(mock_get_model, mock_provider_gen):
    mock_meta = MagicMock()
    mock_meta.model_id = "gemma-3-4b"
    mock_meta.provider = "ollama"
    mock_meta.model_type = "llm"
    mock_meta.available = True
    mock_meta.configuration_status = "configured"
    mock_get_model.return_value = mock_meta

    mock_provider_gen.return_value = ProviderGenerationResponse(
        provider="ollama",
        model_id="gemma-3-4b",
        generated_text="Paris is the capital of France.",
        finish_reason="STOP",
        usage=TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15),
        latency_ms=120.5,
        success=True,
        error_message=None
    )

    manager = ModelManager()
    req = ModelExecutionRequest(model_id="gemma-3-4b", prompt="What is the capital of France?")
    res = manager.execute(req)

    assert res.success is True
    assert res.execution_status == "completed"
    assert res.model_id == "gemma-3-4b"
    assert res.provider == "ollama"
    assert res.generated_text == "Paris is the capital of France."
    assert res.usage.total_tokens == 15

def test_api_model_manager_status_endpoint():
    response = client.get("/api/model-manager/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "registered_models_count" in data
    assert "executable_llm_models_count" in data
    assert "providers_status" in data

def test_api_model_manager_execute_embedding_rejection():
    payload = {
        "model_id": "BAAI/bge-m3",
        "prompt": "What is the capital of France?"
    }
    response = client.post("/api/model-manager/execute", json=payload)
    assert response.status_code == 400
    assert "embedding model" in response.json()["detail"].lower()
