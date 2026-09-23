import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.providers.gemini_provider import GeminiProvider
from app.schemas.provider import ProviderGenerationRequest

client = TestClient(app)

def test_missing_configuration_handling():
    provider = GeminiProvider(api_key="")
    status = provider.get_status()
    assert status.configured is False

    req = ProviderGenerationRequest(model_id="gemini-3.5-flash", prompt="Hello")
    res = provider.generate(req)
    assert res.success is False
    assert "not configured" in res.error_message.lower()

def test_shutdown_and_unregistered_model_rejection():
    provider = GeminiProvider(api_key="fake-key-for-test")
    
    # Test shut-down model rejection
    req_shutdown = ProviderGenerationRequest(model_id="gemini-1.5-flash", prompt="Hello")
    res_shutdown = provider.generate(req_shutdown)
    assert res_shutdown.success is False
    assert "not an active registered gemini" in res_shutdown.error_message.lower()

    # Test unregistered model rejection
    req_unreg = ProviderGenerationRequest(model_id="unregistered-model-xyz", prompt="Hello")
    res_unreg = provider.generate(req_unreg)
    assert res_unreg.success is False
    assert "not an active registered gemini" in res_unreg.error_message.lower()

def test_empty_prompt_rejection():
    provider = GeminiProvider(api_key="fake-key-for-test")
    req = ProviderGenerationRequest(model_id="gemini-3.5-flash", prompt="   ")
    res = provider.generate(req)
    assert res.success is False
    assert "cannot be empty" in res.error_message.lower()

def test_no_routing_methods_in_provider():
    provider = GeminiProvider()
    forbidden_attrs = ["select_model", "choose_model", "rank_models", "route_request", "best_model", "fallback_model"]
    for attr in forbidden_attrs:
        assert not hasattr(provider, attr), f"Provider must not have routing method '{attr}'"

@patch("google.genai.Client")
def test_mocked_gemini_generation_success(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini response text."
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    provider = GeminiProvider(api_key="valid-test-key")
    req = ProviderGenerationRequest(model_id="gemini-3.5-flash", prompt="Explain quantum computing.")

    res = provider.generate(req)
    assert res.success is True
    assert res.generated_text == "Mocked Gemini response text."
    assert res.model_id == "gemini-3.5-flash"
    assert res.provider == "Google Gemini API"

def test_api_gemini_status_endpoint():
    response = client.get("/api/providers/gemini/status")
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert "provider" in data

def test_api_gemini_generate_shutdown_model():
    payload = {
        "model_id": "gemini-1.5-flash",
        "prompt": "Test prompt"
    }
    response = client.post("/api/providers/gemini/generate", json=payload)
    assert response.status_code == 200 or response.status_code == 400
