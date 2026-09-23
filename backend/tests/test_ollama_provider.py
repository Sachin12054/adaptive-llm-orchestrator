import sys
import os
import json
import pytest
import inspect
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.providers.ollama_provider import OllamaProvider
from app.services.model_manager import ModelManager
from app.schemas.provider import ProviderGenerationRequest
from app.schemas.model_manager import ModelExecutionRequest

def test_no_gemini_sdk_import_in_ollama_provider():
    import app.services.providers.ollama_provider as ollama_mod
    source = inspect.getsource(ollama_mod)
    assert "google.genai" not in source
    assert "google.generativeai" not in source
    assert "google" not in source

def test_ollama_provider_get_status_success():
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"models": [{"name": "gemma-3-4b:latest"}]}).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OllamaProvider()
        status = provider.get_status()
        assert status.provider is not None

def test_ollama_provider_get_status_unreachable():
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        provider = OllamaProvider(base_url="http://localhost:11434")
        status = provider.get_status()
        assert status.configured is False
        assert status.available is False

def test_ollama_provider_successful_generation():
    req = ProviderGenerationRequest(model_id="gemma-3-4b", prompt="Hello!")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value.status = 200
    mock_resp.__enter__.return_value.read.return_value = json.dumps({
        "response": "Hello world from Gemma!",
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 6
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OllamaProvider()
        provider.available = True
        res = provider.generate(req)
        assert res.success is True
        assert res.generated_text == "Hello world from Gemma!"

def test_ollama_provider_missing_model_error():
    req = ProviderGenerationRequest(model_id="non-existent-model", prompt="Hello!")

    http_err = urllib.error.HTTPError("http://localhost:11434", 404, "Not Found", {}, MagicMock(read=lambda: b"model 'non-existent-model' not found"))
    with patch("urllib.request.urlopen", side_effect=http_err):
        provider = OllamaProvider()
        res = provider.generate(req)
        assert res.success is False

def test_model_manager_dispatches_ollama():
    req = ModelExecutionRequest(model_id="qwen-coder-3b", prompt="Write a function")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value.status = 200
    mock_resp.__enter__.return_value.read.return_value = json.dumps({
        "response": "def foo(): pass",
        "done": True
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        manager = ModelManager()
        manager.providers["ollama"].available = True
        res = manager.execute(req)
        assert res.success is True
        assert res.model_id == "qwen-coder-3b"
        assert res.generated_text == "def foo(): pass"

def test_no_gemini_fallback_on_ollama_failure():
    manager = ModelManager()
    req = ModelExecutionRequest(model_id="gemma-3-4b", prompt="Test")

    # Mock Ollama failure
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        res = manager.execute(req)
        assert res.success is False
        assert res.execution_status == "failed"
        # Ensure provider remained ollama and did NOT fall back to Gemini
        assert res.provider == "ollama"
