import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.model_registry import ModelRegistry

client = TestClient(app)

def test_registry_loads_active_models():
    registry = ModelRegistry()
    models = registry.list_models()
    model_ids = [m.model_id for m in models]

    assert "BAAI/bge-m3" in model_ids
    assert "gemma-3-4b" in model_ids
    assert "qwen-coder-3b" in model_ids
    assert "deepseek-r1-7b" in model_ids
    assert "gemini-2.0-flash" in model_ids

    assert "gemini-3.6-flash" not in model_ids
    assert "gemini-2.5-flash" not in model_ids

def test_bge_m3_metadata():
    registry = ModelRegistry()
    bge = registry.get_model("BAAI/bge-m3")
    assert bge is not None
    assert bge.model_type == "embedding"
    assert bge.embedding_dimension == 1024
    assert bge.local is True
    assert bge.available is True
    assert bge.configuration_status == "configured"

def test_ollama_models_metadata():
    registry = ModelRegistry()
    gemma = registry.get_model("gemma-3-4b")
    assert gemma is not None
    assert gemma.provider == "ollama"
    assert gemma.execution_mode == "local"
    assert gemma.available is True

    coder = registry.get_model("qwen-coder-3b")
    assert coder is not None
    assert coder.provider == "ollama"
    assert coder.execution_mode == "local"

    deepseek = registry.get_model("deepseek-r1-7b")
    assert deepseek is not None
    assert deepseek.provider == "ollama"
    assert deepseek.execution_mode == "local"

def test_no_routing_logic_present_in_registry():
    registry = ModelRegistry()
    forbidden_attrs = ["choose_model", "select_provider", "rank_models", "best_model", "routing_score"]
    for attr in forbidden_attrs:
        assert not hasattr(registry, attr)

def test_api_list_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["total_count"] == 8

def test_api_get_model_by_id_endpoint():
    response = client.get("/api/models/gemma-3-4b")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "gemma-3-4b"
    assert data["provider"] == "ollama"

def test_api_get_shutdown_model_404():
    response = client.get("/api/models/gemini-3.6-flash")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
