import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.core.config import settings
from app.services.embedding_service import EmbeddingService

client = TestClient(app)

@pytest.fixture
def mock_sentence_transformer():
    """Mocks SentenceTransformer to avoid network model downloads during automated tests."""
    EmbeddingService._shared_model = None
    EmbeddingService._shared_dimension = None
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_model_instance = MagicMock()
        # Mock 1024-dimensional embedding vector
        fake_vector = np.ones(1024, dtype=np.float32)
        mock_model_instance.encode.return_value = fake_vector
        mock_model_instance.get_embedding_dimension.return_value = 1024
        mock_cls.return_value = mock_model_instance
        yield mock_cls
    EmbeddingService._shared_model = None
    EmbeddingService._shared_dimension = None

def test_service_initialization_singleton():
    service1 = EmbeddingService()
    service2 = EmbeddingService()
    assert service1 is service2

def test_model_name_from_config():
    service = EmbeddingService()
    assert service.model_name == settings.EMBEDDING_MODEL

def test_device_resolution():
    service = EmbeddingService()
    dev_cpu = service._resolve_device("cpu")
    assert dev_cpu == "cpu"

    dev_auto = service._resolve_device("auto")
    assert dev_auto in ["cpu", "cuda"]

def test_empty_text_validation():
    service = EmbeddingService()
    with pytest.raises(ValueError, match="cannot be empty"):
        service.generate_embedding("")

def test_embedding_generation_mocked(mock_sentence_transformer):
    service = EmbeddingService()
    
    result = service.generate_embedding("What is the capital of France?", include_vector=False)
    assert result.model == settings.EMBEDDING_MODEL
    assert result.dimension == 1024
    assert result.device in ["cpu", "cuda"]
    assert result.latency_ms >= 0
    assert result.embedding is None  # Default is False

def test_embedding_include_vector(mock_sentence_transformer):
    service = EmbeddingService()
    
    result = service.generate_embedding("Hello world", include_vector=True)
    assert result.embedding is not None
    assert len(result.embedding) == 1024

def test_api_embedding_generate_valid(mock_sentence_transformer):
    service = EmbeddingService()

    payload = {"text": "What is the capital of France?", "include_vector": False}
    response = client.post("/api/embedding/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == settings.EMBEDDING_MODEL
    assert data["dimension"] == 1024
    assert data["embedding"] is None
    assert "latency_ms" in data
    assert "device" in data

def test_api_embedding_generate_empty_text():
    payload = {"text": "   "}
    response = client.post("/api/embedding/generate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "cannot be empty" in data["detail"]

# Optional manual test for verifying real model download and execution
@pytest.mark.manual
def test_real_bge_m3_embedding():
    """Manual integration test - only run when testing real HuggingFace model download."""
    service = EmbeddingService(model_name="BAAI/bge-m3")
    result = service.generate_embedding("Test real model", include_vector=False)
    assert result.dimension == 1024
    assert result.model == "BAAI/bge-m3"
