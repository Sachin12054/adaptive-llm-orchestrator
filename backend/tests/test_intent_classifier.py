import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.intent_classifier import IntentClassifier
from app.services.intent_prototype_service import IntentPrototypeService

client = TestClient(app)

@pytest.fixture
def mock_embedding_and_prototypes():
    """Mocks EmbeddingService and IntentPrototypeService for fast isolated unit tests."""
    with patch("app.services.intent_classifier.EmbeddingService") as mock_emb_cls, \
         patch("app.services.intent_classifier.IntentPrototypeService") as mock_proto_cls:
        
        mock_emb_inst = MagicMock()
        mock_proto_inst = MagicMock()

        # Fixed 1024-dim vectors
        vec_coding = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        vec_factual = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)
        vec_input = np.array([0.9, 0.1] + [0.0] * 1022, dtype=np.float32)

        emb_res = MagicMock()
        emb_res.model = "BAAI/bge-m3"
        emb_res.dimension = 1024
        emb_res.embedding = vec_input.tolist()
        mock_emb_inst.generate_embedding.return_value = emb_res

        mock_proto_inst.get_prototype_embeddings.return_value = {
            "coding": [{"text": "Write code", "vector": vec_coding}],
            "factual": [{"text": "What is x", "vector": vec_factual}]
        }

        mock_emb_cls.return_value = mock_emb_inst
        mock_proto_cls.return_value = mock_proto_inst

        yield mock_emb_inst, mock_proto_inst

def test_cosine_similarity_calculation():
    classifier = IntentClassifier()
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    vec_c = np.array([0.0, 1.0, 0.0])
    
    assert classifier._cosine_similarity(vec_a, vec_b) == pytest.approx(1.0)
    assert classifier._cosine_similarity(vec_a, vec_c) == pytest.approx(0.0)

def test_empty_text_validation():
    classifier = IntentClassifier()
    with pytest.raises(ValueError, match="cannot be empty"):
        classifier.classify_intent("")

def test_intent_classification_logic(mock_embedding_and_prototypes):
    classifier = IntentClassifier(top_k=1, margin_threshold=0.01)
    res = classifier.classify_intent("Write a python function")

    assert res.intent == "coding"
    assert res.top_similarity > res.second_similarity
    assert res.margin > 0
    assert len(res.ranked_intents) == 2
    assert res.ranked_intents[0].intent == "coding"
    assert res.ranked_intents[1].intent == "factual"
    assert not res.is_ambiguous

def test_ambiguity_detection(mock_embedding_and_prototypes):
    # Set high margin threshold to trigger ambiguity
    classifier = IntentClassifier(top_k=1, margin_threshold=0.95)
    res = classifier.classify_intent("Write a python function")
    assert res.is_ambiguous is True

def test_api_intent_classify_endpoint(mock_embedding_and_prototypes):
    payload = {"text": "Write a python function to reverse a list."}
    response = client.post("/api/intent/classify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "intent" in data
    assert "top_similarity" in data
    assert "second_similarity" in data
    assert "margin" in data
    assert "is_ambiguous" in data
    assert "ranked_intents" in data
    assert "breakdown_latency_ms" in data
    assert data["embedding_model"] == "BAAI/bge-m3"

def test_api_intent_classify_empty():
    payload = {"text": "   "}
    response = client.post("/api/intent/classify", json=payload)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]
