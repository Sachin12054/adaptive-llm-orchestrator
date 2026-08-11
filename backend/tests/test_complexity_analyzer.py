import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.complexity_analyzer import ComplexityAnalyzer
from app.services.complexity_prototype_service import ComplexityPrototypeService

client = TestClient(app)

@pytest.fixture
def mock_embedding_and_prototypes():
    """Mocks EmbeddingService and ComplexityPrototypeService for fast isolated unit testing."""
    with patch("app.services.complexity_analyzer.EmbeddingService") as mock_emb_cls, \
         patch("app.services.complexity_analyzer.ComplexityPrototypeService") as mock_proto_cls:
        
        mock_emb_inst = MagicMock()
        mock_proto_inst = MagicMock()

        # Fixed 1024-dim vectors
        vec_low = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        vec_high = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)
        vec_input = np.array([0.9, 0.1] + [0.0] * 1022, dtype=np.float32)

        emb_res = MagicMock()
        emb_res.model = "BAAI/bge-m3"
        emb_res.dimension = 1024
        emb_res.embedding = vec_input.tolist()
        mock_emb_inst.generate_embedding.return_value = emb_res

        mock_proto_inst.get_prototype_embeddings.return_value = {
            "low": [{"text": "What is x", "vector": vec_low}],
            "medium": [{"text": "Explain y", "vector": vec_low}],
            "high": [{"text": "Compare a and b", "vector": vec_high}],
            "very_high": [{"text": "Design system z", "vector": vec_high}]
        }

        mock_emb_cls.return_value = mock_emb_inst
        mock_proto_cls.return_value = mock_proto_inst

        yield mock_emb_inst, mock_proto_inst

def test_empty_text_validation():
    analyzer = ComplexityAnalyzer()
    with pytest.raises(ValueError, match="cannot be empty"):
        analyzer.analyze_complexity("")

def test_task_count_calculation():
    analyzer = ComplexityAnalyzer()
    prompt1 = "What is the capital of France?"
    assert analyzer._calculate_task_count(prompt1) == 1

    prompt2 = "Design a system, compare database options, and implement a deployment pipeline."
    assert analyzer._calculate_task_count(prompt2) >= 3

def test_context_complexity_calculation(mock_embedding_and_prototypes):
    analyzer = ComplexityAnalyzer()
    res = analyzer.analyze_complexity("Short prompt")
    assert res.factors.context_complexity < 0.20

    long_prompt = "word " * 300
    res_long = analyzer.analyze_complexity(long_prompt)
    assert res_long.factors.context_complexity == 1.0

def test_weighted_score_formula_and_thresholds(mock_embedding_and_prototypes):
    analyzer = ComplexityAnalyzer(low_threshold=0.30, medium_threshold=0.60, high_threshold=0.80)
    res = analyzer.analyze_complexity("What is the capital of France?")
    
    # Verify score calculation formula:
    # 0.30*sem + 0.25*reas + 0.20*task + 0.15*ctx + 0.10*out
    f = res.factors
    expected_score = round(
        0.30 * f.semantic_complexity +
        0.25 * f.reasoning_complexity +
        0.20 * f.task_complexity +
        0.15 * f.context_complexity +
        0.10 * f.output_complexity,
        4
    )
    assert res.complexity_score == pytest.approx(expected_score, abs=1e-3)
    assert res.complexity_level in ["low", "medium", "high", "very_high"]

def test_api_complexity_analyze_endpoint(mock_embedding_and_prototypes):
    payload = {"text": "Compare PostgreSQL and MongoDB for telemetry platform scalability."}
    response = client.post("/api/complexity/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "complexity_level" in data
    assert "complexity_score" in data
    assert "factors" in data
    assert "task_count" in data
    assert "estimated_reasoning_depth" in data
    assert "breakdown_latency_ms" in data
    assert data["embedding_model"] == "BAAI/bge-m3"
    assert data["embedding_dimension"] == 1024

def test_api_complexity_analyze_empty():
    payload = {"text": "   "}
    response = client.post("/api/complexity/analyze", json=payload)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]
