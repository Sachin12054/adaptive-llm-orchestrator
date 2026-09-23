import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.complexity_analyzer import ComplexityAnalyzer

client = TestClient(app)

# 6 Fixtures for Regression Testing
@pytest.fixture
def prompt_very_simple_factual():
    return "What is the capital of France?"

@pytest.fixture
def prompt_simple_programming():
    return "Write a Python print statement that outputs Hello World."

@pytest.fixture
def prompt_medium_conceptual():
    return "Compare the core differences between TCP and UDP networking protocols with one practical example."

@pytest.fixture
def prompt_high_algorithmic():
    return "Derive the time complexity of Dijkstra's algorithm using a binary min-heap and explain why it changes compared to an unindexed array implementation."

@pytest.fixture
def prompt_very_high_distributed_architecture():
    return "Design a fault-tolerant distributed LLM serving architecture with dynamic model routing, explain consistency requirements, failure handling, GPU scheduling strategy, and provide pseudocode."

@pytest.fixture
def prompt_very_high_math_reasoning():
    return "Derive the theoretical regret bound O(sqrt(d T ln(T))) for the LinUCB Contextual Bandit algorithm under linear payoff assumptions, detailing the Matrix Inversion Lemma proof steps."

# Mock for fast isolated unit testing
@pytest.fixture
def mock_embedding_and_prototypes():
    with patch("app.services.complexity_analyzer.EmbeddingService") as mock_emb_cls, \
         patch("app.services.complexity_analyzer.ComplexityPrototypeService") as mock_proto_cls:
        
        mock_emb_inst = MagicMock()
        mock_proto_inst = MagicMock()

        vec_low = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        vec_med = np.array([0.5, 0.5] + [0.0] * 1022, dtype=np.float32)
        vec_high = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)

        emb_res = MagicMock()
        emb_res.model = "BAAI/bge-m3"
        emb_res.dimension = 1024
        emb_res.embedding = vec_med.tolist()
        mock_emb_inst.generate_embedding.return_value = emb_res

        mock_proto_inst.get_prototype_embeddings.return_value = {
            "low": [{"text": "What is x", "vector": vec_low}],
            "medium": [{"text": "Explain y", "vector": vec_med}],
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

# Section 6 Protection Tests:
def test_semantic_complexity_floor_scaling():
    """Validates that Softmax-normalized semantic similarity does not inflate floor scores for simple queries."""
    analyzer = ComplexityAnalyzer()
    res_simple = analyzer.analyze_complexity("What is the capital of France?")
    # Semantic complexity for simple query should be bounded well below 0.50
    assert res_simple.factors.semantic_complexity < 0.50

def test_very_high_threshold_reachability(prompt_very_high_distributed_architecture):
    """Validates that complex system architecture prompts reach VERY_HIGH complexity level."""
    analyzer = ComplexityAnalyzer(low_threshold=0.3280, medium_threshold=0.4753, high_threshold=0.5564)
    res = analyzer.analyze_complexity(prompt_very_high_distributed_architecture)
    assert res.complexity_score >= 0.5564
    assert res.complexity_level == "very_high"

def test_complexity_analyzer_quantile_distribution(
    prompt_very_simple_factual,
    prompt_medium_conceptual,
    prompt_very_high_distributed_architecture
):
    """Validates monotonic score progression across complexity tiers."""
    analyzer = ComplexityAnalyzer()
    s_low = analyzer.analyze_complexity(prompt_very_simple_factual).complexity_score
    s_med = analyzer.analyze_complexity(prompt_medium_conceptual).complexity_score
    s_vhigh = analyzer.analyze_complexity(prompt_very_high_distributed_architecture).complexity_score

    assert s_low < s_med < s_vhigh, f"Score progression broken: low={s_low}, med={s_med}, vhigh={s_vhigh}"

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
