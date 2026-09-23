import sys
import os
import pytest
import inspect
import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.experience_buffer import ExperienceBufferService, StateEncoder, ACTION_MAP
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.experience import ExperienceRecord
from app.schemas.orchestration import OrchestrationRequest, OrchestrationResponse
from app.schemas.model import ModelMetadata
from app.schemas.model_manager import ModelExecutionResponse
from app.schemas.provider import TokenUsage

client = TestClient(app)

def test_no_prohibited_learning_methods():
    buffer_service = ExperienceBufferService()
    prohibited_methods = [
        "select_model", "choose_model", "rank_models", "best_model",
        "fallback_model", "route_request", "score_models",
        "train", "fit", "learn", "update_policy"
    ]
    for method in prohibited_methods:
        assert not hasattr(buffer_service, method)

def test_no_direct_gemini_sdk_import():
    import app.services.experience_buffer as buffer_module
    source_code = inspect.getsource(buffer_module)
    assert "google.genai" not in source_code
    assert "google.generativeai" not in source_code

def test_action_mapping():
    assert ACTION_MAP.get("gemma-3-4b") == 0
    assert ACTION_MAP.get("qwen-coder-3b") == 1
    assert ACTION_MAP.get("deepseek-r1-7b") == 2
    assert ACTION_MAP.get("none", -1) == -1

def test_state_vector_encoding():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    encoder = StateEncoder()
    state = encoder.encode_state(res)

    assert isinstance(state, list)
    assert len(state) == 12
    assert all(isinstance(x, float) for x in state)

def test_state_vector_is_deterministic():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    encoder = StateEncoder()
    state1 = encoder.encode_state(res)
    state2 = encoder.encode_state(res)

    assert state1 == state2

def test_experience_record_creation():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    buffer_service = ExperienceBufferService(capacity=100)
    record = buffer_service.record_from_orchestration(res)

    assert record.experience_id.startswith("exp-")
    assert len(record.state) == 12
    assert record.action_model_id == (res.selected_model or "none")
    assert record.next_state is None
    assert record.done is True

def test_experience_record_serialization():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    buffer_service = ExperienceBufferService(capacity=100)
    record = buffer_service.record_from_orchestration(res)

    json_str = record.json()
    deserialized = ExperienceRecord.parse_raw(json_str)

    assert deserialized.experience_id == record.experience_id
    assert deserialized.state == record.state
    assert deserialized.reward == record.reward

def test_reward_is_preserved_exactly():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    buffer_service = ExperienceBufferService(capacity=100)
    record = buffer_service.record_from_orchestration(res)

    assert record.reward == pytest.approx(res.reward.reward)

def test_default_buffer_uses_test_isolated_path():
    default = ExperienceBufferService()
    assert default.persistence_path == os.environ["ADAPTIVE_TEST_BUFFER_PATH"]


def test_temp_buffer_is_isolated_from_default_singleton():
    default_before = ExperienceBufferService()
    temp = ExperienceBufferService(capacity=10, persistence_path="data/test_temp_buffer.jsonl")
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_temp_buffer.jsonl"))

    assert temp.persistence_path == temp_path
    assert default_before.persistence_path == os.environ["ADAPTIVE_TEST_BUFFER_PATH"]

    default_after = ExperienceBufferService()
    assert default_after.persistence_path == os.environ["ADAPTIVE_TEST_BUFFER_PATH"]

    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_temp_buffer_does_not_modify_canonical_buffer():
    canonical_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "rl", "experience_buffer.jsonl"))
    canonical_before = open(canonical_path, "rb").read() if os.path.exists(canonical_path) else b""
    canonical_count_before = sum(1 for line in open(canonical_path, encoding="utf-8") if line.strip()) if os.path.exists(canonical_path) else 0

    temp = ExperienceBufferService(capacity=5, persistence_path="data/test_temp_buffer.jsonl")
    temp.clear()

    pipeline = OrchestrationPipeline(experience_buffer=temp)
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))
    assert len(temp._buffer) >= 1
    canonical_after = open(canonical_path, "rb").read() if os.path.exists(canonical_path) else b""
    canonical_count_after = sum(1 for line in open(canonical_path, encoding="utf-8") if line.strip()) if os.path.exists(canonical_path) else 0
    assert canonical_after == canonical_before
    assert canonical_count_after == canonical_count_before
    assert temp.persistence_path != canonical_path

    if os.path.exists(temp.persistence_path):
        os.remove(temp.persistence_path)


def test_buffer_append():
    buffer_service = ExperienceBufferService(capacity=10, persistence_path="data/test_scratch_buffer.jsonl")
    buffer_service.clear()

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    initial_size = len(buffer_service._buffer)
    buffer_service.record_from_orchestration(res)

    assert len(buffer_service._buffer) == initial_size + 1


def test_buffer_capacity():
    buffer_service = ExperienceBufferService(capacity=5, persistence_path="data/test_scratch_buffer2.jsonl")
    buffer_service.clear()

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    for _ in range(10):
        buffer_service.record_from_orchestration(res)

    assert len(buffer_service._buffer) == 5

def test_fifo_eviction():
    buffer_service = ExperienceBufferService(capacity=3, persistence_path="data/test_scratch_buffer3.jsonl")
    buffer_service.clear()

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    rec1 = buffer_service.record_from_orchestration(res)
    rec2 = buffer_service.record_from_orchestration(res)
    rec3 = buffer_service.record_from_orchestration(res)
    rec4 = buffer_service.record_from_orchestration(res)

    ids = [r.experience_id for r in buffer_service._buffer]
    assert rec1.experience_id not in ids
    assert rec4.experience_id in ids
    assert len(buffer_service._buffer) == 3

def test_batch_sampling():
    buffer_service = ExperienceBufferService(capacity=20, persistence_path="data/test_scratch_buffer4.jsonl")
    buffer_service.clear()

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    for _ in range(5):
        buffer_service.record_from_orchestration(res)

    samples = buffer_service.sample_batch(3)
    assert len(samples) == 3
    assert isinstance(samples[0], ExperienceRecord)

def test_invalid_batch_size():
    buffer_service = ExperienceBufferService(capacity=20, persistence_path="data/test_scratch_buffer5.jsonl")
    buffer_service.clear()

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))
    buffer_service.record_from_orchestration(res)

    with pytest.raises(ValueError, match="invalid for buffer size"):
        buffer_service.sample_batch(10)

def test_empty_buffer_sampling():
    buffer_service = ExperienceBufferService(capacity=20, persistence_path="data/test_scratch_buffer6.jsonl")
    buffer_service.clear()

    with pytest.raises(ValueError, match="empty experience buffer"):
        buffer_service.sample_batch(1)

def test_invalid_experience_rejection():
    with pytest.raises(Exception):
        ExperienceRecord(reward=0.5)

def test_next_state_is_none_for_terminal_single_request():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    buffer_service = ExperienceBufferService(capacity=10)
    record = buffer_service.record_from_orchestration(res)

    assert record.next_state is None
    assert record.done is True

def test_no_secret_values_are_serialized():
    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    buffer_service = ExperienceBufferService(capacity=10)
    record = buffer_service.record_from_orchestration(res)

    record_json = record.json()
    assert "GEMINI_API_KEY" not in record_json
    assert "AIzaSy" not in record_json

@patch("app.services.model_registry.ModelRegistry.list_models")
@patch("app.services.model_manager.ModelManager.execute")
def test_e2e_pipeline_to_experience_buffer(mock_mm_execute, mock_list_models):
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
    res = pipeline.run_pipeline(OrchestrationRequest(prompt="What is the capital of France?"))

    status = pipeline.experience_buffer.get_status()
    assert status.current_size > 0

    latest_record = pipeline.experience_buffer._buffer[-1]
    assert latest_record.action_model_id == res.selected_model == "gemma-3-4b"
    assert latest_record.reward == pytest.approx(res.reward.reward) == pytest.approx(0.7550)

def test_api_experience_endpoints():
    from app.api.routes.experience import buffer_service

    buffer_service.append_experience(ExperienceRecord(
        experience_id="exp-api-isolation",
        state=[0.0] * 12,
        action=0,
        action_model_id="gemma-3-4b",
        reward=0.5,
        timestamp=time.time(),
        done=True,
        executed_action=0,
        executed_model="gemma-3-4b",
        executed_provider="ollama",
        is_valid_rl_sample=True,
        behavior_action=0,
        behavior_model_id="gemma-3-4b",
        propensity_probability=1.0,
        candidate_action_probabilities={"gemma-3-4b": 1.0},
        propensity_available=True,
    ))

    status_resp = client.get("/api/experience/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["service"] == "experience_buffer"
    assert "capacity" in data
    assert "current_size" in data

    sample_resp = client.get("/api/experience/sample?batch_size=1")
    assert sample_resp.status_code == 200
    sample_data = sample_resp.json()
    assert "experiences" in sample_data
