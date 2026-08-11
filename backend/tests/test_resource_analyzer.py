import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.resource_analyzer import ResourceAnalyzer

client = TestClient(app)

def test_resource_snapshot_structure():
    analyzer = ResourceAnalyzer()
    snapshot = analyzer.get_resource_snapshot()
    
    assert snapshot.cpu.logical_cores >= 1
    assert snapshot.cpu.utilization_percent >= 0.0
    assert snapshot.memory.total_gb > 0.0
    assert snapshot.memory.available_gb >= 0.0
    assert snapshot.memory.utilization_percent >= 0.0
    assert isinstance(snapshot.cuda.available, bool)
    assert isinstance(snapshot.gpu.available, bool)
    assert snapshot.gpu.ollama_gpu_status is not None
    assert snapshot.gpu.pytorch_cuda_status is not None
    assert snapshot.process.pid > 0
    assert snapshot.process.memory_mb > 0.0
    assert snapshot.disk.total_gb > 0.0
    assert snapshot.feasibility.status in ["sufficient", "constrained", "critical", "unknown"]
    assert snapshot.measurement_latency_ms >= 0.0

def test_api_resource_snapshot_endpoints():
    response = client.get("/api/resource/snapshot")
    assert response.status_code == 200
    data = response.json()

    assert "cpu" in data
    assert "memory" in data
    assert "gpu" in data
    assert "cuda" in data
    assert data["gpu"]["ollama_gpu_status"] == "GPU Available (Local Ollama)"
    assert "pytorch_cuda_status" in data["gpu"]

    resp_plural = client.get("/api/resources/snapshot")
    assert resp_plural.status_code == 200
