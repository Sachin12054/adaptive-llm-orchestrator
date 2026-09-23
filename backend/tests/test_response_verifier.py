import sys
import os
import pytest
import inspect
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.response_verifier import ResponseVerifier
from app.schemas.verification import VerificationRequest

client = TestClient(app)

def test_architectural_boundaries_no_prohibited_methods():
    verifier = ResponseVerifier()
    prohibited_methods = [
        "select_model", "choose_model", "rank_models", "best_model",
        "fallback_model", "route_request", "score_models",
        "generate_response", "regenerate", "retry_with_fallback"
    ]
    for method in prohibited_methods:
        assert not hasattr(verifier, method)

def test_architectural_boundaries_no_direct_gemini_sdk_import():
    import app.services.response_verifier as verifier_module
    source_code = inspect.getsource(verifier_module)
    assert "google.genai" not in source_code
    assert "google.generativeai" not in source_code

def test_valid_generated_response_passes_baseline_verification():
    verifier = ResponseVerifier()
    req = VerificationRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        generated_text="Paris is the capital of France.",
        generation_success=True,
        execution_status="completed"
    )

    res = verifier.verify_response(req)

    assert res.verified is True
    assert res.verification_status == "verified_baseline"
    assert res.response_present is True
    assert res.relevance_score >= 0.30
    assert res.structural_quality_score >= 0.50
    assert res.factual_verification_status == "not_verified"
    assert res.prompt == "What is the capital of France?"
    assert res.selected_model == "gemini-3.5-flash"
    assert any("Factual correctness is not established" in r for r in res.verification_reasoning)

def test_empty_response_rejected():
    verifier = ResponseVerifier()
    req = VerificationRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        generated_text="   ",
        generation_success=True,
        execution_status="completed"
    )

    res = verifier.verify_response(req)

    assert res.verified is False
    assert res.verification_status == "empty_response"
    assert res.response_present is False
    assert "contains only whitespace" in res.issues[0].lower()

def test_none_response_rejected():
    verifier = ResponseVerifier()
    req = VerificationRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        generated_text=None,
        generation_success=True,
        execution_status="completed"
    )

    res = verifier.verify_response(req)

    assert res.verified is False
    assert res.verification_status == "not_verifiable"
    assert res.response_present is False

def test_not_configured_execution_handling():
    verifier = ResponseVerifier()
    req = VerificationRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        generated_text=None,
        generation_success=False,
        execution_status="not_configured"
    )

    res = verifier.verify_response(req)

    assert res.verified is False
    assert res.verification_status == "not_verifiable"
    assert res.response_present is False
    assert "did not complete" in res.issues[0].lower()

def test_failed_execution_handling():
    verifier = ResponseVerifier()
    req = VerificationRequest(
        prompt="What is the capital of France?",
        selected_model="gemini-3.5-flash",
        generated_text=None,
        generation_success=False,
        execution_status="failed"
    )

    res = verifier.verify_response(req)

    assert res.verified is False
    assert res.verification_status == "not_verifiable"

def test_empty_prompt_rejection():
    verifier = ResponseVerifier()
    req = VerificationRequest(
        prompt="   ",
        selected_model="gemini-3.5-flash",
        generated_text="Paris",
        generation_success=True,
        execution_status="completed"
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        verifier.verify_response(req)

def test_api_verification_status_endpoint():
    response = client.get("/api/verification/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "response_verifier"
    assert data["status"] == "ready"
    assert data["factual_verification_available"] is False

def test_api_verification_verify_endpoint():
    payload = {
        "prompt": "What is the capital of France?",
        "selected_model": "gemini-3.5-flash",
        "generated_text": "Paris is the capital of France.",
        "generation_success": True,
        "execution_status": "completed"
    }
    response = client.post("/api/verification/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["verified"] is True
    assert data["verification_status"] == "verified_baseline"
    assert data["factual_verification_status"] == "not_verified"
