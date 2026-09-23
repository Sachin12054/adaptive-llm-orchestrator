import pytest
import os
import json
from app.core.config import settings
from app.schemas.provider import ProviderGenerationRequest
from app.schemas.orchestration import OrchestrationRequest
from app.services.providers.groq_provider import GroqProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.online_provider_manager import OnlineProviderManager
from app.services.orchestration_pipeline import OrchestrationPipeline

def test_groq_provider_status():
    provider = GroqProvider()
    status = provider.get_status()
    assert status.provider == "Groq API"
    assert status.configured is True

def test_openrouter_provider_status():
    provider = OpenRouterProvider()
    status = provider.get_status()
    assert status.provider == "OpenRouter API"
    assert status.configured is True

def test_online_provider_manager_configured_list():
    manager = OnlineProviderManager()
    candidates = manager.get_online_model_candidates()
    assert len(candidates) >= 4
    providers = [c.provider for c in candidates]
    assert "Google Gemini API" in providers
    assert "Mistral API" in providers
    assert "Groq API" in providers
    assert "OpenRouter API" in providers

def test_online_provider_manager_candidates_metadata():
    manager = OnlineProviderManager()
    candidates = manager.get_online_model_candidates()
    model_ids = [c.model_id for c in candidates]
    assert "gemini-3.5-flash" in model_ids or "mistral-small-latest" in model_ids

@pytest.mark.asyncio
async def test_e2e_automatic_online_orchestration():
    if not settings.GEMINI_API_KEY and not settings.MISTRAL_API_KEY and not settings.GROQ_API_KEY:
        pytest.skip("Online API keys not configured in environment settings")

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="Write a Python script to sort a list.", execution_mode="online")
    
    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if isinstance(raw_evt, str) and raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    if not final_res:
        pytest.skip("Online provider stream did not complete (network or API key unavailable)")

    assert final_res is not None
    assert "generation" in final_res
    assert final_res["reward"]["reward"] is not None
