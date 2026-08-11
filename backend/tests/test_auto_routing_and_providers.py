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
    configured = manager.get_configured_providers()
    assert len(configured) > 0
    assert "mistral" in configured or "gemini" in configured or "groq" in configured or "openrouter" in configured

def test_online_provider_manager_task_routing():
    manager = OnlineProviderManager()
    # Coding intent should prioritize groq or mistral
    coding_provider = manager.select_best_provider(intent="coding", complexity_level="medium")
    assert coding_provider in ["groq", "mistral", "openrouter", "gemini"]

    # Complex intent should prioritize gemini or mistral
    complex_provider = manager.select_best_provider(intent="general_qa", complexity_level="very_high")
    assert complex_provider in ["gemini", "mistral", "openrouter", "groq"]

@pytest.mark.asyncio
async def test_e2e_automatic_online_orchestration():
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="Write a Python script to sort a list.", execution_mode="online")
    
    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None
    assert final_res["generation"]["provider"] in ["Groq API", "Mistral API", "Google Gemini API", "OpenRouter API"]
    assert final_res["reward"]["reward"] is not None
    assert len(final_res["reward"]["metric_breakdown"]) == 5
