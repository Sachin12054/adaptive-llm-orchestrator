import pytest
import os
from app.core.config import settings
from app.services.providers.mistral_provider import MistralProvider
from app.schemas.provider import ProviderGenerationRequest
from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

def test_mistral_provider_status():
    provider = MistralProvider()
    status = provider.get_status()
    assert status.provider == "Mistral API"
    assert "mistral-small-latest" in status.models

def test_mistral_provider_missing_key():
    provider = MistralProvider(api_key="")
    req = ProviderGenerationRequest(prompt="Hello", model_id="mistral-small-latest")
    resp = provider.generate(req)
    assert resp.success is False
    assert "MISTRAL_API_KEY is not configured" in resp.error_message

def test_mistral_provider_generation_with_key():
    api_key = settings.MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not configured in environment")
        
    provider = MistralProvider(api_key=api_key)
    req = ProviderGenerationRequest(prompt="What is 2+2?", model_id="mistral-small-latest")
    resp = provider.generate(req)
    assert resp.success is True
    assert resp.generated_text is not None
    assert len(resp.generated_text) > 0
    assert resp.latency_ms > 0

@pytest.mark.asyncio
async def test_e2e_orchestration_mistral_mode():
    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt="What is Python?", execution_mode="mistral")
    
    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            import json
            payload = json.loads(raw_evt[6:].strip())
            events.append(payload)

    final_res = next((e.get("payload") for e in events if e.get("stage") == "final_response"), None)
    assert final_res is not None
    assert final_res["generation"]["provider"] == "Mistral API"
    assert final_res["reward"]["reward"] is not None
    assert len(final_res["reward"]["metric_breakdown"]) == 5
