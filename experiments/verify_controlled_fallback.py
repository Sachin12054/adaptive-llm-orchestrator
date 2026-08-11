import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def verify_controlled_fallback():
    print("\n==================================================")
    print("  VERIFYING CONTROLLED PROVIDER FALLBACK VIA BASELINE POLICY")
    print("==================================================")

    pipeline = OrchestrationPipeline()
    prompt = "What is the capital of France?"

    # We patch GeminiProvider.generate temporarily to simulate an API key error for gemini-2.5-flash
    orig_gemini_gen = pipeline.response_generator.model_manager.online_manager.providers["gemini"].generate

    def mock_failing_gemini(req):
        if req.model_id == "gemini-2.5-flash":
            print(f"\n[FALLBACK SIMULATION] Simulating provider failure for candidate model '{req.model_id}'...")
            from app.schemas.provider import ProviderGenerationResponse
            return ProviderGenerationResponse(
                provider="Google Gemini API",
                model_id=req.model_id,
                generated_text=None,
                latency_ms=15.0,
                success=False,
                error_message="Simulated 503 Provider Unavailable Error"
            )
        return orig_gemini_gen(req)

    pipeline.response_generator.model_manager.online_manager.providers["gemini"].generate = mock_failing_gemini

    req = OrchestrationRequest(prompt=prompt, execution_mode="online")
    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            events.append(json.loads(raw_evt[6:].strip()))

    pipeline.response_generator.model_manager.online_manager.providers["gemini"].generate = orig_gemini_gen

    final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
    assert final_evt is not None, "final_response event MUST be emitted!"

    payload = final_evt["payload"]
    gen = payload["generation"]
    rwd = payload["reward"]
    ver = payload["verification"]

    print(f"\nFALLBACK EXECUTION RESULTS:")
    print(f" - Primary Model 'gemini-2.5-flash' failed & excluded.")
    print(f" - BaselineAdaptivePolicy Re-evaluated Candidates.")
    print(f" - Fallback Selected Model: {payload['selected_model']}")
    print(f" - Fallback Provider: {gen['provider']}")
    print(f" - Fallback Generation Latency: {gen['latency_ms']} ms")
    print(f" - Response Text Length: {len(gen['generated_text'] if gen['generated_text'] else '')}")
    print(f" - Verification Status: {ver['verification_status']}")
    print(f" - Step 18 Reward: {rwd['reward']:.4f}")

    assert payload['selected_model'] != "gemini-2.5-flash", "Primary failing model MUST be excluded!"
    assert gen['generated_text'] is not None and len(gen['generated_text'].strip()) > 10, "Fallback candidate MUST produce valid response text!"
    assert ver['verified'] == True, "Fallback response MUST pass verification!"
    assert rwd['reward'] > 0.0, "Fallback response MUST yield non-zero Step 18 reward!"

    print("\n=> CONTROLLED PROVIDER FALLBACK VERIFIED SUCCESSFULLY!")

def main():
    asyncio.run(verify_controlled_fallback())

if __name__ == "__main__":
    main()
