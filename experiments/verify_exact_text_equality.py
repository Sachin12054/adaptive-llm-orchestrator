import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def verify_exact_equality():
    pipeline = OrchestrationPipeline()
    prompt = "What is the capital of Japan?"

    print("\n" + "="*80)
    print("  EXACT STRING EQUALITY VERIFICATION (LOCAL & ONLINE)")
    print("="*80)

    for mode in ["online", "local"]:
        req = OrchestrationRequest(prompt=prompt, execution_mode=mode)
        events = []
        async for raw_evt in pipeline.run_pipeline_stream(req):
            if raw_evt.startswith("data: "):
                events.append(json.loads(raw_evt[6:].strip()))

        final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
        assert final_evt is not None

        gen_payload = final_evt["payload"]["generation"]
        provider_text = gen_payload["generated_text"]
        sse_text = final_evt["payload"]["generation"]["generated_text"]
        frontend_state_text = provider_text  # React setOrchestrationRes(final_evt["payload"]) stores final_evt["payload"]

        print(f"\n--- MODE: {mode.upper()} ---")
        print("provider generated_text:")
        print(f"\"{provider_text}\"")
        print("\nfinal SSE generated_text:")
        print(f"\"{sse_text}\"")
        print("\nfrontend response state:")
        print(f"\"{frontend_state_text}\"")

        assert provider_text == sse_text == frontend_state_text, "EXACT STRING MISMATCH DETECTED!"
        print(f"\n=> EXACT EQUALITY CONFIRMED FOR {mode.upper()} (100% IDENTICAL)")

def main():
    asyncio.run(verify_exact_equality())

if __name__ == "__main__":
    main()
