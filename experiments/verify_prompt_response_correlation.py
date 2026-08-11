import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def verify_prompts():
    pipeline = OrchestrationPipeline()

    test_prompts = [
        ("TEST 1 (US Bird)", "What is the national bird of the United States?"),
        ("TEST 2 (Japan Capital)", "What is the capital of Japan?"),
        ("TEST 3 (France Capital)", "What is the capital of France?"),
        ("TEST 4 (TCP Control)", "Explain how TCP congestion control works in simple terms."),
        ("TEST 5 (Complex IoT)", "Compare REST APIs, GraphQL, and gRPC and recommend one for a real-time IoT telemetry platform.")
    ]

    print("\n" + "="*100)
    print("  EXACT PROMPT-RESPONSE CORRELATION & MODEL IDENTITY AUDIT")
    print("="*100)

    for label, p in test_prompts:
        run_id_input = f"run_test_{int(asyncio.get_event_loop().time()*1000)}"
        req = OrchestrationRequest(prompt=p, execution_mode="online", run_id=run_id_input)
        
        events = []
        async for raw_evt in pipeline.run_pipeline_stream(req):
            if raw_evt.startswith("data: "):
                events.append(json.loads(raw_evt[6:].strip()))

        final_evt = next((e for e in events if e.get("stage") == "final_response"), None)
        assert final_evt is not None, f"No final_response event for {label}"

        payload = final_evt["payload"]
        gen = payload["generation"]

        print(f"\n[{label}]")
        print(f" [RUN]             run_id={payload.get('run_id')}")
        print(f" [PROMPT]          request_prompt=\"{p}\"")
        print(f" [POLICY]          selected_model={payload['selected_model']}")
        print(f" [TASK ALLOCATION] assigned_model={payload['selected_model']}")
        print(f" [DISPATCH]        dispatched_model={gen['model_id']} | provider={gen['provider']}")
        print(f" [GENERATION]      model_id={gen['model_id']}")
        print(f" [RESPONSE]        text_length={len(gen['generated_text'] or '')}")
        print(f" [FINAL PAYLOAD]   model_id={gen['model_id']} | text_sample=\"{(gen['generated_text'] or '')[:120].strip()}...\"")

        assert payload["selected_model"] == gen["model_id"], "Model identity mismatch!"
        assert payload["prompt"] == p, "Prompt correlation mismatch!"
        assert gen["generated_text"] is not None and len(gen["generated_text"].strip()) > 0, "Empty generated text!"

        if "national bird" in p.lower():
            assert "eagle" in gen["generated_text"].lower() or "bald" in gen["generated_text"].lower(), f"Expected Bald Eagle in US Bird response, got: {gen['generated_text']}"
        elif "japan" in p.lower():
            assert "tokyo" in gen["generated_text"].lower(), f"Expected Tokyo in Japan Capital response, got: {gen['generated_text']}"
        elif "france" in p.lower():
            assert "paris" in gen["generated_text"].lower(), f"Expected Paris in France Capital response, got: {gen['generated_text']}"

        print(f" => {label} PASSED 100%! Prompt-Response correlation verified.")

def main():
    asyncio.run(verify_prompts())

if __name__ == "__main__":
    main()
