import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline

async def verify_stage_naming(test_id: str, prompt: str, execution_mode: str, expected_stage: str):
    print(f"\n==================================================")
    print(f"  {test_id} (Mode: '{execution_mode.upper()}')")
    print(f"==================================================")

    pipeline = OrchestrationPipeline()
    req = OrchestrationRequest(prompt=prompt, execution_mode=execution_mode)

    events = []
    async for raw_evt in pipeline.run_pipeline_stream(req):
        if raw_evt.startswith("data: "):
            events.append(json.loads(raw_evt[6:].strip()))

    stage5_running_evt = next((e for e in events if e.get("stage") == expected_stage and e.get("status") == "running"), None)
    stage5_completed_evt = next((e for e in events if e.get("stage") == expected_stage and e.get("status") == "completed"), None)

    assert stage5_running_evt is not None, f"Stage 5 running event for stage '{expected_stage}' MUST exist!"
    assert stage5_completed_evt is not None, f"Stage 5 completed event for stage '{expected_stage}' MUST exist!"

    run_meta = stage5_running_evt.get("metadata", {})
    comp_meta = stage5_completed_evt.get("metadata", {})

    print(f"\nObserved SSE Event Sequence for Stage 5:")
    print(f" - Running Event Stage: '{stage5_running_evt.get('stage')}' | Metadata: {run_meta}")
    print(f" - Completed Event Stage: '{stage5_completed_evt.get('stage')}' | Metadata: {comp_meta}")

    assert run_meta.get("execution_mode") == execution_mode, f"Running metadata execution_mode must be '{execution_mode}'"
    assert comp_meta.get("execution_mode") == execution_mode, f"Completed metadata execution_mode must be '{execution_mode}'"
    assert comp_meta.get("provider") is not None, "Provider must be populated in metadata"
    assert comp_meta.get("model") is not None, "Model must be populated in metadata"

    # Confirm that 'local_inference' NEVER appears during online mode
    if execution_mode == "online":
        assert not any(e.get("stage") == "local_inference" for e in events), "'local_inference' stage MUST NOT appear when execution_mode is 'online'!"

    print(f"\n=> {test_id} PASSED SUCCESSFULLY!\n")

def main():
    asyncio.run(verify_stage_naming("TEST A — LOCAL INFERENCE STAGE", "What is the capital of France?", "local", "local_inference"))
    asyncio.run(verify_stage_naming("TEST B — ONLINE INFERENCE STAGE", "Explain how TCP congestion control works.", "online", "online_inference"))

if __name__ == "__main__":
    main()
