import os
import sys
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

def run_five_prompts():
    print("=" * 80)
    print(" FIVE PROMPTS CONTRACT AUDIT & RESPONSE VERIFICATION")
    print("=" * 80)

    prompts = [
        "What is Python?",
        "What is the difference between Python lists and tuples?",
        "Write Python code to read a CSV using pandas.",
        "Design a distributed Python ML pipeline...",
        "Explain how Python garbage collection works"
    ]

    pipeline = OrchestrationPipeline()

    for idx, text in enumerate(prompts, start=1):
        print(f"\n--- PROMPT #{idx}: \"{text}\" ---")
        res = pipeline.run_pipeline(OrchestrationRequest(prompt=text))
        dec = res.decision
        trace = dec.decision_trace

        res_sum = trace.resource_summary
        free_vram = res_sum.get("gpu_free_vram_gb")
        used_vram = res_sum.get("gpu_used_vram_gb")
        total_vram = res_sum.get("gpu_total_vram_gb")

        intent_info = dec.intent_info
        complexity_info = dec.complexity_info
        buffer_rec = pipeline.experience_buffer._buffer[-1]

        print(f"  Selected Model        : {res.selected_model}")
        print(f"  Winning Candidate Score: {res.decision_score:.4f}")
        print(f"  Intent Info           : {intent_info}")
        print(f"  Complexity Info       : {complexity_info}")
        print(f"  GPU VRAM Telemetry    : Free {free_vram} GB / Used {used_vram} GB / Total {total_vram} GB")
        print(f"  Step 18 Calculated Rew: {res.reward.reward:.4f}")
        print(f"  Step 20 State Vector  : Dim = {len(buffer_rec.state)} ({buffer_rec.state[:4]}...)")
        print(f"  Candidate Breakdown & Eligibility:")
        for c in dec.candidates:
            elig_str = "Eligible" if c.eligible else f"Ineligible ({c.ineligible_reason})"
            print(f"    - {c.model_id:<15}: {elig_str:<65} | total={c.candidate_score:.4f}")

    print("\n" + "=" * 80)
    print(" CONTRACT AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_five_prompts()
