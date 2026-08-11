import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

def run_final_audit():
    print("=" * 80)
    print(" FINAL E2E GPU VRAM ROUTING AUDIT")
    print("=" * 80)

    prompts = [
        "What is Python?",
        "Write Python code to read a CSV using pandas.",
        "Solve a difficult multi-step mathematical problem involving probability."
    ]

    pipeline = OrchestrationPipeline()

    for idx, text in enumerate(prompts, start=1):
        print(f"\n--- TEST #{idx}: \"{text}\" ---")
        t0 = time.perf_counter()
        res = pipeline.run_pipeline(OrchestrationRequest(prompt=text))
        t1 = time.perf_counter()

        total_latency = round((t1 - t0) * 1000, 2)
        dec = res.decision
        gen = res.generation
        rew = res.reward

        res_sum = dec.decision_trace.resource_summary
        free_vram = res_sum.get("gpu_free_vram_gb")
        used_vram = res_sum.get("gpu_used_vram_gb")
        total_vram = res_sum.get("gpu_total_vram_gb")

        print(f"  Selected Model               : {res.selected_model}")
        print(f"  GPU VRAM (Free/Used/Total)   : {free_vram} GB / {used_vram} GB / {total_vram} GB")
        print(f"  Candidate Breakdown & Eligibility:")
        for c in dec.candidates:
            elig_str = "Eligible" if c.eligible else f"Ineligible ({c.ineligible_reason})"
            print(f"    - {c.model_id:<15}: {elig_str:<65} | res_fit={c.resource_fit_score:.2f} | total={c.candidate_score:.4f}")
        print(f"  Ollama Generation Latency    : {gen.latency_ms} ms")
        print(f"  Total Pipeline Latency       : {total_latency} ms")
        print(f"  Step 18 Calculated Reward    : {rew.reward:.4f}")

    print("\n" + "=" * 80)
    print(" AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_final_audit()
