import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

def run_four_prompts():
    print("=" * 80)
    print(" FOUR-PROMPT MODEL SELECTION & CANDIDATE BREAKDOWN AUDIT")
    print("=" * 80)

    prompts = [
        ("SIMPLE QA", "What is Python?"),
        ("CSV CODING", "Write Python code to read a CSV using pandas."),
        ("COMPLEX MATH", "Solve a difficult multi-step mathematical problem involving probability."),
        ("ML PIPELINE", "Design a distributed Python ML pipeline...")
    ]

    pipeline = OrchestrationPipeline()

    for label, text in prompts:
        print(f"\n--- [{label}]: \"{text}\" ---")
        res = pipeline.run_pipeline(OrchestrationRequest(prompt=text))
        dec = res.decision
        trace = dec.decision_trace

        res_sum = trace.resource_summary
        free_vram = res_sum.get("gpu_free_vram_gb")
        used_vram = res_sum.get("gpu_used_vram_gb")
        total_vram = res_sum.get("gpu_total_vram_gb")

        print(f"  Selected Model        : {res.selected_model}")
        print(f"  Winning Candidate Score: {res.decision_score:.4f}")
        print(f"  Intent (Ambiguous)    : {trace.intent} (ambiguous={trace.is_ambiguous})")
        print(f"  Complexity Level/Score: {trace.complexity_level.upper()} ({trace.complexity_score:.4f})")
        print(f"  GPU VRAM Telemetry    : Free {free_vram} GB / Used {used_vram} GB / Total {total_vram} GB")
        print(f"  Candidate Scoring Breakdown:")
        for c in dec.candidates:
            elig = "Eligible" if c.eligible else f"Ineligible ({c.ineligible_reason})"
            print(f"    - {c.model_id:<15}: {elig:<65} | cap={c.capability_score:.2f}, cmplx={c.complexity_fit_score:.2f}, res={c.resource_fit_score:.2f}, ctx={c.context_fit_score:.2f} -> total={c.candidate_score:.4f}")

    print("\n" + "=" * 80)
    print(" AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_four_prompts()
