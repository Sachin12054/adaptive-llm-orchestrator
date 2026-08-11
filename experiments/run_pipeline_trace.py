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

def run_trace():
    prompt = "Design a distributed Python ML pipeline..."
    print("=" * 80)
    print(f" PIPELINE TRACE FOR: \"{prompt}\"")
    print("=" * 80)

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))

    dec = res.decision
    trace = dec.decision_trace

    print(f"Intent            : {trace.intent}")
    print(f"Is Ambiguous      : {trace.is_ambiguous}")
    print(f"Complexity Level  : {trace.complexity_level.upper()}")
    print(f"Complexity Score  : {trace.complexity_score:.4f}")
    print(f"Selected Model    : {res.selected_model}")
    print(f"Winning Score     : {res.decision_score:.4f}\n")

    print(f"{'Model ID':<15} | {'Eligible':<8} | {'Cap Score':<9} | {'Cmplx Fit':<9} | {'Res Fit':<8} | {'Context':<7} | {'Total Score':<11} | Ineligible Reason")
    print("-" * 105)
    for c in dec.candidates:
        inelig = c.ineligible_reason or "None"
        print(f"{c.model_id:<15} | {str(c.eligible):<8} | {c.capability_score:>9.4f} | {c.complexity_fit_score:>9.4f} | {c.resource_fit_score:>8.4f} | {c.context_fit_score:>7.4f} | {c.candidate_score:>11.4f} | {inelig}")

if __name__ == "__main__":
    run_trace()
