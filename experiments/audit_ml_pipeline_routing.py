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

def audit_pipeline():
    prompt = "Design a distributed Python ML pipeline..."
    print("=" * 80)
    print(f" AUDITING ROUTING FOR: \"{prompt}\"")
    print("=" * 80)

    pipeline = OrchestrationPipeline()
    res = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))

    dec = res.decision
    trace = dec.decision_trace

    print(f"\n1. Overall Decision Trace:")
    print(f"   Intent Detected       : {trace.intent}")
    print(f"   Is Ambiguous          : {trace.is_ambiguous}")
    print(f"   Complexity Level      : {trace.complexity_level.upper()}")
    print(f"   Complexity Score      : {trace.complexity_score:.4f}")
    print(f"   Selected Model        : {res.selected_model}")
    print(f"   Winning Decision Score: {res.decision_score:.4f}")
    print(f"   Shadow RL Decision    : {trace.shadow_rl_decision.get('proposed_model') if trace.shadow_rl_decision else 'N/A'}")

    print(f"\n2. Candidate Models Breakdown:")
    for c in dec.candidates:
        print(f"   Model: {c.model_id:<15}")
        print(f"     - Eligible          : {c.eligible} ({c.ineligible_reason or 'OK'})")
        print(f"     - Capability Score  : {c.capability_score:.4f}")
        print(f"     - Complexity Fit    : {c.complexity_fit_score:.4f}")
        print(f"     - Resource Fit      : {c.resource_fit_score:.4f}")
        print(f"     - Context Fit       : {c.context_fit_score:.4f}")
        print(f"     - Candidate Score   : {c.candidate_score:.4f}")
        print()

if __name__ == "__main__":
    audit_pipeline()
