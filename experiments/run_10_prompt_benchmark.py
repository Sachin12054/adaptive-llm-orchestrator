import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.schemas.orchestration import OrchestrationRequest
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.intent_classifier import IntentClassifier
from app.services.complexity_analyzer import ComplexityAnalyzer

def main():
    pipeline = OrchestrationPipeline()
    intent_cls = IntentClassifier()
    comp_analyzer = ComplexityAnalyzer()

    prompts = [
        "What is Python?",
        "What is a Database?",
        "What is AI?",
        "Explain Python.",
        "Define machine learning.",
        "What is recursion?",
        "Write Python code to read a CSV using pandas.",
        "Implement binary search in Python and explain its time complexity.",
        "Design a distributed Python ML pipeline for millions of records per day using Kafka, Spark, PostgreSQL, object storage, model training, versioning, monitoring, and fault tolerance.",
        "Make it better."
    ]

    print("=" * 80)
    print("PROD HARDENING BENCHMARK: 10 PROMPT AUDIT")
    print("=" * 80)

    for idx, p in enumerate(prompts, 1):
        print(f"\n[{idx}/10] PROMPT: '{p}'")
        intent_res = intent_cls.classify_intent(p)
        comp_res = comp_analyzer.analyze_complexity(p)

        print(f"  Intent:        {intent_res.intent} (top_sim={intent_res.top_similarity:.4f}, margin={intent_res.margin:.4f}, ambiguous={intent_res.is_ambiguous})")
        print(f"  Complexity:    {comp_res.complexity_level} (score={comp_res.complexity_score:.4f})")

        req = OrchestrationRequest(prompt=p)
        orch_res = pipeline.run_pipeline(req)

        print(f"  Selected Model: {orch_res.selected_model}")
        print(f"  Decision Score: {orch_res.decision_score:.4f}")
        print(f"  Step 18 Reward: {orch_res.reward.reward:.4f}")

        # Check candidate score breakdowns
        cands = orch_res.decision.candidates or []
        print("  Candidates:")
        for c in cands:
            status = "SELECTED" if c.model_id == orch_res.selected_model else ("ELIGIBLE" if c.eligible else "INELIGIBLE")
            print(f"    - {c.model_id:15s} [{status:10s}] Score: {c.candidate_score:.4f} (Cap: {c.capability_score:.2f}, Cmplx: {c.complexity_fit_score:.2f}, Res: {c.resource_fit_score:.2f}, Ctx: {c.context_fit_score:.2f})")
            if not c.eligible and c.ineligible_reason:
                print(f"      Ineligible Reason: {c.ineligible_reason[:90]}...")

        # Generated Text summary
        gen_text = orch_res.generation.generated_text or ""
        text_len = len(gen_text)
        has_think = "<think>" in gen_text
        print(f"  Output Length:  {text_len} chars | Contains <think> tag: {has_think}")
        snippet = gen_text.replace('\n', ' ')[:100]
        print(f"  Snippet:        '{snippet}...'")

if __name__ == "__main__":
    main()
