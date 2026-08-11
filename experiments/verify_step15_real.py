import os
import sys
import json
import time

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.decision import DecisionRequest

def run_step15_real_verification():
    print("=" * 80)
    print(" STEP 15 REAL ADAPTIVE DECISION ENGINE VERIFICATION PASS (BASELINE ADAPTIVE POLICY)")
    print("=" * 80)

    engine = AdaptiveDecisionEngine()
    status_resp = engine.get_status()

    print("\n--- 1. DECISION ENGINE STATUS & POLICY OVERVIEW ---")
    print(f"Engine Status               : {status_resp.status.upper()}")
    print(f"Active Policy               : {status_resp.policy}")
    print(f"Registered Models Count     : {status_resp.registered_models_count}")
    print(f"Executable Candidates Count : {status_resp.executable_candidates_count}")
    print(f"Executable Model IDs        : {', '.join(status_resp.executable_models) if status_resp.executable_models else 'None (GEMINI_API_KEY unpopulated)'}")

    test_cases = [
        (
            "CASE 1: Simple Factual",
            "What is the capital of France?"
        ),
        (
            "CASE 2: Moderate Coding",
            "Write a Python function to reverse a linked list."
        ),
        (
            "CASE 3: High Reasoning Comparison",
            "Compare PostgreSQL and MongoDB for a telemetry platform and recommend one based on scalability, consistency, and operational complexity."
        ),
        (
            "CASE 4: Very High Architecture Design",
            "Design a complete scalable architecture for processing millions of telemetry events per second, compare database and messaging alternatives, explain trade-offs, and provide an implementation strategy."
        )
    ]

    verification_results = []

    print("\n--- 2. REAL ADAPTIVE DECISION EVALUATION (4 REPRESENTATIVE PROMPTS) ---")

    for idx, (label, prompt) in enumerate(test_cases, 1):
        print(f"\n[{idx}/4] {label}")
        print(f"     Prompt: \"{prompt}\"")

        req = DecisionRequest(text=prompt)
        t0 = time.perf_counter()
        res = engine.decide(req)
        t1 = time.perf_counter()

        latency_ms = round((t1 - t0) * 1000, 2)
        trace = res.decision_trace

        print(f"  Intent Analyzed     : {trace.intent.upper()} (is_ambiguous={trace.is_ambiguous})")
        print(f"  Complexity Analyzed : {trace.complexity_level.upper()} (score={trace.complexity_score:.4f})")
        print(f"  Resource Summary    : GPU={trace.resource_summary.get('gpu_available')}, RAM_util={trace.resource_summary.get('memory_utilization_percent')}%, CPU_util={trace.resource_summary.get('cpu_utilization_percent')}%")
        print(f"  Candidate Scores:")
        for cand in res.candidates:
            elig_str = "Eligible" if cand.eligible else f"Ineligible ({cand.ineligible_reason})"
            print(f"    - Model: {cand.model_id:<20} | Score: {cand.candidate_score:.4f} | Status: {elig_str}")
        
        print(f"  Selected Model      : {res.selected_model or 'None'}")
        print(f"  Winning Score       : {res.decision_score:.4f}")
        print(f"  Policy Used         : {res.policy}")
        print(f"  Pipeline Latency    : {res.total_pipeline_latency_ms} ms (Decision Policy: {trace.decision_latency_ms} ms)")
        print("  Decision Rationale  :")
        for bullet in res.reasoning:
            print(f"    • {bullet}")

        verification_results.append({
            "case_label": label,
            "prompt": prompt,
            "decision_response": res.dict()
        })

    # Save verification report
    report_output_path = os.path.join(project_root, "data", "logs", "step15_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "status_summary": status_resp.dict(),
            "cases": verification_results
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Verification output written to '{report_output_path}'.")
    print("STATEMENT: This implementation uses a transparent BaselineAdaptivePolicy and does NOT claim to be a trained RL policy.")
    print("=" * 80)

if __name__ == "__main__":
    run_step15_real_verification()
