import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.response_generator import ResponseGenerator
from validation.strategies.fixed_model_policy import FixedModelPolicy
from validation.strategies.random_policy import RandomPolicy
from validation.strategies.round_robin_policy import RoundRobinPolicy
from validation.strategies.capability_heuristic_policy import CapabilityHeuristicPolicy
from validation.strategies.baseline_adaptive_policy_adapter import BaselineAdaptivePolicyAdapter
from validation.metrics.quality_metrics import QualityMetricsEvaluator
from validation.metrics.cost_metrics import CostMetricsEvaluator
from app.schemas.response import ResponseGenerationRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validation")

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "benchmark_dataset.jsonl"))
RAW_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))

def load_benchmark_dataset() -> List[Dict[str, Any]]:
    prompts = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

def run_live_strategy_benchmark(
    strategy_id: str,
    prompts: List[Dict[str, Any]],
    dry_run: bool = False,
    resume: bool = True
) -> List[Dict[str, Any]]:
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RAW_RESULTS_DIR, f"raw_results_strategy_{strategy_id}.jsonl")
    
    existing_records = []
    processed_ids = set()
    if resume and os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    processed_ids.add(rec["prompt_id"])
        logger.info(f"Resuming Strategy {strategy_id}: {len(processed_ids)} prompts already recorded.")

    # Initialize policy strategy
    if strategy_id == "A":
        policy = FixedModelPolicy("gemma-3-4b", "Local Ollama")
    elif strategy_id == "B":
        policy = FixedModelPolicy("qwen-coder-3b", "Local Ollama")
    elif strategy_id == "C":
        policy = FixedModelPolicy("deepseek-r1-7b", "Local Ollama")
    elif strategy_id == "D":
        policy = RandomPolicy(seed=42)
    elif strategy_id == "E":
        policy = RoundRobinPolicy()
    elif strategy_id == "F":
        policy = CapabilityHeuristicPolicy()
    elif strategy_id == "G":
        policy = BaselineAdaptivePolicyAdapter()
    else:
        raise ValueError(f"Unknown live strategy_id: {strategy_id}")

    response_gen = ResponseGenerator()
    results = list(existing_records)

    with open(out_file, "a" if resume else "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(prompts, start=1):
            p_id = item["prompt_id"]
            if p_id in processed_ids:
                continue

            prompt_text = item["prompt"]
            cat = item["category"]
            diff = item["difficulty"]

            logger.info(f"[{strategy_id}] ({idx}/{len(prompts)}) Processing '{p_id}' ({cat})...")

            dec = policy.decide(prompt_text, cat, execution_mode="online")
            model_id = dec["selected_model"]
            provider = dec["provider"]
            dec_lat_ms = dec.get("decision_latency_ms", 1.0)

            if dry_run:
                gen_text = f"[DRY RUN SYNTHESIZED RESPONSE FOR {p_id} using {model_id}]"
                lat_ms = 120.0
                success = True
                failover = False
                in_tok = len(prompt_text) // 4
                out_tok = len(gen_text) // 4
            else:
                t0 = time.perf_counter()
                gen_req = ResponseGenerationRequest(
                    prompt=prompt_text,
                    selected_model=model_id,
                    execution_mode="online",
                    max_output_tokens=600
                )
                
                # For fixed strategies (A, B, C), disable provider failover to preserve scientific strategy definition
                if strategy_id in ["A", "B", "C"]:
                    # Pass candidate constraints to prevent failover to other providers
                    gen_req.execution_mode = "local" if provider == "Local Ollama" else "online"

                gen_res = response_gen.generate_response(gen_req)
                lat_ms = round((time.perf_counter() - t0) * 1000, 2)
                
                # Check for fixed strategy provider mismatch (if failover occurred when disallowed)
                if strategy_id in ["A", "B", "C"] and gen_res.failover_used and gen_res.model_id != model_id:
                    # Failover occurred on fixed strategy -> mark as NOT MEASURED / Provider Failure for fixed policy
                    success = False
                    failover = True
                    gen_text = ""
                    logger.warning(f"[{strategy_id}] Fixed policy model '{model_id}' failed and was failover-rerouted to '{gen_res.model_id}'; recording NOT_MEASURED for fixed strategy.")
                else:
                    success = gen_res.success
                    gen_text = gen_res.generated_text if success else ""
                    failover = getattr(gen_res, "failover_used", False)
                    if getattr(gen_res, "model_id", None):
                        model_id = gen_res.model_id
                    if getattr(gen_res, "provider", None):
                        provider = gen_res.provider

                in_tok = gen_res.input_tokens if hasattr(gen_res, "input_tokens") and gen_res.input_tokens else len(prompt_text) // 4
                out_tok = gen_res.output_tokens if hasattr(gen_res, "output_tokens") and gen_res.output_tokens else len(gen_text) // 4

            # Evaluate Quality on ACTUAL generated text
            q_eval = QualityMetricsEvaluator.evaluate_quality(prompt_text, cat, gen_text, success=success)

            # Evaluate Cost
            c_eval = CostMetricsEvaluator.calculate_cost(provider, in_tok, out_tok)

            record = {
                "experiment_id": f"EXP_{strategy_id}_{p_id}",
                "strategy": strategy_id,
                "prompt_id": p_id,
                "original_prompt": prompt_text,
                "task_category": cat,
                "difficulty": diff,
                "selected_model": model_id,
                "provider": provider,
                "routing_decision_latency_ms": dec_lat_ms,
                "model_execution_latency_ms": lat_ms,
                "total_e2e_latency_ms": round(dec_lat_ms + lat_ms, 2),
                "total_latency_ms": round(dec_lat_ms + lat_ms, 2),  # Backward compatibility key
                "success": success,
                "failover_used": failover,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_cost_usd": c_eval["total_cost_usd"],
                "score_0_to_5": q_eval["score_0_to_5"],
                "normalized_quality": q_eval["normalized_quality"],
                "rubric_level": q_eval["rubric_level"],
                "response_text_snippet": gen_text[:200] if gen_text else "",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            results.append(record)

    logger.info(f"Finished Strategy {strategy_id} execution: {len(results)} total records.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live Strategy Benchmark (A-G)")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy ID (A, B, C, D, E, F, or G)")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without calling LLM APIs")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume; overwrite raw logs")
    args = parser.parse_args()

    prompts = load_benchmark_dataset()
    run_live_strategy_benchmark(args.strategy, prompts, dry_run=args.dry_run, resume=not args.no_resume)
