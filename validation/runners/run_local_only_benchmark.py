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
logger = logging.getLogger("validation_local_only")

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "benchmark_dataset.jsonl"))
LOCAL_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "local_only"))
RAW_RESULTS_DIR = os.path.join(LOCAL_RESULTS_DIR, "raw")

LOCAL_MODELS = ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]

def load_benchmark_dataset() -> List[Dict[str, Any]]:
    prompts = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

class LocalOnlyCapabilityHeuristicPolicy:
    """Strategy F constrained ONLY to local models."""
    def decide(self, prompt: str, category: str = "", execution_mode: str = "local") -> Dict[str, Any]:
        t0 = time.perf_counter()
        cat_lower = category.lower()
        if "coding" in cat_lower or "math" in cat_lower or "structured" in cat_lower:
            selected_model = "qwen-coder-3b"
        elif "reasoning" in cat_lower or "multi-objective" in cat_lower or "complex" in cat_lower or "analysis" in cat_lower:
            selected_model = "deepseek-r1-7b"
        else:
            selected_model = "gemma-3-4b"
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "selected_model": selected_model,
            "provider": "Local Ollama",
            "decision_latency_ms": dt_ms,
            "weighted_score": 0.85,
            "strategy": "LocalOnlyCapabilityHeuristicPolicy"
        }

class LocalOnlyRandomPolicy:
    """Strategy D constrained ONLY to local models (seed=42)."""
    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)

    def decide(self, prompt: str, category: str = "", execution_mode: str = "local") -> Dict[str, Any]:
        t0 = time.perf_counter()
        selected_model = self.rng.choice(LOCAL_MODELS)
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "selected_model": selected_model,
            "provider": "Local Ollama",
            "decision_latency_ms": dt_ms,
            "weighted_score": 0.50,
            "strategy": "LocalOnlyRandomPolicy"
        }

class LocalOnlyRoundRobinPolicy:
    """Strategy E constrained ONLY to local models."""
    def __init__(self):
        self.idx = 0

    def decide(self, prompt: str, category: str = "", execution_mode: str = "local") -> Dict[str, Any]:
        t0 = time.perf_counter()
        selected_model = LOCAL_MODELS[self.idx % len(LOCAL_MODELS)]
        self.idx += 1
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "selected_model": selected_model,
            "provider": "Local Ollama",
            "decision_latency_ms": dt_ms,
            "weighted_score": 0.50,
            "strategy": "LocalOnlyRoundRobinPolicy"
        }

def run_local_only_strategy_benchmark(
    strategy_id: str,
    prompts: List[Dict[str, Any]],
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
        logger.info(f"Resuming LOCAL_ONLY_V1 Strategy {strategy_id}: {len(processed_ids)} prompts recorded.")

    if strategy_id == "A":
        policy = FixedModelPolicy("gemma-3-4b", "Local Ollama")
    elif strategy_id == "B":
        policy = FixedModelPolicy("qwen-coder-3b", "Local Ollama")
    elif strategy_id == "C":
        policy = FixedModelPolicy("deepseek-r1-7b", "Local Ollama")
    elif strategy_id == "D":
        policy = LocalOnlyRandomPolicy(seed=42)
    elif strategy_id == "E":
        policy = LocalOnlyRoundRobinPolicy()
    elif strategy_id == "F":
        policy = LocalOnlyCapabilityHeuristicPolicy()
    elif strategy_id == "G":
        policy = BaselineAdaptivePolicyAdapter()
    else:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")

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

            logger.info(f"[LOCAL_ONLY_V1 - {strategy_id}] ({idx}/{len(prompts)}) Processing '{p_id}' ({cat})...")

            # Route decision constrained strictly to local execution mode
            dec = policy.decide(prompt_text, cat, execution_mode="local")
            model_id = dec["selected_model"]
            if model_id not in LOCAL_MODELS:
                model_id = "gemma-3-4b"  # Fallback to local gemma if unmapped

            provider = "Local Ollama"
            dec_lat_ms = dec.get("decision_latency_ms", 1.0)

            t0 = time.perf_counter()
            gen_req = ResponseGenerationRequest(
                prompt=prompt_text,
                selected_model=model_id,
                execution_mode="local",
                max_output_tokens=600
            )
            gen_res = response_gen.generate_response(gen_req)
            lat_ms = round((time.perf_counter() - t0) * 1000, 2)

            success = gen_res.success
            gen_text = gen_res.generated_text if success else ""
            
            # Strict Local Isolation: Zero Cloud Fallback
            failure_type = None if success else (gen_res.execution_status or "OLLAMA_EXECUTION_FAILURE")

            in_tok = gen_res.input_tokens if hasattr(gen_res, "input_tokens") and gen_res.input_tokens else len(prompt_text) // 4
            out_tok = gen_res.output_tokens if hasattr(gen_res, "output_tokens") and gen_res.output_tokens else len(gen_text) // 4

            # Quality metrics evaluated ONLY on actual generated text
            q_eval = QualityMetricsEvaluator.evaluate_quality(prompt_text, cat, gen_text, success=success)

            record = {
                "run_id": "RUN_2_LOCAL_ONLY_V1",
                "experiment_condition": "LOCAL_ONLY_V1",
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
                "total_latency_ms": round(dec_lat_ms + lat_ms, 2),
                "success": success,
                "failure_type": failure_type,
                "failover_used": False,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_cost_usd": 0.0,
                "score_0_to_5": q_eval["score_0_to_5"],
                "normalized_quality": q_eval["normalized_quality"],
                "rubric_level": q_eval["rubric_level"],
                "response_text_snippet": gen_text[:200] if gen_text else "",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            results.append(record)

    logger.info(f"Finished LOCAL_ONLY_V1 Strategy {strategy_id} execution: {len(results)} records.")
    return results
